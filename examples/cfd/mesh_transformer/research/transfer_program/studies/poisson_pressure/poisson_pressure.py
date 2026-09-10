# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""H1 study: recover pressure from a (true or predicted) velocity field by
solving the constraint it must satisfy, on the saved DrivAerML volume
predictions (10,000 scattered interior points per car).

Eval-only, CPU (numpy/scipy). See PREREG.md for the equation, the
discretisation and the gates. Usage:

    python poisson_pressure.py --evals <dir with run dirs> --out results.json
        [--runs v0_isla_qtsdfval_seed42 v0_gt_vol_seed42] [--cases N]
        [--ops 40:2 60:2 60:3] [--main 40:2] [--workers 16] [--dump-case 00015]

Operators are named "<k>:<deg>" (stencil size : polynomial degree). Stages
per case: G0 (operator on analytic fields, every operator), G1 (recover from
the TRUE fields, every formulation / BC / wall-set variant, main operator and
the others as sensitivity), G2 (recover from each arm's PREDICTED fields,
every variant, main operator; the prereg decides which one counts).
Everything is written to one JSON; a small NPZ of one case's fields is
written for the figure.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

DT = {"torch.float32": np.float32, "torch.float64": np.float64, "torch.int64": np.int64}
FAR_FRAC_L = 0.4  # far set: SDF >= 0.4 L_ref (p = p_inf imposed there)


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------
def read_td(dd: Path, name: str) -> np.ndarray:
    meta = json.load(open(dd / "meta.json"))[name]
    return np.array(
        np.memmap(dd / f"{name}.memmap", dtype=DT[meta["dtype"]], mode="r", shape=tuple(meta["shape"]))
    )


def read_global(dd: Path) -> dict:
    meta = json.load(open(dd / "meta.json"))
    out = {}
    for k, m in meta.items():
        if isinstance(m, dict) and "shape" in m and (dd / f"{k}.memmap").exists():
            out[k] = np.array(
                np.memmap(dd / f"{k}.memmap", dtype=DT[m["dtype"]], mode="r", shape=tuple(m["shape"]) or (1,))
            )
    return out


def load_case(case_dir: Path) -> dict:
    td = case_dir / "_tensordict"
    pts = read_td(td / "interior" / "_tensordict", "points").astype(np.float64)
    pd = td / "interior" / "_tensordict" / "point_data"
    f = {
        "points": pts,
        "sdf": read_td(pd, "sdf").reshape(-1).astype(np.float64),
        "sdf_normals": read_td(pd, "sdf_normals").astype(np.float64),
        "p_true": read_td(pd, "true_pressure").reshape(-1).astype(np.float64),
        "p_pred": read_td(pd, "pred_pressure").reshape(-1).astype(np.float64),
        "u_true": read_td(pd, "true_velocity").astype(np.float64),
        "u_pred": read_td(pd, "pred_velocity").astype(np.float64),
        "nut_true": read_td(pd, "true_nut").reshape(-1).astype(np.float64),
        "nut_pred": read_td(pd, "pred_nut").reshape(-1).astype(np.float64),
    }
    gd = td / "global_data"
    g = read_global(gd / "_tensordict" if (gd / "_tensordict" / "meta.json").exists() else gd)
    f["rho"] = float(g.get("rho_inf", np.array([1.0])).reshape(-1)[0])
    f["p_inf"] = float(g.get("p_inf", np.array([0.0])).reshape(-1)[0])
    f["L_ref"] = float(g.get("L_ref", np.array([5.0])).reshape(-1)[0])
    return f


# ----------------------------------------------------------------------------
# RBF-FD operators (PHS r^3 + degree-`deg` polynomials) on kNN stencils
# ----------------------------------------------------------------------------
def monomial_exponents(deg: int) -> np.ndarray:
    return np.array([e for e in itertools.product(range(deg + 1), repeat=3) if sum(e) <= deg])


def poly_eval(X: np.ndarray, E: np.ndarray) -> np.ndarray:
    """[..., 3] -> [..., M] monomials with exponents E [M,3]."""
    return np.prod(X[..., None, :] ** E[None, :, :], axis=-1) if X.ndim == 2 else np.prod(
        X[..., None, :] ** E[(None,) * (X.ndim - 1) + (slice(None), slice(None))], axis=-1
    )


def poly_ops_at_zero(E: np.ndarray) -> np.ndarray:
    """Values at x=0 of (d/dx,d/dy,d/dz,dxx,dyy,dzz,dxy,dxz,dyz) applied to each monomial -> [9, M]."""
    M = len(E)
    out = np.zeros((9, M))
    for m, e in enumerate(E):
        e = tuple(e)
        for a in range(3):
            if e == tuple(1 if i == a else 0 for i in range(3)):
                out[a, m] = 1.0
        for a, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]):
            target = [0, 0, 0]
            target[i] += 1
            target[j] += 1
            if e == tuple(target):
                out[3 + a, m] = 2.0 if i == j else 1.0
    return out


def rbf_fd(pts: np.ndarray, k: int, deg: int, chunk: int = 1000):
    """Return neighbour idx [N,k], stencil radius h [N], and weights W [N,9,k]
    for (dx,dy,dz,dxx,dyy,dzz,dxy,dxz,dyz) in physical units."""
    n = len(pts)
    E = monomial_exponents(deg)
    M = len(E)
    POPS = poly_ops_at_zero(E)
    tree = cKDTree(pts)
    d, idx = tree.query(pts, k)
    h = d[:, -1]
    W = np.empty((n, 9, k))
    n_fallback = 0
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        c = e - s
        X = (pts[idx[s:e]] - pts[s:e, None, :]) / h[s:e, None, None]  # [c,k,3] scaled local coords
        D = X[:, :, None, :] - X[:, None, :, :]
        r = np.linalg.norm(D, axis=-1)  # [c,k,k]
        P = np.prod(X[:, :, None, :] ** E[None, None, :, :], axis=-1)  # [c,k,M]
        A = np.zeros((c, k + M, k + M))
        A[:, :k, :k] = r**3
        A[:, :k, k:] = P
        A[:, k:, :k] = np.transpose(P, (0, 2, 1))
        rj = np.linalg.norm(X, axis=-1)  # [c,k]
        rjs = np.where(rj > 0, rj, 1.0)
        B = np.zeros((c, k + M, 9))
        B[:, :k, 0:3] = -3.0 * rj[..., None] * X  # gradient of |x - x_j|^3 at x = 0
        for a, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]):
            B[:, :k, 3 + a] = 3.0 * ((rj if i == j else 0.0) + X[..., i] * X[..., j] / rjs) * (rj > 0)
        B[:, k:, :] = POPS.T[None]
        try:
            sol = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            sol = np.empty_like(B)
            for q in range(c):
                try:
                    sol[q] = np.linalg.solve(A[q], B[q])
                except np.linalg.LinAlgError:
                    sol[q] = np.linalg.lstsq(A[q], B[q], rcond=None)[0]
                    n_fallback += 1
        w = np.transpose(sol[:, :k, :], (0, 2, 1))  # [c,9,k]
        w[:, 0:3] /= h[s:e, None, None]
        w[:, 3:9] /= h[s:e, None, None] ** 2
        W[s:e] = w
    return idx, h, W, n_fallback


def to_sparse(idx: np.ndarray, w: np.ndarray) -> sp.csr_matrix:
    n, k = idx.shape
    rows = np.repeat(np.arange(n), k)
    return sp.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))


class Ops:
    def __init__(self, pts, k, deg):
        t0 = time.time()
        self.idx, self.h, W, self.n_fallback = rbf_fd(pts, k, deg)
        self.G = [to_sparse(self.idx, W[:, a]) for a in range(3)]
        self.L = to_sparse(self.idx, W[:, 3] + W[:, 4] + W[:, 5])
        self.t_build = time.time() - t0
        self.k, self.deg = k, deg

    def grad(self, f):  # scalar [N] -> [N,3]
        return np.stack([g @ f for g in self.G], axis=-1)

    def jac(self, u):  # vector [N,3] -> J[N,i,j] = d_i u_j
        return np.stack([np.stack([self.G[i] @ u[:, j] for j in range(3)], -1) for i in range(3)], 1)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def rel_l2(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    if len(b) == 0:
        return float("nan")
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def rel_l2_frob(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    if len(b) == 0:
        return float("nan")
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-300))


# ----------------------------------------------------------------------------
# G0: analytic fields
# ----------------------------------------------------------------------------
def g0_tests(ops: Ops, pts, bands: dict, lams=(1.0, 0.25)) -> dict:
    out = {}
    for lam in lams:
        a = 2 * np.pi / lam
        s, c = np.sin(a * pts), np.cos(a * pts)
        f = s[:, 0] * s[:, 1] * s[:, 2]
        gf = a * np.stack([c[:, 0] * s[:, 1] * s[:, 2], s[:, 0] * c[:, 1] * s[:, 2], s[:, 0] * s[:, 1] * c[:, 2]], -1)
        lf = -3 * a * a * f
        Gf, Lf = ops.grad(f), ops.L @ f
        r = {}
        for bname, m in bands.items():
            r[bname] = {"grad": rel_l2_frob(Gf, gf, m), "lap": rel_l2(Lf, lf, m), "n": int(m.sum())}
        out[f"lam{lam:g}"] = r
    return out


# ----------------------------------------------------------------------------
# boundary sets
# ----------------------------------------------------------------------------
def boundary_sets(pts, sdf, h, L_ref):
    """far (Dirichlet p = p_inf), hull points near the sample's bounding box
    (with outward face normals), plus the ground band (lowest face)."""
    n = len(pts)
    far = sdf >= FAR_FRAC_L * L_ref
    far_mode = "sdf>=0.4L"
    if far.sum() < 10:
        far = sdf >= np.quantile(sdf, 0.99)
        far_mode = "top1%"
    lo, hi = pts.min(0), pts.max(0)
    d_lo, d_hi = pts - lo, hi - pts  # [n,3]
    dist = np.concatenate([d_lo, d_hi], 1)  # [n,6]
    face = dist.argmin(1)
    fdist = dist.min(1)
    hull = fdist < 0.5 * h
    normals = np.zeros((n, 3))
    for f in range(6):
        ax, sign = f % 3, (-1.0 if f < 3 else 1.0)
        normals[face == f, ax] = sign
    return far, far_mode, hull, normals


# ----------------------------------------------------------------------------
# pressure recovery
# ----------------------------------------------------------------------------
def sources(ops: Ops, u, nut, rho):
    """Return dict with S_A, S_B (Poisson sources), f_A, f_B (momentum RHS
    for grad p), and the divergence of u."""
    J = ops.jac(u)  # J[n,i,j] = d_i u_j
    div = J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]
    S_A = -rho * np.einsum("nij,nji->n", J, J)
    f_A = -rho * np.einsum("ni,nij->nj", u, J)  # -(u . grad) u
    T = nut[:, None, None] * (J + np.transpose(J, (0, 2, 1)))  # nu_t (d_i u_j + d_j u_i)
    V = np.stack([sum(ops.G[i] @ T[:, i, j] for i in range(3)) for j in range(3)], -1)  # V_j = d_i T_ij
    S_B = S_A + rho * sum(ops.G[j] @ V[:, j] for j in range(3))
    f_B = f_A + rho * V
    return {"S_A": S_A, "S_B": S_B, "f_A": f_A, "f_B": f_B, "div": div, "J": J}


def solve_poisson(ops: Ops, S, bnd, far, bc, normals, p_bnd=None, g_bnd=None):
    """Rows: interior L p = S; far: p = 0; boundary (wall + hull): BC in {'D','N0','Nm'}."""
    n = len(S)
    far_idx = np.where(far)[0]
    bnd_idx = np.where(bnd & ~far)[0]
    keep = np.ones(n, bool)
    keep[far_idx] = False
    keep[bnd_idx] = False
    Dk = sp.diags(keep.astype(float))
    A = Dk @ ops.L
    b = np.where(keep, S, 0.0)
    Ifar = sp.csr_matrix((np.ones(len(far_idx)), (far_idx, far_idx)), shape=(n, n))
    A = A + Ifar
    if bc == "D":
        Ib = sp.csr_matrix((np.ones(len(bnd_idx)), (bnd_idx, bnd_idx)), shape=(n, n))
        A = A + Ib
        b[bnd_idx] = p_bnd[bnd_idx]
    else:
        sel = np.zeros(n)
        sel[bnd_idx] = 1.0
        Nrows = sp.diags(sel) @ sum(sp.diags(normals[:, i]) @ ops.G[i] for i in range(3))
        A = A + Nrows
        b[bnd_idx] = 0.0 if bc == "N0" else g_bnd[bnd_idx]
    p = spla.spsolve(A.tocsc(), b)
    return p


def solve_momentum_ls(ops: Ops, f, far):
    """min |G p - f|^2 subject (by weight) to mean(p[far]) = 0 (normal equations, SuperLU)."""
    n = len(f)
    Gs = sp.vstack(ops.G).tocsr()
    rown = np.sqrt(np.asarray(Gs.multiply(Gs).sum(1)).ravel().mean())
    far_idx = np.where(far)[0]
    row = sp.csr_matrix(
        (np.full(len(far_idx), 100.0 * rown / len(far_idx)), (np.zeros(len(far_idx), int), far_idx)), shape=(1, n)
    )
    A = sp.vstack([Gs, row]).tocsr()
    b = np.concatenate([f.T.ravel(), [0.0]])
    return spla.spsolve((A.T @ A).tocsc(), A.T @ b)


def recover_all(ops: Ops, fld, u, nut, wall_sets, far, hull, hull_normals, bands, full=True):
    """Run every variant for one velocity/nu_t pair; return metrics + fields."""
    rho = fld["rho"]
    p_true = fld["p_true"] - fld["p_inf"]
    src = sources(ops, u, nut, rho)
    wn = fld["sdf_normals"]
    nn = np.linalg.norm(wn, axis=1, keepdims=True)
    wn = wn / np.where(nn > 0, nn, 1.0)
    res, fields = {}, {}
    res["div_rel"] = float(np.linalg.norm(src["div"]) / np.linalg.norm(src["J"].reshape(len(u), -1)))
    t0 = time.time()

    def record(name, p):
        r = {b: rel_l2(p, p_true, m) for b, m in bands.items()}
        r["offset_only"] = rel_l2(p - (p - p_true).mean(), p_true)
        r["finite"] = bool(np.all(np.isfinite(p)))
        res[name] = r
        fields[name] = p

    for rhs in ("A", "B"):
        S, f = src[f"S_{rhs}"], src[f"f_{rhs}"]
        t1 = time.time()
        record(f"M{rhs}", solve_momentum_ls(ops, f, far))
        res[f"M{rhs}"]["t"] = time.time() - t1
        for wname, wall in wall_sets.items():
            bnd = wall | hull
            normals = np.where(wall[:, None], wn, hull_normals)
            g_bnd = np.einsum("ni,ni->n", normals, f)
            for bc in ("D", "N0", "Nm"):
                if not full and (bc == "Nm" or wname != "q05"):
                    continue
                t1 = time.time()
                record(f"P{rhs}-{bc}-{wname}", solve_poisson(ops, S, bnd, far, bc, normals, p_bnd=p_true, g_bnd=g_bnd))
                res[f"P{rhs}-{bc}-{wname}"]["t"] = time.time() - t1
    res["t_solve_all"] = time.time() - t0
    return res, fields


# ----------------------------------------------------------------------------
# per case
# ----------------------------------------------------------------------------
def parse_op(s):
    k, d = s.split(":")
    return int(k), int(d)


def process_case(job):
    case, run_dirs, runs, op_specs, main_spec, dump_path = job
    t_case = time.time()
    flds = {r: load_case(run_dirs[r] / case) for r in runs}
    f0 = flds[runs[0]]
    pts, sdf, L_ref = f0["points"], f0["sdf"], f0["L_ref"]
    n = len(pts)
    same_true = all(
        np.allclose(flds[r]["p_true"], f0["p_true"]) and np.allclose(flds[r]["u_true"], f0["u_true"])
        and np.allclose(flds[r]["points"], f0["points"]) for r in runs
    )
    bands = {
        "all": np.ones(n, bool),
        "sdf<0.01L": sdf < 0.01 * L_ref,
        "sdf<0.05L": sdf < 0.05 * L_ref,
        "sdf>=0.05L": sdf >= 0.05 * L_ref,
    }
    tree = cKDTree(pts)
    d1 = tree.query(pts, 2)[0][:, 1]
    p_true = f0["p_true"] - f0["p_inf"]
    cres = {
        "same_true_across_runs": bool(same_true),
        "n": n,
        "rho": f0["rho"], "p_inf": f0["p_inf"], "L_ref": L_ref,
        "bbox": [pts.min(0).tolist(), pts.max(0).tolist()],
        "sdf_quantiles_05_50_90_98_100": [float(q) for q in np.quantile(sdf, [0.05, 0.5, 0.9, 0.98, 1.0])],
        "nn_spacing_quantiles_10_50_90_99": [float(q) for q in np.quantile(d1, [0.1, 0.5, 0.9, 0.99])],
        "band_counts": {b: int(m.sum()) for b, m in bands.items()},
        "p_true_rms": float(np.sqrt(np.mean(p_true**2))),
        "g0": {}, "g1": {}, "g2": {}, "direct": {},
    }
    ops_by = {}
    for spec in op_specs:
        k, deg = parse_op(spec)
        ops = Ops(pts, k, deg)
        ops_by[spec] = ops
        g0 = g0_tests(ops, pts, bands)
        g0["n_fallback"] = ops.n_fallback
        g0["t_build"] = ops.t_build
        g0["h_quantiles_10_50_90"] = [float(q) for q in np.quantile(ops.h, [0.1, 0.5, 0.9])]
        cres["g0"][spec] = g0
    ops = ops_by[main_spec]
    far, far_mode, hull, hull_normals = boundary_sets(pts, sdf, ops.h, L_ref)
    cres["far"] = {"mode": far_mode, "n": int(far.sum()),
                   "p_true_rms_far_over_all": float(np.sqrt(np.mean(p_true[far] ** 2)) / cres["p_true_rms"])}
    cres["hull_n"] = int(hull.sum())
    wall_sets = {"q05": sdf < np.quantile(sdf, 0.05), "halfh": sdf < 0.5 * ops.h}
    cres["wall_counts"] = {w: int(m.sum()) for w, m in wall_sets.items()}

    # G1: true fields, main operator (all variants) and other operators (reduced)
    g1, fields_true = recover_all(ops, f0, f0["u_true"], f0["nut_true"], wall_sets, far, hull, hull_normals, bands)
    cres["g1"][main_spec] = g1
    for spec, ops_s in ops_by.items():
        if spec == main_spec:
            continue
        far_s, _, hull_s, hn_s = boundary_sets(pts, sdf, ops_s.h, L_ref)
        ws_s = {"q05": wall_sets["q05"]}
        g1s, _ = recover_all(ops_s, f0, f0["u_true"], f0["nut_true"], ws_s, far_s, hull_s, hn_s, bands, full=False)
        cres["g1"][spec] = g1s

    # G2: predicted fields per arm, main operator
    fields_pred = {}
    for r in runs:
        fr = flds[r]
        direct = {b: rel_l2(fr["p_pred"] - fr["p_inf"], p_true, m) for b, m in bands.items()}
        direct["offset_only"] = rel_l2(fr["p_pred"] - fr["p_inf"] - (fr["p_pred"] - fr["p_inf"] - p_true).mean(), p_true)
        direct["velocity"] = rel_l2_frob(fr["u_pred"], fr["u_true"])
        direct["nut"] = rel_l2(fr["nut_pred"], fr["nut_true"])
        cres["direct"][r] = direct
        g2, fp = recover_all(ops, fr, fr["u_pred"], fr["nut_pred"], wall_sets, far, hull, hull_normals, bands)
        cres["g2"][r] = g2
        fields_pred[r] = fp
    cres["t_case"] = time.time() - t_case
    if dump_path is not None:
        np.savez_compressed(
            dump_path, points=pts, sdf=sdf, p_true=p_true,
            **{f"true_{k}": v for k, v in fields_true.items()},
            **{f"{r}_direct": flds[r]["p_pred"] - flds[r]["p_inf"] for r in runs},
            **{f"{r}_{k}": v for r, fp in fields_pred.items() for k, v in fp.items()},
        )
    return case, cres


# ----------------------------------------------------------------------------
# aggregate
# ----------------------------------------------------------------------------
def aggregate(cases: dict, runs) -> dict:
    C = list(cases.values())

    def mean_over(path):
        vals = []
        for c in C:
            v = c
            for p in path:
                v = v[p]
            vals.append(v)
        vals = np.array(vals, float)
        return {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "median": float(np.nanmedian(vals)),
                "max": float(np.nanmax(vals))}

    agg = {"g0": {}, "g1": {}, "g2": {}, "direct": {}, "n_cases": len(C)}
    for spec, g0 in C[0]["g0"].items():
        agg["g0"][spec] = {}
        for lam in g0:
            if not lam.startswith("lam"):
                continue
            agg["g0"][spec][lam] = {b: {m: mean_over(["g0", spec, lam, b, m]) for m in ("grad", "lap")} for b in g0[lam]}
    for spec, g1 in C[0]["g1"].items():
        agg["g1"][spec] = {name: {b: mean_over(["g1", spec, name, b]) for b in v if b not in ("finite",)}
                           for name, v in g1.items() if isinstance(v, dict)}
        agg["g1"][spec]["div_rel"] = mean_over(["g1", spec, "div_rel"])
    for r in runs:
        agg["direct"][r] = {b: mean_over(["direct", r, b]) for b in C[0]["direct"][r]}
        agg["g2"][r] = {name: {b: mean_over(["g2", r, name, b]) for b in v if b not in ("finite",)}
                        for name, v in C[0]["g2"][r].items() if isinstance(v, dict)}
        agg["g2"][r]["div_rel"] = mean_over(["g2", r, "div_rel"])
    agg["far_p_rms_ratio"] = mean_over(["far", "p_true_rms_far_over_all"])
    agg["far_n"] = mean_over(["far", "n"])
    agg["hull_n"] = mean_over(["hull_n"])
    agg["t_case"] = mean_over(["t_case"])
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True)
    ap.add_argument("--runs", nargs="+", default=["v0_isla_qtsdfval_seed42", "v0_gt_vol_seed42"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", type=int, default=0, help="limit number of cases (0 = all)")
    ap.add_argument("--ops", nargs="+", default=["40:2", "60:2", "60:3"])
    ap.add_argument("--main", default="40:2")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dump-case", default="00015")
    args = ap.parse_args()

    evals = Path(args.evals)
    run_dirs = {r: evals / r / r / "predictions" for r in args.runs}
    cases = sorted(p.name for p in run_dirs[args.runs[0]].iterdir() if p.name.endswith(".pdmsh"))
    if args.cases:
        cases = cases[: args.cases]
    ops = list(dict.fromkeys(args.ops + [args.main]))
    print(f"{len(cases)} cases, runs {args.runs}, ops {ops}, main {args.main}, workers {args.workers}", flush=True)
    out = Path(args.out)
    jobs = [(c, run_dirs, args.runs, ops, args.main,
             out.with_suffix(".case.npz") if c.startswith(args.dump_case) else None) for c in cases]
    results = {"args": vars(args), "cases": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (case, cres) in enumerate(pool.imap_unordered(process_case, jobs)):
            results["cases"][case] = cres
            g1 = cres["g1"][args.main]
            g0 = cres["g0"][args.main]["lam1"]["sdf<0.05L"]
            print(
                f"[{i+1}/{len(cases)}] {case} G0(lam1,sdf<0.05L) grad={g0['grad']:.3f} lap={g0['lap']:.3f} | "
                f"G1 MA={g1['MA']['all']:.3f} PA-D-q05={g1['PA-D-q05']['all']:.3f} PA-N0-q05={g1['PA-N0-q05']['all']:.3f} "
                f"| far n={cres['far']['n']} hull n={cres['hull_n']} | {cres['t_case']:.0f}s (wall {time.time()-t0:.0f}s)",
                flush=True,
            )
            json.dump(results, open(out, "w"), indent=1)
    results["cases"] = dict(sorted(results["cases"].items()))
    results["aggregate"] = aggregate(results["cases"], args.runs)
    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(results, open(out, "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
