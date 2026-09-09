"""Boundary-response rank and basis-transfer gate (PREREG.md). Run:
    /path/to/.venv/bin/python run_study.py
Writes results.json and figures next to this file.
"""
from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np

from dtn2d import (assemble_dtn, block_diag, circle, ellipse, make_component, principal_part,
                   rank_at, real_fourier_matrix, star, superellipse)

HERE = Path(__file__).parent
TOLS = (1e-2, 1e-3)
K_COMMON = 48  # mode truncation for the basis-transfer registration (|k| <= 48 -> 97 real modes)
RNG = np.random.default_rng(20260909)


# ----------------------------------------------------------------------------- geometry families
def family_cases():
    cases = []
    # source: single smooth bodies
    for asp in (1.0, 1.5, 2.0, 3.0):
        cases.append(dict(name=f"ellipse_a{asp}", family="source_single", comps=[(ellipse(asp, 1.0), +1)], N=96, body=[0], L_body=2 * asp))
    for a in (0.1, 0.2, 0.3):
        for m in (3, 5):
            cases.append(dict(name=f"star_a{a}_m{m}", family="source_single", comps=[(star(a, m), +1)], N=256, body=[0], L_body=2 * (1 + a)))
    # source: wide-gap pairs (two unit circles in a bounding circle of radius 4)
    for gL in (0.5, 0.3):
        g = gL * 2.0
        cases.append(dict(name=f"pair_gL{gL}", family="source_pair", N=192, body=[1, 2], L_body=2.0,
                          comps=[(circle(4.0), +1), (circle(1.0, (-(1 + g / 2), 0.0)), -1), (circle(1.0, (1 + g / 2, 0.0)), -1)], gL=gL))
    # held out: narrow gaps
    for gL in (0.2, 0.1, 0.05, 0.02):
        g = gL * 2.0
        cases.append(dict(name=f"pair_gL{gL}", family="heldout_gap", N=384 if gL <= 0.05 else 256, body=[1, 2], L_body=2.0,
                          comps=[(circle(4.0), +1), (circle(1.0, (-(1 + g / 2), 0.0)), -1), (circle(1.0, (1 + g / 2, 0.0)), -1)], gL=gL))
    # held out: annuli (outer radius 1, inner rho)
    for rho in (0.3, 0.6, 0.85):
        cases.append(dict(name=f"annulus_r{rho}", family="heldout_annulus", N=192 if rho >= 0.85 else 96, body=[0, 1], L_body=2.0,
                          comps=[(circle(1.0), +1), (circle(rho), -1)], gL=(1 - rho) / 2))
    # held out: rounded squares (superellipse, even p)
    for p, N in ((4, 128), (8, 128), (16, 256), (32, 512)):
        rho = 1.0 / ((p - 1) * 2 ** (1.0 / p))  # curvature radius at the corner of |x|^p+|y|^p=1
        cases.append(dict(name=f"square_p{p}", family="heldout_corner", comps=[(superellipse(p), +1)], N=N, body=[0], L_body=2.0, rhoL=rho / 2.0))
    return cases


# ----------------------------------------------------------------------------- operators per case
def build(case, N):
    comps = [make_component(curve, N, sign) for curve, sign in case["comps"]]
    Lam, slices, V, K = assemble_dtn(comps)
    return comps, Lam, slices


def analyse(case, N):
    """All readouts for one case at half-resolution N. Returns dict (JSON-able) plus arrays for figures."""
    comps, Lam, slices = build(case, N)
    body = case["body"]
    idx = np.concatenate([np.arange(slices[b].start, slices[b].stop) for b in body])
    A = Lam[np.ix_(idx, idx)]                       # physically relevant block (bodies, outer grounded)
    # principal part, block-diagonal over the body components, in the arclength Fourier basis
    Ps, Qs, ks_list, Ls = [], [], [], []
    for b in body:
        n = slices[b].stop - slices[b].start
        P, Q, ks = principal_part(n, comps[b]["L"]); Ps.append(P); Qs.append(Q); ks_list.append(ks); Ls.append(comps[b]["L"])
    P = block_diag(Ps); Q = block_diag(Qs); ks_all = np.concatenate(ks_list)
    R = A - P
    # null / component modes: per-component constants on input and output
    n_b = [slices[b].stop - slices[b].start for b in body]
    C = np.zeros((len(idx), len(body))); o = 0
    for i, n in enumerate(n_b):
        C[o:o + n, i] = 1 / np.sqrt(n); o += n
    Pc = np.eye(len(idx)) - C @ C.T
    Rnn = Pc @ R @ Pc
    # energy pairing D^{-1/2} . D^{-1/2}, D = 2 pi max(|k|,1)/L_c
    dvec = np.concatenate([2 * np.pi * np.maximum(ks, 1) / L for ks, L in zip(ks_list, Ls)])
    Dm = Q.T @ np.diag(dvec ** -0.5) @ Q
    A_E, R_E, Rnn_E = Dm @ A @ Dm, Dm @ R @ Dm, Dm @ Rnn @ Dm
    out = {"N": N, "n_nodes": int(len(idx)), "n_components": len(body), "L_components": Ls}
    normA = np.linalg.norm(A); normA_E = np.linalg.norm(A_E)
    spectra = {}
    for pairing, (a, r, rn, na) in {"L2": (A, R, Rnn, normA), "energy": (A_E, R_E, Rnn_E, normA_E)}.items():
        ra, sa = rank_at(a, TOLS); rr, sr = rank_at(r, TOLS); rnn, snn = rank_at(rn, TOLS)
        # combined ranks: error relative to the FULL operator norm (principal part exact + rank-r residual)
        rr_abs, _ = rank_at(r, TOLS, ref=na); rnn_abs, _ = rank_at(rn, TOLS, ref=na)
        out[pairing] = {"rank_full": ra, "rank_residual": rr, "rank_residual_nullfree": rnn,
                        "rank_residual_rel_to_full": rr_abs, "rank_residual_nullfree_rel_to_full": rnn_abs,
                        "norm_residual_over_full": float(np.linalg.norm(r) / na), "norm_nullfree_over_full": float(np.linalg.norm(rn) / na),
                        "sv_full": sa[:64].tolist(), "sv_residual": sr[:64].tolist(), "sv_nullfree": snn[:64].tolist()}
        spectra[pairing] = (sa, sr, snn)
    # spectral point: share of ||A||_F^2 from input modes |k| > N/4 (per body component) and output modes
    Ahat = Q @ A @ Q.T
    high = ks_all > N / 4
    out["energy_share_high_modes_L2"] = {"input": float(np.sum(Ahat[:, high] ** 2) / np.sum(Ahat ** 2)), "output": float(np.sum(Ahat[high, :] ** 2) / np.sum(Ahat ** 2))}
    AEhat = Q @ A_E @ Q.T
    out["energy_share_high_modes_energy"] = {"input": float(np.sum(AEhat[:, high] ** 2) / np.sum(AEhat ** 2))}
    # blocks in truncated common mode coordinates for transfer: per body component self blocks and coupling blocks
    blocks = {}
    for pairing, (a, r, rn) in {"L2": (A, R, Rnn), "energy": (A_E, R_E, Rnn_E)}.items():
        blocks[pairing] = {}
        o_i = 0
        for i, b in enumerate(body):
            Qi = Qs[i]; ki = ks_list[i]; sel_i = np.flatnonzero(ki <= K_COMMON); n_i = n_b[i]
            o_j = 0
            for j, b2 in enumerate(body):
                Qj = Qs[j]; kj = ks_list[j]; sel_j = np.flatnonzero(kj <= K_COMMON); n_j = n_b[j]
                for name, M in (("full", a), ("residual", r), ("nullfree", rn)):
                    Mb = Qi @ M[o_i:o_i + n_i, o_j:o_j + n_j] @ Qj.T
                    blocks[pairing][(name, i, j)] = Mb[np.ix_(sel_i, sel_j)]
                o_j += n_j
            o_i += n_i
    return out, blocks, spectra, A


def common_mode_diff(case):
    outs = {}
    for N in (case["N"], 2 * case["N"]):
        comps, Lam, slices = build(case, N)
        body = case["body"]; mats = []
        for b in body:
            n = slices[b].stop - slices[b].start; Q, ks = real_fourier_matrix(n)
            mats.append((Q, ks))
        idx = np.concatenate([np.arange(slices[b].start, slices[b].stop) for b in body])
        A = Lam[np.ix_(idx, idx)]
        Qb = block_diag([m[0] for m in mats]); ks = np.concatenate([m[1] for m in mats])
        Ahat = Qb @ A @ Qb.T
        keep = np.flatnonzero(ks <= case["N"] // 2)
        outs[N] = Ahat[np.ix_(keep, keep)]
    a, b = outs[case["N"]], outs[2 * case["N"]]
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ----------------------------------------------------------------------------- transfer
def fit_basis(block_list):
    """Frozen output basis U (from horizontal stack) and input basis V (from vertical stack)."""
    H = np.concatenate(block_list, axis=1); Vt = np.concatenate(block_list, axis=0)
    U, _, _ = np.linalg.svd(H, full_matrices=False)
    _, _, Vh = np.linalg.svd(Vt, full_matrices=False)
    return U, Vh.T


def proj_err(M, U, V, r):
    Ur, Vr = U[:, :r], V[:, :r]
    return float(np.linalg.norm(M - Ur @ Ur.T @ M @ Vr @ Vr.T) / np.linalg.norm(M))


def oracle_err(M, r):
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.sqrt(np.sum(s[r:] ** 2)) / np.sqrt(np.sum(s ** 2)))


def random_probe_err(M, U, V, r, n_probe=64):
    g = RNG.standard_normal((M.shape[1], n_probe))
    Ur, Vr = U[:, :r], V[:, :r]
    return float(np.linalg.norm(M @ g - Ur @ Ur.T @ M @ Vr @ Vr.T @ g) / np.linalg.norm(M @ g))


def main():
    t0 = time.time()
    cases = family_cases()
    results = {"prereg": "PREREG.md", "K_common": K_COMMON, "tols": list(TOLS), "cases": {}}
    blocks_all = {}
    for case in cases:
        d = common_mode_diff(case)
        out, blocks, spectra, A = analyse(case, case["N"])
        out2, _, _, _ = analyse(case, 2 * case["N"])
        out["family"] = case["family"]; out["converged_rel_diff"] = d; out["qualified"] = bool(d <= 1e-6)
        out["gL"] = case.get("gL"); out["rhoL"] = case.get("rhoL")
        out["at_2N"] = {p: {k: out2[p][k] for k in ("rank_full", "rank_residual", "rank_residual_nullfree", "norm_residual_over_full", "norm_nullfree_over_full")} for p in ("L2", "energy")}
        results["cases"][case["name"]] = out
        blocks_all[case["name"]] = blocks
        print(f"{case['name']:16s} fam={case['family']:16s} N={case['N']:4d} conv={d:.1e} q={out['qualified']} | L2 ranks 1e-2 full/res/nullfree "
              f"{out['L2']['rank_full']['0.01']}/{out['L2']['rank_residual']['0.01']}/{out['L2']['rank_residual_nullfree']['0.01']} "
              f"(2N: {out2['L2']['rank_full']['0.01']}/{out2['L2']['rank_residual']['0.01']}/{out2['L2']['rank_residual_nullfree']['0.01']}) | energy "
              f"{out['energy']['rank_full']['0.01']}/{out['energy']['rank_residual']['0.01']}/{out['energy']['rank_residual_nullfree']['0.01']} "
              f"| |R|/|A| {out['L2']['norm_residual_over_full']:.3f} nullfree {out['L2']['norm_nullfree_over_full']:.3f} | high-mode share {out['energy_share_high_modes_L2']['input']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    # ---- transfer: source basis from single-body self blocks + wide-pair self blocks
    results["transfer"] = {}
    for pairing in ("L2", "energy"):
        for obj in ("full", "residual", "nullfree"):
            src = [blocks_all[c["name"]][pairing][(obj, i, i)] for c in cases if c["family"].startswith("source") for i in range(len(c["body"]))]
            U, V = fit_basis(src)
            rows = {}
            for c in cases:
                if c["family"].startswith("source"):
                    continue
                bl = blocks_all[c["name"]][pairing]
                for (name, i, j), M in bl.items():
                    if name != obj:
                        continue
                    # rank: oracle rank for 1e-2 of THIS block, capped at the common mode count
                    ro, _ = rank_at(M, (1e-2,)); r = max(int(ro["0.01"]), 1); r = min(r, U.shape[1], V.shape[1])
                    rows[f"{c['name']}[{i},{j}]"] = {"rank": r, "oracle_err": oracle_err(M, r), "source_basis_err": proj_err(M, U, V, r),
                                                    "random_probe_err": random_probe_err(M, U, V, r), "kind": "self" if i == j else "coupling", "family": c["family"]}
            results["transfer"][f"{pairing}/{obj}"] = rows
            worst = max(v["source_basis_err"] / max(v["oracle_err"], 1e-12) for v in rows.values())
            print(f"transfer {pairing}/{obj}: worst source/oracle ratio {worst:.2f}; median source err {np.median([v['source_basis_err'] for v in rows.values()]):.3f}", flush=True)
    results["seconds"] = time.time() - t0
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print("wrote results.json in", round(results["seconds"]), "s")


if __name__ == "__main__":
    main()
