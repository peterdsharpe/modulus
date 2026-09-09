"""Composition gate (nonlearned): exact and reduced port responses in 2D diffusion.

Preregistration: PREREG.md (same directory). Run:
    /home/psharpe/gh/physicsnemo/.venv/bin/python composition_study.py [--quick]
Writes results.json and figures fig_*.png next to this file.

Model problem: -div(kappa grad u) = 0 (optionally + c u) on unit-square components,
P1 finite elements on a structured triangulation (N_SUB subdivisions per side), glued
into chains and grids. Each component's response is its Schur complement onto the
perimeter nodes (port-to-port Dirichlet-to-Neumann map). The assembly is solved as an
interface problem S_II u_I = -S_IE g and compared with a monolithic FEM solve.
"""
import json, math, os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = "--quick" in sys.argv
N_SUB = 16
LIBRARY = ["plain", "contrast", "hole", "channel_wide", "channel_narrow"]
NEW_SHAPES = ["hole_offset", "diag_barrier", "checker8"]
rng_global = np.random.default_rng(20260909)


# ----------------------------------------------------------------------------- mesh / FEM
def unit_mesh(n):
    xs = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="xy")  # X[j,i] = xs[i]
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)  # id = j*(n+1)+i
    ij = np.array([(i, j) for j in range(n + 1) for i in range(n + 1)], dtype=int)
    tris = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i; b = a + 1; c = a + (n + 1); d = c + 1
            tris.append((a, b, c)); tris.append((b, d, c))
    return nodes, ij, np.array(tris, dtype=int)


def p1_matrices(nodes, tris, kappa, c=0.0):
    """Stiffness with per-element kappa plus c * consistent mass, over all nodes."""
    nn = len(nodes)
    rows, cols, vals = [], [], []
    for t, (a, b, cc) in enumerate(tris):
        P = nodes[[a, b, cc]]
        B = np.array([[P[1, 1] - P[2, 1], P[2, 1] - P[0, 1], P[0, 1] - P[1, 1]],
                      [P[2, 0] - P[1, 0], P[0, 0] - P[2, 0], P[1, 0] - P[0, 0]]])
        det = (P[1, 0] - P[0, 0]) * (P[2, 1] - P[0, 1]) - (P[2, 0] - P[0, 0]) * (P[1, 1] - P[0, 1])
        area = 0.5 * abs(det)
        K = kappa[t] * (B.T @ B) / (4.0 * area)
        M = c * area / 12.0 * (np.ones((3, 3)) + np.eye(3))
        loc = K + M
        for r in range(3):
            for s in range(3):
                rows.append((a, b, cc)[r]); cols.append((a, b, cc)[s]); vals.append(loc[r, s])
    return sp.csr_matrix((vals, (rows, cols)), shape=(nn, nn))


def kappa_and_mask(kind, tris, nodes, rng):
    cent = nodes[tris].mean(axis=1)
    x, y = cent[:, 0], cent[:, 1]
    kappa = np.ones(len(tris)); active = np.ones(len(tris), dtype=bool)
    if kind == "plain":
        pass
    elif kind == "contrast":
        blocks = 10.0 ** rng.uniform(-1, 1, size=(4, 4))
        kappa = blocks[np.minimum((y * 4).astype(int), 3), np.minimum((x * 4).astype(int), 3)]
    elif kind == "hole":
        active = ~((x > 0.3) & (x < 0.7) & (y > 0.3) & (y < 0.7))
    elif kind == "hole_offset":
        active = ~((x > 0.55) & (x < 0.9) & (y > 0.1) & (y < 0.45))
    elif kind in ("channel_wide", "channel_narrow"):
        h = 0.25 if kind == "channel_wide" else 1.0 / N_SUB
        barrier = (x > 0.45) & (x < 0.55) & ~(np.abs(y - 0.5) < h / 2)
        kappa = np.where(barrier, 1e-3, 1.0)
    elif kind == "diag_barrier":
        barrier = (np.abs(x - y) < 0.07) & ~(np.abs(x + y - 1.0) < 0.15)
        kappa = np.where(barrier, 1e-3, 1.0)
    elif kind == "checker8":
        cb = ((x * 8).astype(int) + (y * 8).astype(int)) % 2
        kappa = np.where(cb == 0, 0.1, 10.0)
    else:
        raise ValueError(kind)
    return kappa, active


class Component:
    def __init__(self, kind, rng, n=N_SUB, c=0.0):
        self.kind = kind; self.n = n
        self.nodes, self.ij, tris_all = unit_mesh(n)
        kappa, active = kappa_and_mask(kind, tris_all, self.nodes, rng)
        self.tris = tris_all[active]; self.kappa = kappa[active]
        self.A = p1_matrices(self.nodes, self.tris, self.kappa, c)
        used = np.zeros(len(self.nodes), dtype=bool); used[self.tris.ravel()] = True
        self.dofs = np.flatnonzero(used)
        i, j = self.ij[:, 0], self.ij[:, 1]
        perim = (i == 0) | (i == n) | (j == 0) | (j == n)
        self.B = np.flatnonzero(perim & used); self.I = np.flatnonzero(~perim & used)
        A_II = self.A[self.I][:, self.I].tocsc(); A_IB = self.A[self.I][:, self.B].toarray(); A_BB = self.A[self.B][:, self.B].toarray()
        self.lu_II = spla.splu(A_II)
        self.A_IB = A_IB
        X = self.lu_II.solve(A_IB)  # A_II^{-1} A_IB
        self.S = A_BB - A_IB.T @ X
        self.S = 0.5 * (self.S + self.S.T)

    def interior(self, u_B):
        return -self.lu_II.solve(self.A_IB @ u_B)


def response_checks(comp):
    S = comp.S; sym = norm(S - S.T) / norm(S)
    w = np.linalg.eigvalsh(S)
    return {"sym_rel": float(sym), "min_eig_over_norm": float(w[0] / abs(w).max()),
            "null_vector_residual": float(norm(S @ np.ones(len(S))) / norm(S)), "size": int(len(S))}


def norm(x):
    return float(np.linalg.norm(x))


# ----------------------------------------------------------------------------- assemblies
class Assembly:
    """Components placed on integer cells (ci, cj) of a full rectangle W x H."""

    def __init__(self, layout, kinds, rng, c=0.0):
        n = N_SUB
        self.layout = layout; self.kinds = kinds
        self.comps = [Component(k, rng, n, c) for k in kinds]
        W = max(ci for ci, _ in layout) + 1; H = max(cj for _, cj in layout) + 1
        self.W, self.H = W, H
        key2gid = {}; self.gmap = []
        for (ci, cj), comp in zip(layout, self.comps):
            gm = np.empty(len(comp.nodes), dtype=int)
            for lid in comp.dofs:
                key = (ci * n + comp.ij[lid, 0], cj * n + comp.ij[lid, 1])
                if key not in key2gid:
                    key2gid[key] = len(key2gid)
                gm[lid] = key2gid[key]
            self.gmap.append(gm)
        self.ndof = len(key2gid)
        self.coords = np.zeros((self.ndof, 2))
        for key, gid in key2gid.items():
            self.coords[gid] = (key[0] / n, key[1] / n)
        # monolithic matrix
        A = sp.csr_matrix((self.ndof, self.ndof))
        for comp, gm in zip(self.comps, self.gmap):
            loc = comp.A[comp.dofs][:, comp.dofs].tocoo()
            A = A + sp.csr_matrix((loc.data, (gm[comp.dofs][loc.row], gm[comp.dofs][loc.col])), shape=(self.ndof, self.ndof))
        self.A = A.tocsr()
        x, y = self.coords[:, 0], self.coords[:, 1]
        outer = (np.abs(x) < 1e-12) | (np.abs(x - W) < 1e-12) | (np.abs(y) < 1e-12) | (np.abs(y - H) < 1e-12)
        self.E = np.flatnonzero(outer)
        perim_g = np.unique(np.concatenate([gm[comp.B] for comp, gm in zip(self.comps, self.gmap)]))
        self.P = perim_g
        self.Iint = np.setdiff1d(perim_g, self.E)  # interface unknowns
        self.U = np.setdiff1d(np.arange(self.ndof), self.E)  # all monolithic unknowns
        # vertex / edge partition of the interface (component corners vs edge interiors)
        corner = np.zeros(self.ndof, dtype=bool)
        for comp, gm in zip(self.comps, self.gmap):
            cn = [lid for lid in comp.B if comp.ij[lid, 0] in (0, n) and comp.ij[lid, 1] in (0, n)]
            corner[gm[cn]] = True
        self.I_vertex = self.Iint[corner[self.Iint]]
        self.edges = []  # list of arrays of edge-interior global ids, oriented by increasing coordinate
        seen = set()
        for (ci, cj), comp, gm in zip(layout, self.comps, self.gmap):
            for side in ("right", "top"):
                if side == "right":
                    nb = (ci + 1, cj); lids = [lid for lid in comp.B if comp.ij[lid, 0] == n and 0 < comp.ij[lid, 1] < n]
                    lids.sort(key=lambda l: comp.ij[l, 1])
                else:
                    nb = (ci, cj + 1); lids = [lid for lid in comp.B if comp.ij[lid, 1] == n and 0 < comp.ij[lid, 0] < n]
                    lids.sort(key=lambda l: comp.ij[l, 0])
                if nb in layout:
                    g = gm[lids]
                    if tuple(g) not in seen:
                        seen.add(tuple(g)); self.edges.append(g)
        self.rng = rng

    def dirichlet(self, rng):
        x, y = self.coords[:, 0] / self.W, self.coords[:, 1] / self.H
        g = rng.normal() + rng.normal() * x + rng.normal() * y
        for _ in range(3):
            w = rng.uniform(1.0, 3.0, size=2) * math.pi; ph = rng.uniform(0, 2 * math.pi)
            g = g + rng.normal() * np.sin(w[0] * x + w[1] * y + ph)
        return g

    def solve_monolithic(self, g):
        u = np.zeros(self.ndof); u[self.E] = g[self.E]
        A_UU = self.A[self.U][:, self.U].tocsc(); A_UE = self.A[self.U][:, self.E]
        u[self.U] = spla.splu(A_UU).solve(-A_UE @ g[self.E])
        return u

    def global_S(self, S_list=None):
        S_list = S_list or [c.S for c in self.comps]
        S = np.zeros((self.ndof, self.ndof))
        for comp, gm, Sc in zip(self.comps, self.gmap, S_list):
            gb = gm[comp.B]
            S[np.ix_(gb, gb)] += Sc
        return S

    def solve_schur(self, g, S_list=None, P=None):
        """Interface solve; P (ndof_I x r) restricts interface unknowns to a subspace (Galerkin)."""
        S = self.global_S(S_list)
        S_II = S[np.ix_(self.Iint, self.Iint)]; S_IE = S[np.ix_(self.Iint, self.E)]
        rhs = -S_IE @ g[self.E]
        if P is None:
            u_I = np.linalg.solve(S_II, rhs)
        else:
            u_I = P @ np.linalg.solve(P.T @ S_II @ P, P.T @ rhs)
        u = np.zeros(self.ndof); u[self.E] = g[self.E]; u[self.Iint] = u_I
        for comp, gm in zip(self.comps, self.gmap):
            u[gm[comp.I]] = comp.interior(u[gm[comp.B]])
        return u, S_II

    def errors(self, u, u_ref):
        e = u - u_ref
        en = math.sqrt(max(e @ (self.A @ e), 0.0)) / math.sqrt(u_ref @ (self.A @ u_ref))
        l2 = norm(e) / norm(u_ref - u_ref.mean())
        return {"energy": float(en), "l2": float(l2)}

    def reduction_matrix(self, Phi, r):
        """Block basis: identity on interface vertices, Phi[:, :r] on each shared edge's interior nodes."""
        cols = []
        nI = len(self.Iint); pos = {g: k for k, g in enumerate(self.Iint)}
        for v in self.I_vertex:
            col = np.zeros(nI); col[pos[v]] = 1.0; cols.append(col)
        for e in self.edges:
            idx = [pos[g] for g in e]
            for k in range(r):
                col = np.zeros(nI); col[idx] = Phi[:, k]; cols.append(col)
        return np.stack(cols, axis=1)


def layouts_for(N, kind):
    if kind == "chain":
        return [(i, 0) for i in range(N)]
    shapes = {4: (2, 2), 8: (4, 2), 16: (4, 4), 2: (2, 1)}
    W, H = shapes[N]
    return [(i, j) for j in range(H) for i in range(W)]


# ----------------------------------------------------------------------------- perturbations
def perturb(S, eps, kind, rng):
    n = len(S)
    if kind == "random":
        G = rng.standard_normal((n, n)); E = 0.5 * (G + G.T)
    else:
        w, V = np.linalg.eigh(S); third = n // 3
        idx = np.arange(n - third, n) if kind == "high" else np.arange(1, 1 + third)
        coef = rng.standard_normal(len(idx)); E = V[:, idx] @ np.diag(coef) @ V[:, idx].T
    return S + eps * norm(S) * E / norm(E)


# ----------------------------------------------------------------------------- PCG with/without coarse space
def pcg_iterations(asm, S_II, rhs, coarse, tol=1e-8, maxit=2000):
    nI = len(S_II); pos = {g: k for k, g in enumerate(asm.Iint)}
    blocks = [[pos[v]] for v in asm.I_vertex] + [[pos[g] for g in e] for e in asm.edges]
    inv_blocks = [(b, np.linalg.inv(S_II[np.ix_(b, b)])) for b in blocks]

    def M(r):
        z = np.zeros_like(r)
        for b, Bi in inv_blocks:
            z[b] += Bi @ r[b]
        return z

    if coarse:
        Z = np.zeros((nI, len(asm.comps)))
        mult = np.zeros(nI)
        for comp, gm in zip(asm.comps, asm.gmap):
            for g in gm[comp.B]:
                if g in pos: mult[pos[g]] += 1
        for k, (comp, gm) in enumerate(zip(asm.comps, asm.gmap)):
            for g in gm[comp.B]:
                if g in pos: Z[pos[g], k] = 1.0 / mult[pos[g]]
        Zc = Z - Z.mean(axis=1, keepdims=True) if False else Z
        C = Zc.T @ S_II @ Zc; Cinv = np.linalg.pinv(C)

        def M2(r):
            return M(r) + Zc @ (Cinv @ (Zc.T @ r))
        prec = M2
    else:
        prec = M
    x = np.zeros(nI); r = rhs.copy(); z = prec(r); p = z.copy(); rz = r @ z; b0 = norm(rhs)
    for it in range(1, maxit + 1):
        Ap = S_II @ p; alpha = rz / (p @ Ap); x += alpha * p; r -= alpha * Ap
        if norm(r) <= tol * b0:
            return it
        z = prec(r); rz_new = r @ z; p = z + (rz_new / rz) * p; rz = rz_new
    return maxit


# ----------------------------------------------------------------------------- main study
def main():
    t0 = time.time(); res = {"n_sub": N_SUB, "quick": QUICK}
    rng = rng_global
    # 1. response checks and composition law
    checks = {}
    for kind in LIBRARY + NEW_SHAPES:
        checks[kind] = response_checks(Component(kind, np.random.default_rng(1)))
    res["response_checks"] = checks
    pair = Assembly(layouts_for(2, "chain"), ["plain", "contrast"], np.random.default_rng(2))
    S = pair.global_S(); Sg = S[np.ix_(pair.P, pair.P)]
    E_idx = np.searchsorted(pair.P, pair.E); I_idx = np.searchsorted(pair.P, pair.Iint)
    S_comp = Sg[np.ix_(E_idx, E_idx)] - Sg[np.ix_(E_idx, I_idx)] @ np.linalg.solve(Sg[np.ix_(I_idx, I_idx)], Sg[np.ix_(I_idx, E_idx)])
    A_EE = pair.A[pair.E][:, pair.E].toarray(); A_EU = pair.A[pair.E][:, pair.U].toarray(); A_UU = pair.A[pair.U][:, pair.U].tocsc()
    S_mono = A_EE - A_EU @ spla.splu(A_UU).solve(A_EU.T)
    res["composition_law_rel_error"] = float(norm(S_comp - S_mono) / norm(S_mono))
    g = pair.dirichlet(np.random.default_rng(3)); u_m = pair.solve_monolithic(g); u_s, _ = pair.solve_schur(g)
    res["exact_assembly_error"] = pair.errors(u_s, u_m)
    print("checks", json.dumps(checks["plain"]), "composition law", res["composition_law_rel_error"], "exact assembly", res["exact_assembly_error"], flush=True)

    # 2. error propagation with injected local errors
    sizes = [2, 4, 8, 16]; eps_list = [1e-3, 1e-2, 1e-1]; kinds = ["random", "high", "low"]
    draws = 2 if QUICK else 5
    prop = []
    for c_val, label in ((0.0, "diffusion"), (1.0, "screened")):
        for lay in (["chain", "grid"] if label == "diffusion" else ["chain"]):
            for N in sizes:
                if lay == "grid" and N == 2:
                    continue
                for d in range(draws):
                    r_ = np.random.default_rng(100 + 17 * N + d)
                    kinds_c = list(r_.choice(LIBRARY, size=N))
                    asm = Assembly(layouts_for(N, lay), kinds_c, r_, c=c_val)
                    g = asm.dirichlet(r_); u_m = asm.solve_monolithic(g)
                    u_s, S_II = asm.solve_schur(g)
                    base = asm.errors(u_s, u_m)
                    w = np.linalg.eigvalsh(S_II); cond = float(w[-1] / max(w[0], 1e-300))
                    lu = spla.splu(asm.A[asm.U][:, asm.U].tocsc())
                    rec = {"variant": label, "layout": lay, "N": N, "draw": d, "kinds": kinds_c, "n_I": int(len(asm.Iint)), "n_dof": int(len(asm.U)),
                           "cond_S_II": cond, "nnz_LU_monolithic": int(lu.L.nnz + lu.U.nnz), "exact": base, "perturbed": {}}
                    for eps in eps_list:
                        for pk in kinds:
                            S_list = [perturb(c.S, eps, pk, r_) for c in asm.comps]
                            u_p, _ = asm.solve_schur(g, S_list)
                            rec["perturbed"][f"{pk}_{eps:g}"] = asm.errors(u_p, u_m)
                    if label == "diffusion":
                        rhs = -(asm.global_S()[np.ix_(asm.Iint, asm.E)] @ g[asm.E])
                        rec["pcg_iters"] = {"block_jacobi": pcg_iterations(asm, S_II, rhs, False), "two_level": pcg_iterations(asm, S_II, rhs, True)}
                    prop.append(rec)
                    print(f"{label} {lay} N={N} d={d} n_I={rec['n_I']} cond={cond:.2e} rand1e-2={rec['perturbed']['random_0.01']['energy']:.4f} high1e-2={rec['perturbed']['high_0.01']['energy']:.4f} low1e-2={rec['perturbed']['low_0.01']['energy']:.4f} pcg={rec.get('pcg_iters')} ({time.time()-t0:.0f}s)", flush=True)
    res["propagation"] = prop

    # 3. reduced responses: transfer-optimal modes from a plain reference pair; POD from random library assemblies
    ref = Assembly(layouts_for(2, "chain"), ["plain", "plain"], np.random.default_rng(5))
    S = ref.global_S(); S_II = S[np.ix_(ref.Iint, ref.Iint)]; S_IE = S[np.ix_(ref.Iint, ref.E)]
    T = -np.linalg.solve(S_II, S_IE)  # outer data -> interface trace (15 x |E|)
    order = np.argsort([ref.coords[g, 1] for g in ref.Iint]); T = T[order]
    U_, sv, _ = np.linalg.svd(T, full_matrices=False)
    Phi_transfer = U_; res["transfer_singular_values"] = sv.tolist()
    snaps = []
    for d in range(10 if QUICK else 40):
        r_ = np.random.default_rng(500 + d); N = int(r_.choice([4, 8])); lay = str(r_.choice(["chain", "grid"]))
        asm = Assembly(layouts_for(N, lay), list(r_.choice(LIBRARY, size=N)), r_)
        u_m = asm.solve_monolithic(asm.dirichlet(r_))
        for e in asm.edges:
            tr = u_m[e]; snaps.append(tr - tr.mean() if False else tr)
    Up, sp_, _ = np.linalg.svd(np.stack(snaps, axis=1), full_matrices=False)
    Phi_pod = Up; res["pod_singular_values"] = sp_.tolist()
    nE = Phi_transfer.shape[0]
    rank_scan = []
    for shape_set, label in ((LIBRARY, "library"), (NEW_SHAPES, "new_shapes")):
        for lay, N in (("chain", 8), ("grid", 8), ("grid", 16)):
            for d in range(draws):
                r_ = np.random.default_rng(900 + 31 * N + d + (7 if label == "new_shapes" else 0))
                asm = Assembly(layouts_for(N, lay), list(r_.choice(shape_set, size=N)), r_)
                g = asm.dirichlet(r_); u_m = asm.solve_monolithic(g)
                rec = {"shapes": label, "layout": lay, "N": N, "draw": d, "transfer": {}, "pod": {}, "truncated_S": {}}
                for r in range(1, nE + 1):
                    for name, Phi in (("transfer", Phi_transfer), ("pod", Phi_pod)):
                        P = asm.reduction_matrix(Phi, r)
                        u_r, _ = asm.solve_schur(g, P=P)
                        rec[name][r] = asm.errors(u_r, u_m)["energy"]
                for r in (8, 16, 32, 48, 63):
                    S_list = []
                    for c_ in asm.comps:
                        w, V = np.linalg.eigh(c_.S); keep = np.argsort(-np.abs(w))[:r]
                        S_list.append(V[:, keep] @ np.diag(w[keep]) @ V[:, keep].T)
                    try:
                        u_t, _ = asm.solve_schur(g, S_list); rec["truncated_S"][r] = asm.errors(u_t, u_m)["energy"]
                    except np.linalg.LinAlgError:
                        rec["truncated_S"][r] = None
                rank_scan.append(rec)
                print(f"rank scan {label} {lay} N={N} d={d}: transfer r=4 {rec['transfer'][4]:.4f} r=8 {rec['transfer'][8]:.4f}; pod r=4 {rec['pod'][4]:.4f}; truncS r=32 {rec['truncated_S'][32]} ({time.time()-t0:.0f}s)", flush=True)
    res["rank_scan"] = rank_scan
    # near-field: channel_narrow vs channel_wide chains (all components of one kind)
    nf = {}
    for kind in ("channel_wide", "channel_narrow", "plain"):
        errs = {"transfer": {}, "amp_random_1e-2": [], "amp_high_1e-2": []}
        for d in range(draws):
            r_ = np.random.default_rng(1500 + d)
            asm = Assembly(layouts_for(8, "chain"), [kind] * 8, r_)
            g = asm.dirichlet(r_); u_m = asm.solve_monolithic(g)
            for r in (2, 4, 8):
                P = asm.reduction_matrix(Phi_transfer, r); u_r, _ = asm.solve_schur(g, P=P)
                errs["transfer"].setdefault(r, []).append(asm.errors(u_r, u_m)["energy"])
            for pk in ("random", "high"):
                S_list = [perturb(c.S, 1e-2, pk, r_) for c in asm.comps]
                u_p, _ = asm.solve_schur(g, S_list); errs[f"amp_{pk}_1e-2"].append(asm.errors(u_p, u_m)["energy"] / 1e-2)
        nf[kind] = {"transfer_median_energy_by_rank": {r: float(np.median(v)) for r, v in errs["transfer"].items()},
                    "amplification_random_1e-2_median": float(np.median(errs["amp_random_1e-2"])),
                    "amplification_high_1e-2_median": float(np.median(errs["amp_high_1e-2"]))}
    res["near_field"] = nf
    res["seconds"] = time.time() - t0
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    summarize(res)


def summarize(res):
    import statistics as st
    prop = [r for r in res["propagation"] if r["variant"] == "diffusion"]
    print("\n== amplification (median global energy error / eps) by layout, N, kind ==")
    for lay in ("chain", "grid"):
        for N in (2, 4, 8, 16):
            rows = [r for r in prop if r["layout"] == lay and r["N"] == N]
            if not rows: continue
            line = f"{lay:5s} N={N:2d} n_I={rows[0]['n_I']:4d} n_dof={rows[0]['n_dof']:5d} cond={st.median(r['cond_S_II'] for r in rows):.1e} "
            for pk in ("random", "high", "low"):
                for eps in (1e-3, 1e-2, 1e-1):
                    v = st.median(r["perturbed"][f"{pk}_{eps:g}"]["energy"] for r in rows)
                    if eps == 1e-2: line += f"{pk}:{v:.4f} "
            pc = [r["pcg_iters"] for r in rows]
            line += f"pcg bj/2lvl={st.median(p['block_jacobi'] for p in pc):.0f}/{st.median(p['two_level'] for p in pc):.0f}"
            print(line)
    print("\n== rank needed for <=1% median energy error ==")
    for label in ("library", "new_shapes"):
        for lay, N in (("chain", 8), ("grid", 8), ("grid", 16)):
            rows = [r for r in res["rank_scan"] if r["shapes"] == label and r["layout"] == lay and r["N"] == N]
            for name in ("transfer", "pod"):
                med = {r_: st.median(row[name][r_] for row in rows) for r_ in rows[0][name]}
                need = next((r_ for r_ in sorted(med) if med[r_] <= 0.01), None)
                print(f"{label:10s} {lay:5s} N={N:2d} {name:8s} rank for 1%: {need}  (r=2 {med[2]:.4f}, r=4 {med[4]:.4f}, r=8 {med[8]:.4f})")
    print("\n== near field ==", json.dumps(res["near_field"], indent=1))


if __name__ == "__main__":
    main()
