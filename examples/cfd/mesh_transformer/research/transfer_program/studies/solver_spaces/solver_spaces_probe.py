"""SOLVER-SPACES probe (PREREG.md in this directory): does a frozen source-fitted
correction space transfer to held-out 2D FEM systems better than an equally cheap
classical coarse space?

P1 finite elements on a uniform right-triangle mesh of the unit square with circular
holes (Neumann) and a circular kappa inclusion; Dirichlet data lifted on the outer
boundary. Diffusion (SPD, PCG, energy norm) and convection-diffusion (nonsymmetric,
GMRES, Euclidean norm). Correction spaces of equal dimension r: frozen source POD of
smoother-resistant errors (Y_src), per-target oracle POD, aggregation, smoothed
aggregation, geometric bilinear coarse grid. Probes: random / slow / near-null error
directions. Complete solves with the additive two-level preconditioner.

Run:  python solver_spaces_probe.py [out_dir]   (numpy, scipy, matplotlib; ~minutes on CPU)
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
N_SIDE = 48
OMEGA = 2.0 / 3.0
K_SMOOTH = 3
RANKS = (16, 32)
N_SRC, N_HELD, N_RHS = 20, 8, 10
TOL, MAXIT, N_RHS_AMORTIZE = 1e-8, 500, 10
rng_global = np.random.default_rng(20260909)


# ----------------------------------------------------------------------------- mesh
def grid_nodes(n):
    xs = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)  # node id = i*(n+1)+j


def grid_triangles(n):
    idx = lambda i, j: i * (n + 1) + j
    tris = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1)
            tris += [(a, b, d), (a, d, c)]
    return np.array(tris)


NODES, TRIS = grid_nodes(N_SIDE), grid_triangles(N_SIDE)
N_GRID = len(NODES)
OUTER = (np.isclose(NODES[:, 0], 0) | np.isclose(NODES[:, 0], 1) | np.isclose(NODES[:, 1], 0) | np.isclose(NODES[:, 1], 1))
CENT = NODES[TRIS].mean(1)


class Geometry:
    def __init__(self, holes, inclusion, kappa_c, tag):
        self.holes, self.inclusion, self.kappa_c, self.tag = holes, inclusion, kappa_c, tag
        inside = np.zeros(len(TRIS), bool)
        for cx, cy, r in holes:
            inside |= (CENT[:, 0] - cx) ** 2 + (CENT[:, 1] - cy) ** 2 < r ** 2
        self.tris = TRIS[~inside]
        active = np.zeros(N_GRID, bool); active[self.tris.ravel()] = True
        self.active = active
        self.dirichlet = active & OUTER
        self.unknown = np.flatnonzero(active & ~OUTER)
        self.n_u = len(self.unknown)
        cx, cy, r = inclusion
        c = CENT[~inside]
        self.kappa = np.where((c[:, 0] - cx) ** 2 + (c[:, 1] - cy) ** 2 < r ** 2, kappa_c, 1.0)
        self._assembled = {}

    def assemble(self, pde):
        if pde in self._assembled:
            return self._assembled[pde]
        P = NODES[self.tris]  # (T, 3, 2)
        v0, v1, v2 = P[:, 0], P[:, 1], P[:, 2]
        J = np.stack([v1 - v0, v2 - v0], 2)  # (T, 2, 2)
        detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        area = 0.5 * np.abs(detJ)
        # gradients of the three barycentric functions
        invJT = np.linalg.inv(J).transpose(0, 2, 1)  # (T,2,2)
        gref = np.array([[-1, -1], [1, 0], [0, 1]], float)  # (3,2)
        grad = np.einsum("tij,kj->tki", invJT, gref)  # (T,3,2)
        if pde == "diffusion":
            coef = self.kappa
            Ke = coef[:, None, None] * area[:, None, None] * np.einsum("tki,tli->tkl", grad, grad)
        else:  # convection-diffusion: eps*grad.grad + phi_i (b.grad phi_j)
            eps, b = 0.02, np.array([1.0, 0.3])
            Ke = eps * area[:, None, None] * np.einsum("tki,tli->tkl", grad, grad)
            Ke = Ke + (area / 3.0)[:, None, None] * np.einsum("i,tli->tl", b, grad)[:, None, :]
        rows = np.repeat(self.tris, 3, axis=1).ravel(); cols = np.tile(self.tris, (1, 3)).ravel()
        A_full = sp.csr_matrix((Ke.ravel(), (rows, cols)), shape=(N_GRID, N_GRID))
        # lumped load for unit source and for arbitrary nodal f: M_lumped
        m_l = np.zeros(N_GRID); np.add.at(m_l, self.tris.ravel(), np.repeat(area / 3.0, 3))
        g = NODES[:, 0] + 0.5 * NODES[:, 1]
        u = self.unknown; d = np.flatnonzero(self.dirichlet)
        A = A_full[u][:, u].tocsr()
        lift = A_full[u][:, d] @ g[d]
        self._assembled[pde] = (A, lift, m_l[u])
        return A, lift, m_l[u]

    def rhs(self, pde, f_nodal):
        A, lift, m_l = self.assemble(pde)
        return m_l * f_nodal[self.unknown] - lift

    def embed(self, v):
        out = np.zeros((N_GRID,) + v.shape[1:]); out[self.unknown] = v; return out

    def restrict(self, V):
        return V[self.unknown]


def random_geometry(rng, tag, n_holes=1, kappa_c=10.0, narrow_gap=False):
    holes = []
    h = 1.0 / N_SIDE
    for k in range(n_holes):
        r = rng.uniform(0.08, 0.18)
        if narrow_gap and k == 0:
            gap = rng.uniform(2, 3) * h
            cx = r + gap; cy = rng.uniform(0.3, 0.7)
        else:
            cx, cy = rng.uniform(r + 0.12, 1 - r - 0.12, 2)
        holes.append((cx, cy, r))
    ri = rng.uniform(0.10, 0.20)
    inclusion = (rng.uniform(ri + 0.05, 1 - ri - 0.05), rng.uniform(ri + 0.05, 1 - ri - 0.05), ri)
    return Geometry(holes, inclusion, kappa_c, tag)


def smooth_rhs(rng):
    f = np.zeros(N_GRID)
    for _ in range(4):
        kx, ky = rng.integers(1, 4, 2); ph = rng.uniform(0, 2 * np.pi, 2)
        f += rng.normal() * np.cos(kx * np.pi * NODES[:, 0] + ph[0]) * np.cos(ky * np.pi * NODES[:, 1] + ph[1])
    return f


# ------------------------------------------------------------------ smoother, spaces
def jacobi(A, b, x, k, omega=OMEGA):
    dinv = 1.0 / A.diagonal()
    for _ in range(k):
        x = x + omega * dinv * (b - A @ x)
    return x


def error_snapshots(geo, pde, rng, n_rhs=N_RHS):
    A, _, _ = geo.assemble(pde)
    E = []
    for _ in range(n_rhs):
        b = geo.rhs(pde, smooth_rhs(rng))
        u = spla.spsolve(A.tocsc(), b)
        E.append(geo.embed(u - jacobi(A, b, np.zeros_like(b), K_SMOOTH)))
    return np.stack(E, 1)  # (N_GRID, n_rhs)


def pod(E, r):
    U, s, _ = np.linalg.svd(E, full_matrices=False)
    return U[:, :r], s


def orthonormal(Y):
    Q, R = np.linalg.qr(Y)
    keep = np.abs(np.diag(R)) > 1e-10 * np.abs(R).max()
    return Q[:, keep]


def aggregation_space(geo, r):
    m = int(round(math.sqrt(r)))
    xy = NODES[geo.unknown]
    cell = np.minimum((xy * m).astype(int), m - 1)
    ids = cell[:, 0] * m + cell[:, 1]
    cols = []
    for c in np.unique(ids):
        v = np.zeros(geo.n_u); v[ids == c] = 1.0; cols.append(v)
    return np.stack(cols, 1)


def geometric_space(geo, r):
    m = int(round(math.sqrt(r)))  # interior coarse nodes per side -> m*m tents
    xy = NODES[geo.unknown]; H = 1.0 / (m + 1)
    cols = []
    for i in range(1, m + 1):
        for j in range(1, m + 1):
            tent = np.maximum(0, 1 - np.abs(xy[:, 0] - i * H) / H) * np.maximum(0, 1 - np.abs(xy[:, 1] - j * H) / H)
            if tent.any():
                cols.append(tent)
    return np.stack(cols, 1)


class Coarse:
    def __init__(self, A, Y):
        self.Y = orthonormal(Y); self.AY = A @ self.Y
        self.G = self.Y.T @ self.AY
        self.r = self.Y.shape[1]

    def correct(self, res):
        return self.Y @ np.linalg.solve(self.G, self.Y.T @ res)


def build_spaces(geo, pde, r, Y_src, rng):
    A, _, _ = geo.assemble(pde)
    Ysrc = geo.restrict(Y_src)
    reg_loss = 1.0 - (Ysrc ** 2).sum() / (Y_src ** 2).sum()
    E_t = error_snapshots(geo, pde, rng)
    Yor, _ = pod(E_t, r)
    Yagg = aggregation_space(geo, r)
    dinv = 1.0 / A.diagonal()
    Ysa = Yagg - OMEGA * (dinv[:, None] * (A @ Yagg))
    spaces = {"src_frozen": Coarse(A, Ysrc), "oracle_pod": Coarse(A, geo.restrict(Yor)),
              "aggregation": Coarse(A, Yagg), "smoothed_aggregation": Coarse(A, Ysa),
              "geometric": Coarse(A, geometric_space(geo, r))}
    return spaces, reg_loss


# --------------------------------------------------------------------------- probes
def probes(geo, pde, rng):
    A, _, _ = geo.assemble(pde)
    S = A if pde == "diffusion" else 0.5 * (A + A.T)
    D = sp.diags(A.diagonal())
    vals, vecs = spla.eigsh(S.tocsc(), k=20, M=D.tocsc(), sigma=0.0, which="LM")
    order = np.argsort(vals); vecs = vecs[:, order]
    return {"random": rng.normal(size=(geo.n_u, 20)), "slow": vecs[:, 3:20], "near_null": vecs[:, :3]}


def contraction(A, coarse, E, pde):
    R = A @ E
    Ec = E - coarse.correct(R) if E.ndim == 1 else E - np.stack([coarse.correct(R[:, i]) for i in range(E.shape[1])], 1)
    Rc = A @ Ec
    if pde == "diffusion":
        num = np.sqrt(np.einsum("ij,ij->j", Ec, A @ Ec)); den = np.sqrt(np.einsum("ij,ij->j", E, A @ E))
    else:
        num = np.linalg.norm(Ec, axis=0); den = np.linalg.norm(E, axis=0)
    return float(np.mean(num / den)), float(np.mean(np.linalg.norm(Rc, axis=0) / np.linalg.norm(R, axis=0)))


# ----------------------------------------------------------------------- solves
def pcg(A, b, x0, precond, u_exact, tol=TOL, maxit=MAXIT):
    x = x0.copy(); r = b - A @ x; z = precond(r); p = z.copy(); rz = r @ z; bn = np.linalg.norm(b)
    err = [float(np.sqrt((x - u_exact) @ (A @ (x - u_exact))))]; res = [float(np.linalg.norm(r) / bn)]
    for it in range(1, maxit + 1):
        Ap = A @ p; alpha = rz / (p @ Ap); x = x + alpha * p; r = r - alpha * Ap
        e = x - u_exact; err.append(float(np.sqrt(abs(e @ (A @ e))))); res.append(float(np.linalg.norm(r) / bn))
        if res[-1] < tol:
            return it, res, err
        z = precond(r); rz_new = r @ z; p = z + (rz_new / rz) * p; rz = rz_new
    return maxit, res, err


def gmres_solve(A, b, x0, precond, u_exact, tol=TOL, maxit=MAXIT):
    n = len(b); res, err = [], []; bn = np.linalg.norm(b)

    def cb(xk):
        res.append(float(np.linalg.norm(b - A @ xk) / bn)); err.append(float(np.linalg.norm(xk - u_exact)))

    M = spla.LinearOperator((n, n), matvec=precond)
    x, info = spla.gmres(A, b, x0=x0, M=M, rtol=tol, restart=50, maxiter=maxit, callback=cb, callback_type="x")
    its = len(res) if res else maxit
    return (its if info == 0 else maxit), res, err


def additive_precond(A, coarse):
    dinv = 1.0 / A.diagonal()
    if coarse is None:
        return lambda r: OMEGA * dinv * r
    return lambda r: OMEGA * dinv * r + coarse.correct(r)


def work_per_iter(A, r):
    nnz = A.nnz; n = A.shape[0]
    return 1.0 + (2 * r * n + r * r) / nnz  # matvec + coarse apply, in matvec equivalents


def setup_work(A, r):
    nnz = A.nnz; n = A.shape[0]
    return r + (r * r * n + 2 * r * r * n) / nnz  # AY (r matvecs) + Galerkin + QR


# ------------------------------------------------------------------------------ main
def main():
    t0 = time.time()
    rng = np.random.default_rng(1)
    sources = [random_geometry(rng, f"src{i}") for i in range(N_SRC)]
    held = {"within_family": [random_geometry(rng, f"wf{i}") for i in range(N_HELD)],
            "topology_shift": [random_geometry(rng, f"top{i}", n_holes=2) if i % 2 == 0 else random_geometry(rng, f"gap{i}", narrow_gap=True) for i in range(N_HELD)],
            "coefficient_shift": [random_geometry(rng, f"kap{i}", kappa_c=100.0) for i in range(N_HELD)]}
    results = {"config": {"n_side": N_SIDE, "omega": OMEGA, "k_smooth": K_SMOOTH, "ranks": RANKS, "n_src": N_SRC, "n_held": N_HELD,
                          "n_rhs": N_RHS, "tol": TOL, "maxit": MAXIT, "amortize_rhs": N_RHS_AMORTIZE,
                          "n_unknowns_range": [min(g.n_u for g in sources), max(g.n_u for g in sources)]},
               "pde": {}}
    for pde in ("diffusion", "convection"):
        print(f"== {pde}", flush=True)
        E = np.concatenate([error_snapshots(g, pde, rng) for g in sources], 1)
        Uall, svals = pod(E, max(RANKS))
        # source mean solution for the learned-initial-guess control (unit source)
        mean_u = np.mean([g.embed(spla.spsolve(g.assemble(pde)[0].tocsc(), g.rhs(pde, np.ones(N_GRID)))) for g in sources], 0)
        pres = {"pod_singular_values": (svals / svals[0]).tolist(), "ranks": {}}
        for r in RANKS:
            Y_src = Uall[:, :r]
            rres = {}
            for set_name, geos in held.items():
                acc = {}
                for geo in geos:
                    A, _, _ = geo.assemble(pde)
                    spaces, reg_loss = build_spaces(geo, pde, r, Y_src, rng)
                    pr = probes(geo, pde, rng)
                    rec = {"n_u": geo.n_u, "registration_loss": reg_loss, "dims": {k: v.r for k, v in spaces.items()}, "contraction": {}, "solve": {}}
                    for name, cs in spaces.items():
                        rec["contraction"][name] = {cls: contraction(A, cs, Eprobe, pde) for cls, Eprobe in pr.items()}
                    # complete solves, unit source
                    b = geo.rhs(pde, np.ones(N_GRID)); u_ex = spla.spsolve(A.tocsc(), b)
                    solver = pcg if pde == "diffusion" else gmres_solve
                    for name, cs in list(spaces.items()) + [("none", None)]:
                        its, res, err = solver(A, b, np.zeros_like(b), additive_precond(A, cs), u_ex)
                        rec["solve"][name] = {"iterations": its, "converged": its < MAXIT and res[-1] < TOL,
                                              "work_iter": work_per_iter(A, cs.r) if cs else 1.0,
                                              "setup": setup_work(A, cs.r) if cs else 0.0,
                                              "residual_hist": res, "error_hist": err}
                    its, res, err = solver(A, b, geo.restrict(mean_u), additive_precond(A, spaces["smoothed_aggregation"]), u_ex)
                    rec["solve"]["smoothed_aggregation+mean_x0"] = {"iterations": its, "converged": its < MAXIT and res[-1] < TOL,
                                                                    "work_iter": work_per_iter(A, spaces["smoothed_aggregation"].r),
                                                                    "setup": setup_work(A, spaces["smoothed_aggregation"].r), "residual_hist": res, "error_hist": err}
                    acc[geo.tag] = rec
                    print(f"  r={r} {set_name} {geo.tag}: n_u={geo.n_u} regloss={reg_loss:.3f} slow-contraction src={rec['contraction']['src_frozen']['slow'][0]:.3f} "
                          f"sa={rec['contraction']['smoothed_aggregation']['slow'][0]:.3f} geo={rec['contraction']['geometric']['slow'][0]:.3f} oracle={rec['contraction']['oracle_pod']['slow'][0]:.3f} | "
                          f"its src={rec['solve']['src_frozen']['iterations']} sa={rec['solve']['smoothed_aggregation']['iterations']} none={rec['solve']['none']['iterations']}  ({time.time()-t0:.0f}s)", flush=True)
                rres[set_name] = acc
            pres["ranks"][str(r)] = rres
        results["pde"][pde] = pres
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "solver_spaces_results.json").write_text(json.dumps(results))
    print("wrote", OUT / "solver_spaces_results.json", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
