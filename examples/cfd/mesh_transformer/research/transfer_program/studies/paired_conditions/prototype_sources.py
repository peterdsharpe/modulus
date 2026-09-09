"""Instrument repair prototype (recorded in NOTEBOOK.md): normal-offset MFS sources vs the
shrunken-copy placement that failed Q1-Q3; ground-clearance sweep for the 5-20% band."""
import numpy as np, time, sys
sys.argv = ["x", "."]
import reference as R


def run(a, b, c, p, k, h=None, n_theta=26, n_phi=40):
    V, T = R.superellipsoid(a, b, c, p, n_theta, n_phi); cen, nrm, area = R.mesh_quantities(V, T)
    s = cen - k * np.sqrt(area)[:, None] * nrm      # one source per cell, inward normal offset
    images = None
    if h is not None:
        z0 = V[:, 2].min() - h; images = s.copy(); images[:, 2] = 2 * z0 - images[:, 2]
    q = R.solve(cen, nrm, s, images); u = R.velocity(cen, s, q, images); cp = 1 - (u ** 2).sum(1)
    vn = np.zeros_like(V)
    for kk in range(3): np.add.at(vn, T[:, kk], nrm * area[:, None])
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-30
    resid = np.abs((R.velocity(V, s, q, images) * vn).sum(1))
    return V, T, cen, cp, float(np.quantile(resid, .9)), s, q


t0 = time.time()
for k in (0.5, 1.0, 1.5, 2.5):
    V, T, cen, cp, rq, s, q = run(1, 1, 1, 2, k)
    proj = cen / np.linalg.norm(cen, axis=1, keepdims=True); th = np.arccos(np.clip(proj[:, 0], -1, 1)); cpan = 1 - 2.25 * np.sin(th) ** 2
    us = R.velocity(proj, s, q); cps = 1 - (us ** 2).sum(1)
    print(f"sphere k={k}: Q1 surf {np.abs(cps-cpan).max():.2e} centroid {np.abs(cp-cpan).max():.2e}  resid q90 {rq:.2e}  ({time.time()-t0:.0f}s)", flush=True)
for (a, b, c, p) in [(1.29, 0.53, 0.91, 2.5), (1.29, 0.53, 0.91, 2.74)]:
    V, T, cen, cp, rq, _, _ = run(a, b, c, p, 1.0); print(f"boxy p={p}: resid q90 {rq:.2e}", flush=True)
for hoc in (0.5, 1.0, 1.5):
    V, T, cen, cp0, _, _, _ = run(1, 1, 1, 2, 1.0); V, T, cen, cp1, rq, _, _ = run(1, 1, 1, 2, 1.0, h=hoc)
    print(f"sphere ground h/c={hoc}: rms change {np.sqrt(np.mean((cp1-cp0)**2))/np.sqrt(np.mean(cp0**2)):.3f} resid {rq:.2e}", flush=True)
V, T, cen, cpf, rq, s, q = run(1, 1, 1, 2, 1.0, n_theta=36, n_phi=56)
print("fine sphere resid", rq, f"({time.time()-t0:.0f}s)")
