"""Qualified 2D interior Laplace Dirichlet-to-Neumann (DtN) operators by a Nystrom
boundary-integral method, plus the principal-part / null-mode / basis-transfer
readouts of PREREG.md.

Direct formulation on a boundary Gamma = union of smooth closed curves:
    V psi = (1/2 I + K) g,  Lambda = V^{-1} (1/2 I + K),
with G(x, y) = -(1/2 pi) log|x - y| (so -Delta G = delta), V the single layer,
K the double layer with the DOMAIN-outward normal, psi the outward flux.
Each component is resampled at equal arclength (2N nodes) so the discrete
Fourier basis in the parameter is the arclength Fourier basis. The log kernel
uses Kress quadrature; the double layer (continuous on smooth curves) uses the
trapezoid rule with the analytic diagonal limit -(x1' x2'' - x2' x1'') / (2 |x'|^2).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


# ----------------------------------------------------------------------------- curves
def resample_arclength(curve, n_nodes: int, fine: int = 16384):
    """Return nodes x (n,2), tangent x' and x'' w.r.t. t in [0, 2pi) of an arclength
    parameterization of the analytic closed curve `curve(theta) -> (n,2)`, plus length L."""
    th = np.linspace(0.0, 2 * np.pi, fine, endpoint=False)
    p = curve(th)
    dp = np.fft.irfft(np.fft.rfft(p, axis=0) * (1j * np.fft.rfftfreq(fine, 1.0 / fine))[:, None], n=fine, axis=0)
    speed = np.linalg.norm(dp, axis=1)
    # cumulative arclength by spectral integration of speed (periodic, smooth)
    s_hat = np.fft.rfft(speed)
    k = np.fft.rfftfreq(fine, 1.0 / fine)
    L = float(s_hat[0].real / fine * 2 * np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        int_hat = np.where(k == 0, 0.0, s_hat / (1j * k))
    s = np.fft.irfft(int_hat, n=fine) + th * s_hat[0].real / fine  # s(theta), s(0)=const
    s -= s[0]
    th_ext = np.concatenate([th, [2 * np.pi]]); s_ext = np.concatenate([s, [L]])
    inv = CubicSpline(s_ext, th_ext)  # theta(s), monotone
    s_nodes = np.arange(n_nodes) * L / n_nodes
    x = curve(inv(s_nodes))
    t = 2 * np.pi * np.arange(n_nodes) / n_nodes
    kk = np.fft.rfftfreq(n_nodes, 1.0 / n_nodes)
    xh = np.fft.rfft(x, axis=0)
    xd = np.fft.irfft(xh * (1j * kk)[:, None], n=n_nodes, axis=0)
    xdd = np.fft.irfft(xh * (-(kk ** 2))[:, None], n=n_nodes, axis=0)
    return x, xd, xdd, t, L


def ellipse(a, b, c=(0.0, 0.0)):
    return lambda th: np.stack([c[0] + a * np.cos(th), c[1] + b * np.sin(th)], -1)


def star(amp, m, c=(0.0, 0.0)):
    return lambda th: np.stack([c[0] + (1 + amp * np.cos(m * th)) * np.cos(th), c[1] + (1 + amp * np.cos(m * th)) * np.sin(th)], -1)


def superellipse(p, c=(0.0, 0.0)):
    """|x|^p + |y|^p = 1 in polar form; analytic for even p; corner radius ~ 1/((p-1) 2^{1/p})."""
    def f(th):
        r = (np.abs(np.cos(th)) ** p + np.abs(np.sin(th)) ** p) ** (-1.0 / p)
        return np.stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th)], -1)
    return f


def circle(r, c=(0.0, 0.0)):
    return ellipse(r, r, c)


# ----------------------------------------------------------------------------- Nystrom
def kress_weights(n_half: int):
    """R_j(t_i) for the log(4 sin^2((t-tau)/2)) kernel on 2N nodes (Kress, eq. 12.18)."""
    N = n_half
    t = np.pi * np.arange(2 * N) / N
    m = np.arange(1, N)
    # R depends only on the difference t_i - t_j: build the 1D profile on the 2N distinct differences
    prof = -(2 * np.pi / N) * (np.cos(np.outer(t, m)) / m[None, :]).sum(axis=1) - (np.pi / N ** 2) * np.cos(N * t)
    idx = (np.arange(2 * N)[:, None] - np.arange(2 * N)[None, :]) % (2 * N)
    return prof[idx]


R0 = 20.0  # log-kernel length scale: G = -(1/2pi) log(|x-y|/R0). The interior flux has zero total
           # over the whole boundary, so the constant shift does not change psi; it removes the unit-
           # capacity singularity of the plain log single layer (V is positive definite for diameters < R0).


def assemble_dtn(components):
    """components: list of dicts with keys x, xd, xdd, t, L, sign (+1 outer/CCW-outward, -1 inner).
    Returns Lambda (n_tot x n_tot), block slices, and the assembled V, K."""
    n_tot = sum(len(c["x"]) for c in components)
    V = np.zeros((n_tot, n_tot)); K = np.zeros((n_tot, n_tot))
    slices = []; off = 0
    for c in components:
        slices.append(slice(off, off + len(c["x"]))); off += len(c["x"])
    for i, ci in enumerate(components):
        xi = ci["x"]
        for j, cj in enumerate(components):
            xj, xdj, xddj = cj["x"], cj["xd"], cj["xdd"]
            nj = len(xj); N = nj // 2; h = np.pi / N
            speed_j = np.linalg.norm(xdj, axis=1)
            ntil = cj["sign"] * np.stack([xdj[:, 1], -xdj[:, 0]], -1)  # domain-outward normal * |x'|
            diff = xi[:, None, :] - xj[None, :, :]
            r2 = np.sum(diff ** 2, axis=2)
            if i == j:
                t = cj["t"]; d = t[:, None] - t[None, :]
                s2 = 4 * np.sin(d / 2) ** 2
                with np.errstate(divide="ignore", invalid="ignore"):
                    Lsm = np.log(r2 / s2) - 2 * np.log(R0)
                np.fill_diagonal(Lsm, np.log(speed_j ** 2) - 2 * np.log(R0))
                Vb = -(1.0 / (4 * np.pi)) * (kress_weights(N) + h * Lsm) * speed_j[None, :]
                with np.errstate(divide="ignore", invalid="ignore"):
                    kd = np.einsum("ijk,jk->ij", diff, ntil) / r2
                kdiag = -cj["sign"] * (xdj[:, 0] * xddj[:, 1] - xdj[:, 1] * xddj[:, 0]) / (2 * speed_j ** 2)
                np.fill_diagonal(kd, kdiag)
                Kb = h / (2 * np.pi) * kd
            else:
                Vb = -(1.0 / (2 * np.pi)) * 0.5 * (np.log(r2) - 2 * np.log(R0)) * speed_j[None, :] * h
                Kb = h / (2 * np.pi) * np.einsum("ijk,jk->ij", diff, ntil) / r2
            V[slices[i], slices[j]] = Vb; K[slices[i], slices[j]] = Kb
    rhs = 0.5 * np.eye(n_tot) + K
    Lam = np.linalg.solve(V, rhs)
    return Lam, slices, V, K


def make_component(curve, n_half, sign=+1):
    x, xd, xdd, t, L = resample_arclength(curve, 2 * n_half)
    return {"x": x, "xd": xd, "xdd": xdd, "t": t, "L": L, "sign": sign}


# ----------------------------------------------------------------------------- Fourier tools
def real_fourier_matrix(n):
    """Orthonormal real DFT matrix Q (n x n): rows const, cos k, sin k (k=1..n/2-1), cos(n/2)."""
    t = 2 * np.pi * np.arange(n) / n
    rows = [np.full(n, 1 / np.sqrt(n))]
    ks = [0]
    for k in range(1, n // 2):
        rows.append(np.sqrt(2 / n) * np.cos(k * t)); ks.append(k)
        rows.append(np.sqrt(2 / n) * np.sin(k * t)); ks.append(k)
    rows.append(np.cos((n // 2) * t) / np.sqrt(n)); ks.append(n // 2)
    return np.stack(rows), np.array(ks)


def principal_part(n, L):
    """Half-Laplacian symbol 2 pi |k| / L in the arclength Fourier basis, as an n x n node-basis matrix."""
    Q, ks = real_fourier_matrix(n)
    return Q.T @ np.diag(2 * np.pi * ks / L) @ Q, Q, ks


def block_diag(mats):
    n = sum(m.shape[0] for m in mats); out = np.zeros((n, n)); o = 0
    for m in mats:
        out[o:o + m.shape[0], o:o + m.shape[0]] = m; o += m.shape[0]
    return out


def rank_at(A, tols=(1e-2, 1e-3), ref=None):
    """Smallest r with ||A - A_r||_F <= tol * ref (ref defaults to ||A||_F). Returns dict tol->rank and the singular values."""
    s = np.linalg.svd(A, compute_uv=False)
    tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]  # tail[r] = ||A - A_r||_F
    tail = np.concatenate([tail, [0.0]])
    ref = np.sqrt(np.sum(s ** 2)) if ref is None else ref
    out = {}
    for tol in tols:
        r = int(np.argmax(tail <= tol * ref))
        out[str(tol)] = r
    return out, s
