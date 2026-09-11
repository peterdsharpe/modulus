"""MOM2: is a per-slice second-moment channel worth having? (notebook #sec-nb-mom2-prereg)

Eval-only expressivity benchmark on the first-moment collision family of the
2026-09-08 audit (item 3). Each draw gives two arrangements of five equal
spherical component samples about an axial drive, at azimuths
{0, 120, 240, phi, phi+180} and {0, 120, 240, phi', phi'+180} degrees with
phi, phi' ~ U(0, 120) and |phi - phi'| >= 10 degrees: both have zero
transverse first moment (an equilateral triple plus an antipodal pair), so
first-moment slice anchors stay axial and ISLA's eight relational invariants
coincide on the common (0 degree) component. Congruent control pairs (the
same arrangement rotated about the drive) must be called identical by every
arm.

Arms (untrained ISLA, hidden 64, 4 layers, 32 slices, float64, three seeds):
  base            default gauge
  base_gauge      similarity gauge
  local           use_local_features=True, radii (0.2, 0.4)
  mom2            second_moment_features=True
Readouts per pair: max |output_A - output_B| on the common component; over the
200 pairs, 5-fold cross-validated logistic-probe accuracy classifying the
common-component output vector as phi- or phi'-family (chance 50%).

Bars (preregistered): supported if mom2 probe accuracy >= 95% on every seed
while base stays within 45-55% with differences < 1e-12; falsified if neither
mom2 nor local exceeds 60%. Controls: every arm < 1e-12 on all congruent pairs.

Amendment (first run, 2026-09-08): the preregistered family-label probe is
ill-posed. phi and phi' are exchangeable draws from U(0, 120), so "phi-family"
versus "phi'-family" is an arbitrary label and every arm scores at chance by
construction (observed 0.46-0.50 for all four arms, including the two whose
common-component outputs differ by 1e-3 to 1e-1). The probe is kept for the
record and two well-posed readouts of the same question (can the free azimuth
be read off the common component's outputs?) are added: the 5-fold CV R^2 of a
ridge regression of phi on the output vector, and the 5-fold CV accuracy of a
logistic probe for phi >= 60 degrees (chance 50%). The preregistered bars are
applied to the phi >= 60 accuracy in place of the family-label accuracy.

Usage: uv run --no-sync python results/mom2_second_moment_benchmark_2026-09-08.py [out.json]
"""
import json, math, sys, time
import numpy as np
import torch

from physicsnemo.experimental.nn.isla import ISLA

torch.set_num_threads(4)
dtype = torch.float64
N_PAIRS, N_CONTROL, SEEDS = 200, 200, (0, 1, 2)
out_path = sys.argv[1] if len(sys.argv) > 1 else "results/mom2_second_moment_benchmark_2026-09-08.json"

normal = torch.tensor([[i, j, k] for i in (-1.0, 1.0) for j in (-1.0, 1.0) for k in (-1.0, 1.0)], dtype=dtype) / math.sqrt(3)
point = torch.tensor([2.0, 0.0, 0.0], dtype=dtype) + 0.15 * normal
drive = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype)
weights = torch.full((1, 40), 4 * math.pi * 0.15**2 / 8, dtype=dtype)


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)


def cloud(angles):
    ps, ns = [], []
    for a in angles:
        R = rot_z(a)
        ps.append(point @ R.T)
        ns.append(normal @ R.T)
    return torch.cat(ps)[None], torch.cat(ns)[None]


rng = np.random.default_rng(2026_09_08)
pairs = []
while len(pairs) < N_PAIRS:
    a, b = rng.uniform(0, 120, size=2)
    if abs(a - b) >= 10:
        pairs.append((float(a), float(b)))
controls = [(float(rng.uniform(0, 120)), float(rng.uniform(0, 360))) for _ in range(N_CONTROL)]

ARMS = {
    "base": {},
    "base_gauge": {"similarity_gauge": True},
    "local": {"use_local_features": True, "local_radii": (0.2, 0.4)},
    "mom2": {"second_moment_features": True},
}


def probe_accuracy(X, y, folds=5, seed=0):
    """5-fold CV logistic regression (gradient descent, L2 1e-3) on standardized features."""
    X = torch.as_tensor(X, dtype=dtype); y = torch.as_tensor(y, dtype=dtype)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(y), generator=g)
    accs = []
    for f in range(folds):
        test = perm[f::folds]; train = torch.tensor([i for i in perm.tolist() if i not in set(test.tolist())])
        mu, sd = X[train].mean(0), X[train].std(0).clamp_min(1e-12)
        Xt, Xs = (X[train] - mu) / sd, (X[test] - mu) / sd
        w = torch.zeros(X.shape[1], dtype=dtype, requires_grad=True); b = torch.zeros(1, dtype=dtype, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = Xt @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y[train]) + 1e-3 * (w * w).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            accs.append(float((((Xs @ w + b) > 0).double() == y[test]).double().mean()))
    return float(np.mean(accs))


def ridge_r2(X, t, folds=5, seed=0):
    """5-fold CV R^2 of a ridge regression (lambda 1e-3) of a scalar target on standardized features."""
    X = torch.as_tensor(X, dtype=dtype); t = torch.as_tensor(t, dtype=dtype)
    perm = torch.randperm(len(t), generator=torch.Generator().manual_seed(seed))
    ss_res, ss_tot = 0.0, 0.0
    for f in range(folds):
        test = perm[f::folds]; mask = torch.ones(len(t), dtype=torch.bool); mask[test] = False; train = torch.arange(len(t))[mask]
        mu, sd = X[train].mean(0), X[train].std(0).clamp_min(1e-12)
        Xt, Xs = (X[train] - mu) / sd, (X[test] - mu) / sd
        A = Xt.T @ Xt + 1e-3 * torch.eye(X.shape[1], dtype=dtype)
        w = torch.linalg.solve(A, Xt.T @ (t[train] - t[train].mean()))
        pred = Xs @ w + t[train].mean()
        ss_res += float(((pred - t[test]) ** 2).sum()); ss_tot += float(((t[test] - t[train].mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot


results = {"n_pairs": N_PAIRS, "n_controls": N_CONTROL, "seeds": list(SEEDS), "arms": {k: dict(v) for k, v in ARMS.items()},
           "model": {"hidden": 64, "n_layers": 4, "n_slices": 32, "dtype": "float64"}, "per_arm": {}}
t0 = time.time()
for arm, kw in ARMS.items():
    kw = {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()}
    per_seed = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        m = ISLA(hidden=64, n_layers=4, n_slices=32, **ARMS[arm]).double().eval()
        diffs, X, y, phis, sep_by_dphi = [], [], [], [], []
        with torch.no_grad():
            for (a, b) in pairs:
                pa, na = cloud([0, 120, 240, a, a + 180]); pb, nb = cloud([0, 120, 240, b, b + 180])
                oa, ob = m(points=pa, normals=na, drive=drive, measure_weights=weights)[0, :8], m(points=pb, normals=nb, drive=drive, measure_weights=weights)[0, :8]
                diffs.append(float((oa - ob).abs().max()))
                sep_by_dphi.append((abs(a - b), diffs[-1]))
                X.append(oa.reshape(-1).numpy()); y.append(0); phis.append(a)
                X.append(ob.reshape(-1).numpy()); y.append(1); phis.append(b)
            ctrl = []
            for (a, psi) in controls:
                pa, na = cloud([0, 120, 240, a, a + 180]); R = rot_z(psi)
                oa = m(points=pa, normals=na, drive=drive, measure_weights=weights)[0, :8]
                ob = m(points=pa @ R.T, normals=na @ R.T, drive=drive @ R.T, measure_weights=weights)[0, :8]
                ctrl.append(float((oa[..., :1] - ob[..., :1]).abs().max()))  # scalar output on the common component
        ### Preregistered probe (kept for the record): phi and phi' are exchangeable
        ### draws, so the family label is arbitrary and no arm can beat chance.
        acc = probe_accuracy(np.stack(X), np.array(y), seed=seed)
        ### Well-posed substitutes for the same question (is the free azimuth
        ### recoverable from the common component's outputs?): CV R^2 of a ridge
        ### regression of phi, and CV accuracy of a logistic probe for phi >= 60.
        phi_arr = np.array(phis)
        r2 = ridge_r2(np.stack(X), phi_arr, seed=seed)
        acc60 = probe_accuracy(np.stack(X), (phi_arr >= 60).astype(float), seed=seed)
        per_seed.append({"seed": seed, "probe_accuracy_family_label_illposed": acc, "phi_regression_cv_r2": r2, "phi_ge60_probe_accuracy": acc60, "common_component_max_abs_diff": {"median": float(np.median(diffs)), "max": float(np.max(diffs))},
                         "congruent_control_max_abs_diff": float(np.max(ctrl)),
                         "separation_by_dphi_bins": {f"{lo}-{hi}": float(np.median([d for dp, d in sep_by_dphi if lo <= dp < hi]) if any(lo <= dp < hi for dp, _ in sep_by_dphi) else float("nan"))
                                                     for lo, hi in ((10, 30), (30, 60), (60, 90), (90, 120))}})
        print(f"{arm:11s} seed {seed}: family-probe {acc:.3f} (ill-posed)  phi R2 {r2:.3f}  phi>=60 acc {acc60:.3f}  common-comp diff median {np.median(diffs):.2e} max {np.max(diffs):.2e}  control max {np.max(ctrl):.1e}  ({time.time()-t0:.0f}s)", flush=True)
    results["per_arm"][arm] = per_seed
json.dump(results, open(out_path, "w"), indent=1)
print("wrote", out_path)
