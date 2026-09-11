"""Train the C3 arms on the qualified potential-flow labels (PREREG.md) and evaluate.

Arms: A current encoding (mixed conditions, no condition input); B complete encoding
(mixed, plus the per-point ground-proximity scalar s = 1/(1 + z/L), 0 in free air);
C in-distribution reference (free air only); M per-point MLP on body-frame coordinates,
normals and drive (mixed, no condition input). Three seeds each. Split: 30 training
shapes, 10 held-out shapes (the unit sphere is case 0 and is held out for the figure).

Usage: python train_arms.py <dir with labels.npz> [steps]
"""
import json, sys, time
import numpy as np
import torch

import os
sys.path.insert(0, os.environ.get("PHYSICSNEMO_SRC", "/home/psharpe/gh/physicsnemo/.claude/worktrees/agent-a2e037b2d41bdf9e1"))
from physicsnemo.experimental.nn.isla import ISLA

D = sys.argv[1] if len(sys.argv) > 1 else "."
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
torch.set_num_threads(int(os.environ.get("C3_THREADS", "8")))
z = np.load(f"{D}/labels.npz")
all_cases = sorted({k.split("__")[0] for k in z.files})
cases = [c for c in all_cases if bool(z[f"{c}__qualified"])]        # only cases that passed the reference gate (Q2, Q3 per case)
data = {}
for c in cases:
    g = lambda kk: z[f"{c}__{kk}"]
    data[c] = dict(params=g("params"), L=float(g("L")), h=float(g("h")), z_plane=float(g("z_plane")), cen=g("cen"), nrm=g("nrm"), area=g("area"),
                   cp={"free": g("cp_free"), "ground": g("cp_ground")})
rng = np.random.default_rng(0)
shapes = [c for c in cases if c != "case00"]; rng.shuffle(shapes)
n_test = max(8, len(shapes) // 4)
train_shapes, test_shapes = shapes[n_test:], shapes[:n_test] + (["case00"] if "case00" in cases else [])
print(f"{len(cases)} qualified of {len(all_cases)} cases; train shapes {len(train_shapes)}, test shapes {len(test_shapes)}", flush=True)
drive = torch.tensor([[1.0, 0.0, 0.0]])


def tensors(c, cond):
    d = data[c]
    pts = torch.tensor(d["cen"])[None]; nrm = torch.tensor(d["nrm"])[None]; w = torch.tensor(d["area"])[None]
    y = torch.tensor(d["cp"][cond])[None, :, None]
    zabove = (d["cen"][:, 2] - d["z_plane"]) / d["L"]
    s = torch.tensor((1.0 / (1.0 + zabove)).astype(np.float32))[None, :, None] if cond == "ground" else torch.zeros(1, len(zabove), 1)
    return pts, nrm, w, y, s


class MLP(torch.nn.Module):
    def __init__(self, extra=0):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(9 + extra, 128), torch.nn.GELU(), torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 1))

    def forward(self, pts, nrm, drv, w, boundary_scalars=None):
        x = torch.cat([pts, nrm, drv[:, None, :].expand(-1, pts.shape[1], -1)] + ([boundary_scalars] if boundary_scalars is not None else []), -1)
        return self.net(x)


def make(arm, seed):
    torch.manual_seed(seed)
    if arm == "M":
        return MLP()
    return ISLA(hidden=64, n_layers=4, n_slices=32, out_scalars=1, out_vectors=1, n_boundary_scalars=1 if arm == "B" else 0)


def predict(m, arm, c, cond):
    pts, nrm, w, y, s = tensors(c, cond)
    out = m(points=pts, normals=nrm, drive=drive, measure_weights=w, boundary_scalars=s) if arm in ("B",) else (m(points=pts, normals=nrm, drive=drive, measure_weights=w) if arm != "M" else m(points=pts, normals=nrm, drive=drive, measure_weights=w))
    return out[..., :1], y


def rel_l2(pred, y):
    return float(torch.linalg.norm(pred - y) / torch.linalg.norm(y))


def train(arm, seed):
    m = make(arm, seed); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    conds = ["free"] if arm == "C" else ["free", "ground"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    r = np.random.default_rng(seed)
    for step in range(STEPS):
        c = train_shapes[r.integers(len(train_shapes))]; cond = conds[r.integers(len(conds))]
        pred, y = predict(m, arm, c, cond)
        loss = torch.nn.functional.huber_loss(pred, y, delta=0.5)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    m.eval()
    return m


results = {"steps": STEPS, "train_shapes": train_shapes, "test_shapes": test_shapes, "arms": {}, "floor": {}}
# analytic equal-mixture floor per case and condition
for c in cases:
    yf, yg = data[c]["cp"]["free"], data[c]["cp"]["ground"]; half = (yf - yg) / 2
    results["floor"][c] = {"free": float(np.linalg.norm(half) / np.linalg.norm(yf)), "ground": float(np.linalg.norm(half) / np.linalg.norm(yg)),
                           "ground_effect_rms_change": float(np.sqrt(np.mean((yg - yf) ** 2)) / np.sqrt(np.mean(yf ** 2)))}
t0 = time.time()
per_point_err = {}
for arm in ("A", "B", "C", "M"):
    results["arms"][arm] = {}
    for seed in (0, 1, 2):
        m = train(arm, seed)
        rec = {"test": {}, "train": {}}
        with torch.no_grad():
            for split, shp in (("test", test_shapes), ("train", train_shapes)):
                for cond in ("free", "ground"):
                    if arm == "C" and cond == "ground" and split == "train":
                        continue
                    errs = {}
                    for c in shp:
                        pred, y = predict(m, arm, c, cond); errs[c] = rel_l2(pred, y)
                        if arm == "A" and split == "test" and seed == 0:
                            per_point_err[(c, cond)] = ((pred - y)[0, :, 0].numpy(), data[c]["cen"][:, 2], data[c]["z_plane"], data[c]["L"])
                        if c == "case00" and split == "test" and seed == 0:
                            rec.setdefault("sphere_pred", {})[cond] = pred[0, :, 0].numpy().tolist()
                    rec[split][cond] = errs
        results["arms"][arm][str(seed)] = rec
        summ = {f"{sp}_{cd}": float(np.mean(list(v.values()))) for sp, dd in rec.items() if sp in ("test", "train") for cd, v in dd.items()}
        print(f"arm {arm} seed {seed}: {summ}  ({time.time()-t0:.0f}s)", flush=True)
# spatial signature of arm A (seed 0, test shapes): fraction of squared error in the lowest height quarter
sig = []
for (c, cond), (e, zc, zp, L) in per_point_err.items():
    if cond != "ground":
        continue
    hgt = zc - zp; q = np.quantile(hgt, 0.25); frac = float((e[hgt <= q] ** 2).sum() / (e ** 2).sum())
    sig.append(frac)
results["armA_signature_fraction_sq_err_lowest_height_quarter_ground_cases"] = {"mean": float(np.mean(sig)), "min": float(min(sig)), "max": float(max(sig)), "uniform_expectation": 0.25}


def summarize(arm, split, cond):
    vals = []
    for seed, rec in results["arms"][arm].items():
        if cond in rec[split]:
            vals.append(np.mean(list(rec[split][cond].values())))
    return {"seed_mean": float(np.mean(vals)), "seeds": [float(v) for v in vals]} if vals else None


summary = {arm: {f"{sp}_{cd}": summarize(arm, sp, cd) for sp in ("test", "train") for cd in ("free", "ground")} for arm in results["arms"]}
floor_test = {cd: float(np.mean([results["floor"][c][cd] for c in test_shapes])) for cd in ("free", "ground")}
floor_test["paired"] = float(np.mean([floor_test["free"], floor_test["ground"]]))
summary["floor_test_mean"] = floor_test
A_paired = float(np.mean([summary["A"]["test_free"]["seed_mean"], summary["A"]["test_ground"]["seed_mean"]]))
C_in = summary["C"]["test_free"]["seed_mean"]
summary["bars"] = {
    "A_paired_over_floor": A_paired / floor_test["paired"], "A_paired_over_C_in_distribution": A_paired / C_in,
    "floor_confirmed": bool(A_paired >= 0.8 * floor_test["paired"] and A_paired >= 3 * C_in),
    "B_test_free_over_C": summary["B"]["test_free"]["seed_mean"] / C_in, "B_test_ground_over_C": summary["B"]["test_ground"]["seed_mean"] / C_in,
    "repair_confirmed": bool(summary["B"]["test_free"]["seed_mean"] <= 1.5 * C_in and summary["B"]["test_ground"]["seed_mean"] <= 1.5 * C_in),
    "M_paired_over_floor": float(np.mean([summary["M"]["test_free"]["seed_mean"], summary["M"]["test_ground"]["seed_mean"]])) / floor_test["paired"],
    "B_train_fit_over_C_train": float(np.mean([summary["B"]["train_free"]["seed_mean"], summary["B"]["train_ground"]["seed_mean"]])) / summary["C"]["train_free"]["seed_mean"],
}
results["summary"] = summary
json.dump(results, open(f"{D}/results.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
