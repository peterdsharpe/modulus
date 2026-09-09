"""Figures for the C3 paired-condition control (self-contained: legends, labeled axes).

fig_sphere_cp.png      C_p on the unit sphere along the streamwise angle, free air vs ground effect
                       (reference labels) with arm A's and arm B's seed-0 predictions.
fig_error_vs_floor.png Per-shape relative L2 error of arm A (current encoding) on held-out shapes vs the
                       analytic equal-mixture floor; arms B, C and M for comparison.
Usage: python make_figures.py <dir with labels.npz and results.json>
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "."
z = np.load(f"{D}/labels.npz"); R = json.load(open(f"{D}/results.json"))
BLUE, RED, GREY, GREEN = "#2a78d6", "#e34948", "#52514e", "#3a6b3a"

# --- sphere figure -----------------------------------------------------------------------------
cen = z["case00__cen"]; th = np.degrees(np.arccos(np.clip(cen[:, 0] / np.linalg.norm(cen, axis=1), -1, 1)))
order = np.argsort(th)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for k, (cond, title) in enumerate((("free", "free air"), ("ground", "ground plane at 0.6 body half-heights"))):
    y = z[f"case00__cp_{cond}"]
    ax[k].scatter(th, y, s=4, color=GREY, label="reference (potential flow)")
    for arm, color in (("A", RED), ("B", BLUE)):
        pred = R["arms"][arm]["0"].get("sphere_pred", {}).get(cond)
        if pred is not None:
            ax[k].scatter(th, pred, s=4, color=color, alpha=0.6, label=f"arm {arm} ({'no condition input' if arm == 'A' else 'ground-proximity input'}), seed 0")
    ax[k].set(title=f"Unit sphere, {title}", xlabel="angle from the +x (freestream) axis [deg]")
    ax[k].spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("pressure coefficient C_p [–]"); ax[0].legend(frameon=False, fontsize=8, loc="lower center")
fig.tight_layout(); fig.savefig(f"{D}/fig_sphere_cp.png", dpi=130)

# --- error vs floor ----------------------------------------------------------------------------
test = R["test_shapes"]; floor = R["floor"]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
x = np.arange(len(test)); w = 0.18
fl = np.array([(floor[c]["free"] + floor[c]["ground"]) / 2 for c in test])
ax.bar(x - 1.5 * w, fl, w, color=GREY, label="analytic equal-mixture floor (paired labels)")
for j, (arm, color, lab) in enumerate((("A", RED, "arm A: current encoding, mixed conditions"), ("M", "#bc6421", "arm M: per-point MLP, mixed conditions"),
                                          ("B", BLUE, "arm B: + ground-proximity scalar"), ("C", GREEN, "arm C: free air only, evaluated in free air (in-distribution reference)"))):
    vals = []
    for c in test:
        per_seed = []
        for seed, rec in R["arms"][arm].items():
            conds = ["free"] if arm == "C" else [cd for cd in ("free", "ground") if cd in rec["test"] and c in rec["test"][cd]]   # arm C: in-distribution only
            per_seed.append(np.mean([rec["test"][cd][c] for cd in conds]))
        vals.append(np.mean(per_seed))
    ax.bar(x + (j - 0.5) * w, vals, w, color=color, label=lab)
ax.set(xticks=x, xticklabels=[c.replace("case", "") for c in test], xlabel="held-out shape (case id)", ylabel="relative L2 error of C_p, seed mean (arms A, B, M: mean over both conditions) [–]")
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{D}/fig_error_vs_floor.png", dpi=130)
print("figures written")
