"""Figures for the boundary-response gate, from results.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
R = json.loads((HERE / "results.json").read_text())
C = R["cases"]
BLUE, RED, GREY, ORANGE = "#2a78d6", "#e34948", "#52514e", "#bc6421"

# 1. singular-value decay, self-relative, for representative geometries, both pairings
reps = ["ellipse_a2.0", "star_a0.3_m5", "pair_gL0.02", "annulus_r0.85", "square_p32"]
fig, axes = plt.subplots(2, len(reps), figsize=(3.2 * len(reps), 6.2), sharey="row")
for col, name in enumerate(reps):
    for row, pairing in enumerate(("L2", "energy")):
        ax = axes[row, col]; d = C[name][pairing]
        for key, lab, color in (("sv_full", "full DtN Λ", RED), ("sv_residual", "Λ − principal part", BLUE), ("sv_nullfree", "residual, constants removed", ORANGE)):
            s = np.array(d[key]); s = s / s[0]
            ax.semilogy(np.arange(1, len(s) + 1), s, color=color, lw=1.6, label=lab)
        ax.set_title(f"{name}\n{pairing} pairing", fontsize=9)
        ax.set_xlabel("singular value index [–]")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("σ_i / σ_1 [–]")
axes[0, 0].legend(fontsize=8, frameon=False)
fig.suptitle("2D Laplace DtN: singular values of the full operator and of the residuals after exact principal-part / null-mode removal", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "fig_sv_decay.png", dpi=130)

# 2. ranks vs gap / corner sharpness (L2 pairing, 1e-2), full vs nullfree residual; and 2N check
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
gaps = sorted([(v["gL"], k) for k, v in C.items() if k.startswith("pair_")])
x = [g for g, _ in gaps]
for key, lab, color, mk in (("rank_full", "full DtN Λ (bodies block)", RED, "o"), ("rank_residual", "Λ − principal part", BLUE, "s"), ("rank_residual_nullfree", "residual, constants removed", ORANGE, "^")):
    axes[0].plot(x, [C[k]["L2"][key]["0.01"] for _, k in gaps], color=color, marker=mk, label=f"{lab}, N")
    axes[0].plot(x, [C[k]["at_2N"]["L2"][key]["0.01"] for _, k in gaps], color=color, marker=mk, ls="--", alpha=0.6, label=f"{lab}, 2N")
axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel("gap / body diameter g/L [–]"); axes[0].set_ylabel("rank at 1% self-relative Frobenius error [–]")
axes[0].set_title("Two unit circles in a grounded circle of radius 4", fontsize=9); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7, frameon=False)
corners = sorted([(v["rhoL"], k) for k, v in C.items() if k.startswith("square_")])
x = [r for r, _ in corners]
for key, lab, color, mk in (("rank_full", "full DtN Λ", RED, "o"), ("rank_residual", "Λ − principal part", BLUE, "s"), ("rank_residual_nullfree", "residual, constants removed", ORANGE, "^")):
    axes[1].plot(x, [C[k]["L2"][key]["0.01"] for _, k in corners], color=color, marker=mk, label=f"{lab}, N")
    axes[1].plot(x, [C[k]["at_2N"]["L2"][key]["0.01"] for _, k in corners], color=color, marker=mk, ls="--", alpha=0.6, label=f"{lab}, 2N")
axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel("corner radius / side ρ/L [–]"); axes[1].set_title("Rounded squares (superellipse, even p)", fontsize=9)
axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7, frameon=False)
fig.suptitle("Rank needed at 1% (L2 pairing): solid at the working resolution N, dashed at 2N. The full operator's rank is the node count (resolution-bound); the residuals' ranks are not.", fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "fig_rank_scaling.png", dpi=130)

# 3. transfer: source-fitted basis vs oracle, per held-out block (energy and L2, nullfree residual and full)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, pairing in zip(axes, ("L2", "energy")):
    for obj, color, mk in (("full", RED, "o"), ("nullfree", ORANGE, "^")):
        rows = R["transfer"][f"{pairing}/{obj}"]
        xs = [v["oracle_err"] for v in rows.values()]; ys = [v["source_basis_err"] for v in rows.values()]
        ax.scatter(xs, ys, color=color, marker=mk, s=28, alpha=0.8, label=f"{obj}: held-out blocks")
    lim = (1e-4, 1.5)
    ax.plot(lim, lim, color=GREY, lw=1, label="source basis = oracle"); ax.plot(lim, [2 * l for l in lim], color=GREY, lw=1, ls="--", label="2x oracle (bar)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("oracle projection error at the block's 1% rank [–]"); ax.set_title(f"{pairing} pairing", fontsize=9); ax.grid(alpha=0.3)
axes[0].set_ylabel("source-fitted basis projection error at the same rank [–]"); axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("Basis transfer: source single bodies + wide pairs → held-out gaps, annuli, corners (|k| ≤ 48 arclength-Fourier coordinates)", fontsize=10)
fig.tight_layout(); fig.savefig(HERE / "fig_transfer.png", dpi=130)
print("figures written")
