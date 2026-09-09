"""Reduce solver_spaces_results.json to the preregistered readouts and draw the figures.

Readouts (PREREG.md): per held-out set and probe class, mean error contraction per space and
the ratio classical / frozen-source (>= 2 within family, >= 1.5 topology shift are the
"advances" bars); residual-vs-error contraction ratio (residual-not-error failure if > 10);
complete-solve work with setup amortized over N_RHS_AMORTIZE right-hand sides; break-even.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
R = json.loads((D / "solver_spaces_results.json").read_text())
AMORT = R["config"]["amortize_rhs"]
SPACES = ["src_frozen", "oracle_pod", "aggregation", "smoothed_aggregation", "geometric"]
CLASSICAL = ["aggregation", "smoothed_aggregation", "geometric"]
LABEL = {"src_frozen": "frozen source POD", "oracle_pod": "per-target oracle POD", "aggregation": "aggregation",
         "smoothed_aggregation": "smoothed aggregation", "geometric": "geometric coarse grid", "none": "no coarse space"}
COL = {"src_frozen": "#2a78d6", "oracle_pod": "#52514e", "aggregation": "#bc6421", "smoothed_aggregation": "#e34948",
       "geometric": "#3a9d5d", "none": "#898781"}
summary = {"config": R["config"], "pde": {}}

for pde, P in R["pde"].items():
    S = {"ranks": {}}
    for r, sets in P["ranks"].items():
        S_r = {}
        for set_name, geos in sets.items():
            rows = list(geos.values())
            contr = {sp: {cls: float(np.mean([g["contraction"][sp][cls][0] for g in rows])) for cls in ("random", "slow", "near_null")} for sp in SPACES}
            rescontr = {sp: {cls: float(np.mean([g["contraction"][sp][cls][1] for g in rows])) for cls in ("random", "slow", "near_null")} for sp in SPACES}
            best_classical = {cls: min(CLASSICAL, key=lambda sp: contr[sp][cls]) for cls in contr["src_frozen"]}
            ratio_vs_src = {cls: contr[best_classical[cls]][cls] / contr["src_frozen"][cls] for cls in contr["src_frozen"]}
            # residual-not-error: how much more the residual contracts than the error (error_left / residual_left)
            res_vs_err = {sp: {cls: (contr[sp][cls] / max(rescontr[sp][cls], 1e-300)) for cls in contr[sp]} for sp in SPACES}
            # solves: work including amortized setup
            solve = {}
            for sp in SPACES + ["none", "smoothed_aggregation+mean_x0"]:
                its = [g["solve"][sp]["iterations"] for g in rows]; conv = [g["solve"][sp]["converged"] for g in rows]
                work = [g["solve"][sp]["iterations"] * g["solve"][sp]["work_iter"] + g["solve"][sp]["setup"] / AMORT for g in rows]
                solve[sp] = {"iterations_mean": float(np.mean(its)), "iterations_max": int(np.max(its)), "all_converged": bool(all(conv)),
                             "work_mean_amortized": float(np.mean(work)),
                             "work_iter_mean": float(np.mean([g["solve"][sp]["work_iter"] for g in rows])),
                             "setup_mean": float(np.mean([g["solve"][sp]["setup"] for g in rows]))}
            bc = min(CLASSICAL, key=lambda sp: solve[sp]["work_mean_amortized"])
            saving = solve[bc]["work_mean_amortized"] - solve["src_frozen"]["work_mean_amortized"]
            per_solve_saving = (solve[bc]["iterations_mean"] * solve[bc]["work_iter_mean"]) - (solve["src_frozen"]["iterations_mean"] * solve["src_frozen"]["work_iter_mean"])
            setup_diff = solve["src_frozen"]["setup_mean"] - solve[bc]["setup_mean"]
            S_r[set_name] = {"n_geometries": len(rows), "registration_loss_mean": float(np.mean([g["registration_loss"] for g in rows])),
                             "dims": rows[0]["dims"], "error_contraction": contr, "residual_contraction": rescontr,
                             "best_classical_by_class": best_classical, "ratio_best_classical_over_src": ratio_vs_src,
                             "error_left_over_residual_left": res_vs_err, "solve": solve, "best_classical_solve": bc,
                             "work_saving_fraction_vs_best_classical": saving / solve[bc]["work_mean_amortized"],
                             "break_even_rhs": (setup_diff / per_solve_saving) if per_solve_saving > 0 else None}
        S["ranks"][r] = S_r
    S["pod_singular_values"] = P["pod_singular_values"]
    summary["pde"][pde] = S
(D / "solver_spaces_summary.json").write_text(json.dumps(summary, indent=1))

# ------------------------------------------------------------------ figures
for pde, P in R["pde"].items():
    for r in P["ranks"]:
        S_r = summary["pde"][pde]["ranks"][r]
        sets = list(S_r.keys()); classes = ["random", "slow", "near_null"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for ax, cls in zip(axes, classes):
            x = np.arange(len(sets)); w = 0.15
            for k, sp in enumerate(SPACES):
                vals = [S_r[s]["error_contraction"][sp][cls] for s in sets]
                ax.bar(x + (k - 2) * w, vals, w, color=COL[sp], label=LABEL[sp])
            ax.set_xticks(x); ax.set_xticklabels([s.replace("_", "\n") for s in sets], fontsize=8)
            ax.set_title(f"{cls.replace('_', '-')} probes"); ax.set_ylim(0, 1.05)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("error left after one coarse correction [–]" if pde == "diffusion" else "Euclidean error left [–]")
        axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
        fig.suptitle(f"{pde}: one-step error contraction by correction space, rank {r} (lower is better; mean over held-out geometries)", fontsize=10)
        fig.tight_layout(); fig.savefig(D / f"fig_contraction_{pde}_r{r}.png", dpi=130); plt.close(fig)
        # residual vs error curves for one topology-shift system
        geos = P["ranks"][r]["topology_shift"]; tag = sorted(geos)[0]; g = geos[tag]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for sp in SPACES + ["none"]:
            sol = g["solve"][sp]
            axes[0].semilogy(sol["residual_hist"], color=COL[sp], label=LABEL[sp])
            axes[1].semilogy(np.maximum(sol["error_hist"], 1e-16), color=COL[sp], label=LABEL[sp])
        axes[0].set(xlabel="iteration [–]", ylabel="relative residual [–]"); axes[1].set(xlabel="iteration [–]", ylabel=("A-norm error [–]" if pde == "diffusion" else "Euclidean error [–]"))
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        axes[1].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{pde}: complete solve on topology-shift system {tag}, rank {r} (additive two-level preconditioner)", fontsize=10)
        fig.tight_layout(); fig.savefig(D / f"fig_solve_{pde}_r{r}.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.semilogy(np.arange(1, len(P["pod_singular_values"]) + 1), P["pod_singular_values"], "o-", color="#2a78d6", ms=3)
    ax.set(xlabel="POD mode index [–]", ylabel="normalized singular value [–]", title=f"{pde}: source error-snapshot POD spectrum")
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(D / f"fig_pod_{pde}.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------------ console table
for pde, S in summary["pde"].items():
    print(f"== {pde}")
    for r, S_r in S["ranks"].items():
        for s, v in S_r.items():
            rc = v["ratio_best_classical_over_src"]; sv = v["solve"]
            print(f" r={r} {s:17s} regloss {v['registration_loss_mean']:.3f} | slow: src {v['error_contraction']['src_frozen']['slow']:.3f} best-cls {v['error_contraction'][v['best_classical_by_class']['slow']]['slow']:.3f} ({v['best_classical_by_class']['slow']}) ratio {rc['slow']:.2f} | near-null ratio {rc['near_null']:.2f} random ratio {rc['random']:.2f} | oracle slow {v['error_contraction']['oracle_pod']['slow']:.3f}")
            print(f"     err/res-left max over spaces&classes: {max(x for sp in v['error_left_over_residual_left'].values() for x in sp.values()):.1f} | solves (amortized work): src {sv['src_frozen']['work_mean_amortized']:.0f} ({sv['src_frozen']['iterations_mean']:.0f} it) best-cls {sv[v['best_classical_solve']]['work_mean_amortized']:.0f} ({sv[v['best_classical_solve']]['iterations_mean']:.0f} it, {v['best_classical_solve']}) none {sv['none']['iterations_mean']:.0f} it | saving {100*v['work_saving_fraction_vs_best_classical']:+.0f}% break-even {v['break_even_rhs']} | src converged {sv['src_frozen']['all_converged']}")
