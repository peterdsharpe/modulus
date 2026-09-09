"""Figures for the composition gate study (reads results.json next to this file)."""
import json, os, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "results.json")))
BLUE, RED, GREY, GREEN, ORANGE = "#2a78d6", "#e34948", "#52514e", "#3a9d5d", "#bc6421"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})

prop = [r for r in res["propagation"] if r["variant"] == "diffusion"]
# ---- Figure 1: global energy error vs N at eps = 1e-2, per perturbation kind, chains and grids
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, lay in zip(axes, ("chain", "grid")):
    Ns = sorted({r["N"] for r in prop if r["layout"] == lay})
    for pk, color, mk in (("random", BLUE, "o"), ("low", ORANGE, "s"), ("high", GREEN, "^")):
        med = [st.median(r["perturbed"][f"{pk}_0.01"]["energy"] for r in prop if r["layout"] == lay and r["N"] == N) for N in Ns]
        lo = [min(r["perturbed"][f"{pk}_0.01"]["energy"] for r in prop if r["layout"] == lay and r["N"] == N) for N in Ns]
        hi = [max(r["perturbed"][f"{pk}_0.01"]["energy"] for r in prop if r["layout"] == lay and r["N"] == N) for N in Ns]
        ax.fill_between(Ns, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(Ns, med, color=color, marker=mk, label=f"local error on {pk} modes" if pk != "random" else "random local error")
    ref = med[0] if False else None
    base = [st.median(r["perturbed"]["random_0.01"]["energy"] for r in prop if r["layout"] == lay and r["N"] == Ns[0])][0]
    ax.plot(Ns, [base * (N / Ns[0]) ** 0.5 for N in Ns], color=GREY, ls="--", lw=1, label="√N reference")
    ax.plot(Ns, [base * (N / Ns[0]) for N in Ns], color=GREY, ls=":", lw=1, label="linear-in-N reference")
    ax.axhline(0.05, color=RED, lw=0.8, ls="-.", label="5% bar")
    ax.set_yscale("log"); ax.set_xscale("log", base=2); ax.set_xticks(Ns); ax.set_xticklabels([str(N) for N in Ns])
    ax.set_xlabel("Components in the assembly N [–]"); ax.set_title(f"{lay}s")
axes[0].set_ylabel("Global relative energy error [–]"); axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Every local port response perturbed by 1e-2 relative Frobenius error; median and range over 5 draws", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_amplification.png")); plt.close(fig)

# ---- Figure 2: rank scan (transfer-optimal vs POD modes; library vs new shapes)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
for ax, (lay, N) in zip(axes, (("chain", 8), ("grid", 8), ("grid", 16))):
    for label, ls in (("library", "-"), ("new_shapes", "--")):
        rows = [r for r in res["rank_scan"] if r["shapes"] == label and r["layout"] == lay and r["N"] == N]
        for name, color in (("transfer", BLUE), ("pod", ORANGE)):
            ranks = sorted(int(k) for k in rows[0][name]); med = [st.median(row[name][str(k)] if str(k) in row[name] else row[name][k] for row in rows) for k in ranks]
            ax.plot(ranks, med, color=color, ls=ls, marker="o" if name == "transfer" else "s", ms=3,
                    label=f"{'transfer-optimal' if name == 'transfer' else 'POD'} modes, {label.replace('_', ' ')}")
    ax.axhline(0.01, color=RED, lw=0.8, ls="-.", label="1% bar"); ax.axvline(4, color=GREY, lw=0.8, ls=":", label="rank 4 (quarter of 15)")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 3); ax.set_xlabel("Modes per shared edge r (of 15) [–]"); ax.set_title(f"{lay}, N = {N} (r = 15 is exact, below the axis)")
axes[0].set_ylabel("Global relative energy error [–]"); axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_rank_scan.png")); plt.close(fig)

# ---- Figure 3: interface conditioning, unknown counts, PCG iterations vs N
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for lay, color, mk in (("chain", BLUE, "o"), ("grid", ORANGE, "s")):
    Ns = sorted({r["N"] for r in prop if r["layout"] == lay})
    rows = lambda N: [r for r in prop if r["layout"] == lay and r["N"] == N]
    axes[0].plot(Ns, [st.median(r["cond_S_II"] for r in rows(N)) for N in Ns], color=color, marker=mk, label=lay)
    axes[1].plot(Ns, [rows(N)[0]["n_I"] for N in Ns], color=color, marker=mk, label=f"{lay}: interface unknowns")
    axes[1].plot(Ns, [rows(N)[0]["n_dof"] for N in Ns], color=color, marker=mk, ls="--", label=f"{lay}: monolithic unknowns")
    axes[2].plot(Ns, [st.median(r["pcg_iters"]["block_jacobi"] for r in rows(N)) for N in Ns], color=color, marker=mk, label=f"{lay}: block Jacobi")
    axes[2].plot(Ns, [st.median(r["pcg_iters"]["two_level"] for r in rows(N)) for N in Ns], color=color, marker=mk, ls="--", label=f"{lay}: + coarse space")
axes[0].set(yscale="log", xlabel="Components N [–]", ylabel="cond(S_II) [–]", title="Interface conditioning"); axes[0].legend(frameon=False, fontsize=8)
axes[1].set(yscale="log", xlabel="Components N [–]", ylabel="Unknowns [count]", title="Problem size"); axes[1].legend(frameon=False, fontsize=7)
axes[2].set(xlabel="Components N [–]", ylabel="PCG iterations to 1e-8 [count]", title="Interface Krylov solve"); axes[2].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cost.png")); plt.close(fig)

# ---- Figure 4: spectra of transfer operator and POD snapshots
fig, ax = plt.subplots(figsize=(5.5, 3.6))
sv = np.array(res["transfer_singular_values"]); pv = np.array(res["pod_singular_values"])
ax.semilogy(np.arange(1, len(sv) + 1), sv / sv[0], color=BLUE, marker="o", ms=3, label="transfer operator (plain reference pair)")
ax.semilogy(np.arange(1, len(pv) + 1), pv / pv[0], color=ORANGE, marker="s", ms=3, label="POD of interface traces (library assemblies)")
ax.set(xlabel="Mode index [–]", ylabel="Normalized singular value [–]", title="Port-mode spectra (15 edge-interior nodes)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_spectra.png")); plt.close(fig)
print("figures written")
