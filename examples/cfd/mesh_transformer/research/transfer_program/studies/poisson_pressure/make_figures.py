# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figures and tables for the H1 Poisson-pressure study from the results JSON
(+ optional one-case NPZ). Usage: python make_figures.py results.json [outdir]"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARM = {"v0_isla_qtsdfval_seed42": "ISLA query tokens + SDF (seed 42)", "v0_gt_vol_seed42": "GeoTransolver-volume (seed 42)"}
VARIANT_ORDER = ["MA", "MB", "PA-D-q05", "PB-D-q05", "PA-N0-q05", "PB-N0-q05", "PA-Nm-q05", "PB-Nm-q05",
                 "PA-D-halfh", "PB-D-halfh", "PA-N0-halfh", "PB-N0-halfh", "PA-Nm-halfh", "PB-Nm-halfh"]
VLABEL = {
    "MA": "momentum LS, RHS-A", "MB": "momentum LS, RHS-B",
    "PA-D-q05": "Poisson A, wall p=true", "PB-D-q05": "Poisson B, wall p=true",
    "PA-N0-q05": "Poisson A, wall dp/dn=0", "PB-N0-q05": "Poisson B, wall dp/dn=0",
    "PA-Nm-q05": "Poisson A, wall dp/dn=n.f", "PB-Nm-q05": "Poisson B, wall dp/dn=n.f",
    "PA-D-halfh": "Poisson A, wall p=true (half-h set)", "PB-D-halfh": "Poisson B, wall p=true (half-h set)",
    "PA-N0-halfh": "Poisson A, dp/dn=0 (half-h set)", "PB-N0-halfh": "Poisson B, dp/dn=0 (half-h set)",
    "PA-Nm-halfh": "Poisson A, dp/dn=n.f (half-h set)", "PB-Nm-halfh": "Poisson B, dp/dn=n.f (half-h set)",
}


def main():
    res_path = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else res_path.parent
    R = json.load(open(res_path))
    C = R["cases"]
    cases = sorted(C)
    runs = R["args"]["runs"]
    main_spec = R["args"]["main"]
    stem = res_path.stem

    # ---------------- figure 1: G0 operator tests by band and operator ----------------
    bands = ["sdf<0.01L", "sdf<0.05L", "sdf>=0.05L", "all"]
    specs = list(C[cases[0]]["g0"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, lam, title in zip(axes, ["lam1", "lam0.25"], ["wavelength 1 m", "wavelength 0.25 m"]):
        x = np.arange(len(bands))
        w = 0.8 / (2 * len(specs))
        for si, spec in enumerate(specs):
            for mi, (m, hatch) in enumerate([("grad", ""), ("lap", "//")]):
                vals = np.array([[C[c]["g0"][spec][lam][b][m] for b in bands] for c in cases])
                med = np.median(vals, 0)
                q1, q3 = np.quantile(vals, 0.25, 0), np.quantile(vals, 0.75, 0)
                k, d = spec.split(":")
                ax.bar(x + (2 * si + mi - len(specs) + 0.5) * w, med, w, yerr=[med - q1, q3 - med],
                       label=f"{'gradient' if m == 'grad' else 'Laplacian'}, k={k}, degree {d}", hatch=hatch,
                       color=f"C{si}", alpha=0.9 if m == "grad" else 0.5, capsize=2)
        ax.axhline(0.10, color="k", ls="--", lw=1, label="10% bar" if lam == "lam1" else None)
        ax.set_xticks(x)
        ax.set_xticklabels(bands)
        ax.set_yscale("log")
        ax.set_title(f"analytic test field, {title}")
        ax.set_xlabel("point band (SDF relative to L_ref = 5 m)")
    axes[0].set_ylabel("relative L2 error of the meshfree operator (median over 48 cars, IQR)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_g0_operator.png", dpi=150)
    plt.close(fig)

    # ---------------- figure 2: G1 recovered-from-true error per variant ----------------
    g1 = {c: C[c]["g1"][main_spec] for c in cases}
    variants = [v for v in VARIANT_ORDER if v in g1[cases[0]]]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    data_all = [[g1[c][v]["all"] for c in cases] for v in variants]
    data_nw = [[g1[c][v]["sdf<0.01L"] for c in cases] for v in variants]
    pos = np.arange(len(variants))
    b1 = ax.boxplot(data_all, positions=pos - 0.18, widths=0.32, patch_artist=True, showfliers=False)
    b2 = ax.boxplot(data_nw, positions=pos + 0.18, widths=0.32, patch_artist=True, showfliers=False)
    for p in b1["boxes"]:
        p.set_facecolor("C0")
    for p in b2["boxes"]:
        p.set_facecolor("C1")
    ax.axhline(0.10, color="k", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(pos)
    ax.set_xticklabels([VLABEL[v] for v in variants], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("relative L2 of recovered pressure vs true pressure")
    ax.set_title("G1: pressure recovered from the TRUE velocity (and true nu_t), 48 DrivAerML validation cars, k=40, degree 2")
    ax.legend([b1["boxes"][0], b2["boxes"][0], ax.lines[-1]], ["all 10,000 points", "near-wall band SDF < 0.01 L_ref", "G1 bar 0.10"], fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_g1_true.png", dpi=150)
    plt.close(fig)

    # ---------------- figure 3: G2 recovered-from-predicted vs direct per arm ----------------
    fig, axes = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, runs):
        direct = [C[c]["direct"][r]["all"] for c in cases]
        data = [[C[c]["g2"][r][v]["all"] for c in cases] for v in variants]
        b = ax.boxplot(data, positions=np.arange(len(variants)), widths=0.6, patch_artist=True, showfliers=False)
        for p in b["boxes"]:
            p.set_facecolor("C2")
        md = float(np.mean(direct))
        ax.axhline(md, color="C3", lw=1.5, label=f"directly predicted pressure, mean {md:.4f}")
        ax.axhline(0.9 * md, color="C3", lw=1, ls="--", label="0.9x direct (supported) / 1.1x direct (falsified)")
        ax.axhline(1.1 * md, color="C3", lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels([VLABEL[v] for v in variants], rotation=35, ha="right", fontsize=8)
        ax.set_title(f"G2: recovered from PREDICTED velocity, {ARM.get(r, r)}", fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("relative L2 vs true pressure (all points), box over 48 cars")
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_g2_pred.png", dpi=150)
    plt.close(fig)

    # ---------------- figure 4: one case, scatter of pressures ----------------
    npz = res_path.with_suffix(".case.npz")
    if npz.exists():
        Z = np.load(npz)
        p_true = Z["p_true"]
        best = min(variants, key=lambda v: np.mean([g1[c][v]["all"] for c in cases]))
        panels = [("true_" + best, f"from TRUE velocity, {VLABEL[best]}")]
        for r in runs:
            panels.append((f"{r}_direct", f"direct prediction, {ARM.get(r, r)}"))
            panels.append((f"{r}_{best}", f"from PREDICTED velocity, {ARM.get(r, r)}, {VLABEL[best]}"))
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), sharex=True, sharey=True)
        lim = np.quantile(np.abs(p_true), 0.999)
        for ax, (key, lab) in zip(axes, panels):
            p = Z[key]
            e = float(np.linalg.norm(p - p_true) / np.linalg.norm(p_true))
            ax.scatter(p_true, p, s=2, c=np.log10(np.maximum(Z["sdf"], 1e-5)), cmap="viridis", alpha=0.6)
            ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_title(f"{lab}\nrel L2 {e:.3f}", fontsize=8)
            ax.set_xlabel("true pressure [Pa, gauge, rho = 1]")
        axes[0].set_ylabel("recovered / predicted pressure [Pa]")
        fig.suptitle("case 00015 (run_112): colour = log10 SDF [m]", fontsize=9)
        fig.tight_layout()
        fig.savefig(outdir / f"{stem}_case_scatter.png", dpi=150)
        plt.close(fig)

    # ---------------- tables ----------------
    A = R.get("aggregate")
    if A is None:
        return
    lines = []
    lines.append("## G0 (mean over cars): operator error, wavelength 1 m\n")
    lines.append("| operator | band | gradient | Laplacian |\n|---|---|---|---|")
    for spec in specs:
        for b in bands:
            g = A["g0"][spec]["lam1"][b]
            lines.append(f"| k={spec.split(':')[0]}, deg {spec.split(':')[1]} | {b} | {g['grad']['mean']:.3f} | {g['lap']['mean']:.3f} |")
    lines.append("\n## G1 (mean over cars): recovered from TRUE fields, main operator\n")
    lines.append("| variant | all | SDF<0.01L | SDF<0.05L | SDF>=0.05L | offset removed |\n|---|---|---|---|---|---|")
    for v in variants:
        g = A["g1"][main_spec][v]
        lines.append(f"| {VLABEL[v]} | {g['all']['mean']:.3f} | {g['sdf<0.01L']['mean']:.3f} | {g['sdf<0.05L']['mean']:.3f} | {g['sdf>=0.05L']['mean']:.3f} | {g['offset_only']['mean']:.3f} |")
    for spec in A["g1"]:
        if spec == main_spec:
            continue
        lines.append(f"\nSensitivity operator {spec}: " + ", ".join(f"{v} {A['g1'][spec][v]['all']['mean']:.3f}" for v in A["g1"][spec] if v != "div_rel"))
    lines.append("\n## G2 (mean over cars): recovered from PREDICTED fields vs direct\n")
    lines.append("| arm | direct pressure | velocity | best recovered variant | recovered (all) | ratio | recovered SDF<0.01L | direct SDF<0.01L |\n|---|---|---|---|---|---|---|---|")
    for r in runs:
        d = A["direct"][r]
        best = min(variants, key=lambda v: A["g2"][r][v]["all"]["mean"])
        g = A["g2"][r][best]
        lines.append(f"| {ARM.get(r, r)} | {d['all']['mean']:.4f} | {d['velocity']['mean']:.4f} | {VLABEL[best]} | {g['all']['mean']:.4f} | {g['all']['mean']/d['all']['mean']:.2f} | {g['sdf<0.01L']['mean']:.4f} | {d['sdf<0.01L']['mean']:.4f} |")
    lines.append("\nAll G2 variants (all points): ")
    for r in runs:
        lines.append(f"- {ARM.get(r, r)}: " + ", ".join(f"{VLABEL[v]} {A['g2'][r][v]['all']['mean']:.3f}" for v in variants))
    lines.append(f"\nfar set: mode {C[cases[0]]['far']['mode']}, mean n {A['far_n']['mean']:.0f}; rms(true p on far)/rms(true p) {A['far_p_rms_ratio']['mean']:.3f}; hull n {A['hull_n']['mean']:.0f}; mean case time {A['t_case']['mean']:.0f}s")
    lines.append("div_rel true fields: %.4f; predicted: %s" % (A["g1"][main_spec]["div_rel"]["mean"], ", ".join(f"{r} {A['g2'][r]['div_rel']['mean']:.4f}" for r in runs)))
    (outdir / f"{stem}_tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
