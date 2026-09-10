# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Figure for the post-hoc wall-distance-subset diagnostic.
Usage: python make_posthoc_figure.py posthoc_subsets.json"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    path = Path(sys.argv[1])
    R = json.load(open(path))
    C = R["cases"]
    keys = [k for k in R["aggregate"]]
    thr = [float(k.split(">=")[1].rstrip("L")) for k in keys]
    order = np.argsort(thr)
    keys = [keys[i] for i in order]
    thr = [thr[i] for i in order]

    def coll(metric):
        return [np.array([c[k][metric] for c in C.values() if k in c]) for k in keys]

    def stats(d):
        med = np.array([np.nanmedian(v) for v in d])
        q1 = np.array([np.nanquantile(v, 0.25) for v in d])
        q3 = np.array([np.nanquantile(v, 0.75) for v in d])
        return med, [med - q1, q3 - med]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(keys))
    labels = [f"SDF ≥ {t:g} L\nn ={np.median([c[k]['n'] for c in C.values() if k in c]):.0f}\nh = {np.median([c[k]['h_med'] for c in C.values() if k in c])*100:.1f} cm" for k, t in zip(keys, thr)]
    ax = axes[0]
    for metric, lab, col in [("MA_true", "recovered from TRUE velocity (momentum least squares)", "C0"),
                             ("direct_v0_isla_qtsdfval_seed42", "ISLA query tokens + SDF, direct prediction", "C3"),
                             ("direct_v0_gt_vol_seed42", "GeoTransolver-volume, direct prediction", "C1")]:
        med, err = stats(coll(metric))
        ax.errorbar(x, med, yerr=err, fmt="o-", color=col, label=lab, capsize=3)
    nan_frac = [np.mean(~np.isfinite(v)) for v in coll("MA_true")]
    for xi, nf in zip(x, nan_frac):
        if nf > 0:
            ax.annotate(f"{nf*100:.0f}% of cars\nsingular", (xi, 30), fontsize=6.5, ha="center", color="C0")
    ax.axhline(0.10, color="k", ls="--", lw=1, label="G1 bar 0.10")
    ax.set_yscale("log")
    ax.set_ylabel("relative L2 vs true pressure on the subset\n(median, IQR over 48 cars)")
    ax.set_title("pressure error on wall-distance subsets", fontsize=10)
    ax.legend(fontsize=7, loc="center left")
    ax = axes[1]
    for metric, lab, col in [("div_rel_true", "|div u|_h / |grad u|_h, TRUE velocity", "C0"),
                             ("div_rel_v0_isla_qtsdfval_seed42", "same, ISLA predicted velocity", "C3"),
                             ("div_rel_v0_gt_vol_seed42", "same, GeoTransolver-volume predicted velocity", "C1"),
                             ("g0_lap", "Laplacian error, analytic 1 m field (operator test)", "C2")]:
        med, err = stats(coll(metric))
        ax.errorbar(x, med, yerr=err, fmt="s-", color=col, label=lab, capsize=3)
    ax.set_yscale("log")
    ax.set_ylabel("dimensionless ratio (median, IQR over 48 cars)")
    ax.set_title("is the velocity field resolved by the sample?", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[2]
    med, err = stats(coll("f_over_gradp_true"))
    ax.errorbar(x, med, yerr=err, fmt="^-", color="C0", label="|(u.grad)u|_h / |grad p_true|_h, TRUE fields", capsize=3)
    ax.axhline(1.0, color="k", ls=":", lw=1, label="1 (momentum balance would give about 1)")
    ax.set_yscale("log")
    ax.set_ylabel("ratio of norms over the subset (median, IQR over 48 cars)")
    ax.set_title("magnitude of the discrete momentum source", fontsize=10)
    ax.legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlabel("subset (n = median points per car, h = median stencil radius)", fontsize=8)
    fig.suptitle("Post-hoc diagnostic (not preregistered): momentum least-squares recovery on subsets by wall distance, RBF-FD k=40 degree 2 rebuilt on each subset", fontsize=9)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=150)
    print("wrote", path.with_suffix(".png"))


if __name__ == "__main__":
    main()
