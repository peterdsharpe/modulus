# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""POST-HOC diagnostic for the H1 study (written after G1 failed on the full
sample): repeat the momentum least-squares recovery on wall-distance-filtered
subsets of the 10,000 points, to locate where the constraint solve breaks.
Not a preregistered test; reported as a diagnostic only.

For each threshold s in SDF >= s * L_ref: build the RBF-FD operator on the
subset alone, compute f = -rho (u . grad) u from the TRUE (and predicted)
velocity on the subset, solve min |G p - f|^2 with mean(p[far]) = 0, and
report the relative L2 against the true pressure on the subset, the
discrete divergence of the true velocity relative to its gradient norm on
the subset, and the analytic-field Laplacian error of the subset operator.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import poisson_pressure as pp

THRESH = [0.0, 0.0005, 0.002, 0.01, 0.03, 0.1]


def one_case(job):
    case, run_dirs, runs = job
    flds = {r: pp.load_case(run_dirs[r] / case) for r in runs}
    f0 = flds[runs[0]]
    pts, sdf, L = f0["points"], f0["sdf"], f0["L_ref"]
    p_true = f0["p_true"] - f0["p_inf"]
    out = {}
    for s in THRESH:
        m = sdf >= s * L
        n = int(m.sum())
        if n < 200:
            continue
        P, S = pts[m], sdf[m]
        ops = pp.Ops(P, 40, 2)
        far = S >= pp.FAR_FRAC_L * L
        if far.sum() < 10:
            far = S >= np.quantile(S, 0.99)
        g0 = pp.g0_tests(ops, P, {"all": np.ones(n, bool)}, lams=(1.0,))["lam1"]["all"]
        r = {"n": n, "h_med": float(np.median(ops.h)), "g0_grad": g0["grad"], "g0_lap": g0["lap"]}
        pt = p_true[m]
        for tag, (u, nut) in {"true": (f0["u_true"], f0["nut_true"]),
                              **{rr: (flds[rr]["u_pred"], flds[rr]["nut_pred"]) for rr in runs}}.items():
            src = pp.sources(ops, u[m], nut[m], f0["rho"])
            p = pp.solve_momentum_ls(ops, src["f_A"], far)
            r[f"MA_{tag}"] = pp.rel_l2(p, pt)
            r[f"MA_{tag}_offset_removed"] = pp.rel_l2(p - (p - pt).mean(), pt)
            r[f"div_rel_{tag}"] = float(np.linalg.norm(src["div"]) / np.linalg.norm(src["J"].reshape(n, -1)))
            # magnitude check: |f| (momentum RHS) against |grad_h p_true| on the subset
            gp = ops.grad(pt)
            r[f"f_over_gradp_{tag}"] = float(np.linalg.norm(src["f_A"]) / np.linalg.norm(gp))
        for rr in runs:
            r[f"direct_{rr}"] = pp.rel_l2(flds[rr]["p_pred"][m] - flds[rr]["p_inf"], pt)
        out[f"sdf>={s:g}L"] = r
    return case, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True)
    ap.add_argument("--runs", nargs="+", default=["v0_isla_qtsdfval_seed42", "v0_gt_vol_seed42"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cases", type=int, default=0)
    args = ap.parse_args()
    evals = Path(args.evals)
    run_dirs = {r: evals / r / r / "predictions" for r in args.runs}
    cases = sorted(p.name for p in run_dirs[args.runs[0]].iterdir() if p.name.endswith(".pdmsh"))
    if args.cases:
        cases = cases[: args.cases]
    res = {"args": vars(args), "cases": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    with Pool(args.workers) as pool:
        for i, (case, r) in enumerate(pool.imap_unordered(one_case, [(c, run_dirs, args.runs) for c in cases])):
            res["cases"][case] = r
            print(f"[{i+1}/{len(cases)}] {case} " + " ".join(f"{k}: MA_true={v['MA_true']:.3f} div={v['div_rel_true']:.3f}" for k, v in r.items()), flush=True)
            json.dump(res, open(args.out, "w"), indent=1)
    C = list(res["cases"].values())
    agg = {}
    for key in C[0]:
        agg[key] = {}
        for m in C[0][key]:
            vals = [c[key][m] for c in C if key in c]
            agg[key][m] = {"mean": float(np.mean(vals)), "median": float(np.median(vals)), "max": float(np.max(vals))}
    res["aggregate"] = agg
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(res, open(args.out, "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
