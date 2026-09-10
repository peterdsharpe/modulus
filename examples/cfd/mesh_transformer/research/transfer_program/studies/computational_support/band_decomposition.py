"""Wall-distance band decomposition of the interior errors from saved float32 predictions.

For each run (ISLA hidden 344, ISLA hidden 256 reference, unit-drive GeoTransolver-volume, GeoTransolver-volume without
local features) and each of the 48 DrivAerML validation cars, read the saved interior predictions
(`<root>/<run>/<run>/predictions/<case>/_tensordict/interior/_tensordict/point_data/{pred,true}_{pressure,nut,velocity}.memmap`
and `sdf.memmap`) and report the relative L2 of eddy viscosity, pressure and velocity on wall-distance bands
(SDF < 0.01 L, 0.01-0.05 L, >= 0.05 L, with L = the case's reference length from global_data/L_ref), plus the share of
each field's squared error carried by each band. Answers: is GeoTransolver-volume's remaining eddy-viscosity lead a
near-wall effect? Runs on the login node with the recipe venv (numpy only). Writes band_decomposition_<date>.json.
"""
import json
import os
import statistics
import sys
from datetime import date

import numpy as np

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
RUNS = {
    "isla_h256": ("v0_evals_support_fp32", ["v0_isla_qtsdfval_seed42", "v0_isla_qtsdfval_seed43"]),
    "isla_h344": ("v0_evals_fp32", ["v0_isla_qtsdf_h344_seed42", "v0_isla_qtsdf_h344_seed43"]),
    "gt_unit": ("v0_evals_support_fp32", ["udrv_gt_vol_seed42", "udrv_gt_vol_seed43"]),
    "gt_nolocal": ("v0_evals_fp32", ["v0_gt_vol_nolocal_seed42", "v0_gt_vol_nolocal_seed43"]),
    # far-field context arms (PREREG_WAKE.md): wake tokens on the drive axis; latent volume tokens with query tokens
    "isla_h344_wake": ("v0_evals_fp32", ["v0_isla_qtsdf_h344_wake_seed42", "v0_isla_qtsdf_h344_wake_seed43"]),
    "isla_h344_lvt": ("v0_evals_fp32", ["v0_isla_qtsdf_h344_lvt_seed42", "v0_isla_qtsdf_h344_lvt_seed43"]),
}
BANDS = [("sdf<0.01L", 0.0, 0.01), ("0.01-0.05L", 0.01, 0.05), ("sdf>=0.05L", 0.05, np.inf)]
FIELDS = ("nut", "pressure", "velocity")


def mm(d, name):
    meta = json.load(open(os.path.join(d, "meta.json")))[name]
    return np.memmap(os.path.join(d, f"{name}.memmap"), dtype=np.dtype(meta["dtype"].replace("torch.", "")), mode="r", shape=tuple(meta["shape"]))


def case_bands(pred_dir, case):
    td = os.path.join(pred_dir, case, "_tensordict", "interior", "_tensordict")
    pd_ = os.path.join(td, "point_data")
    sdf = np.asarray(mm(pd_, "sdf")).ravel().astype(np.float64)
    gd = os.path.join(td, "global_data")
    try:
        L = float(np.asarray(mm(gd, "L_ref")).ravel()[0])
    except Exception:
        L = 1.0
    out = {}
    for f in FIELDS:
        p = np.asarray(mm(pd_, f"pred_{f}")).astype(np.float64).reshape(len(sdf), -1)
        t = np.asarray(mm(pd_, f"true_{f}")).astype(np.float64).reshape(len(sdf), -1)
        err2 = ((p - t) ** 2).sum(-1); tru2 = (t ** 2).sum(-1)
        res = {"all": {"rel_l2": float(np.sqrt(err2.sum() / tru2.sum())), "n": int(len(sdf))}}
        for name, lo, hi in BANDS:
            m = (sdf >= lo * L) & (sdf < hi * L)
            if m.sum() == 0:
                continue
            res[name] = {"rel_l2": float(np.sqrt(err2[m].sum() / max(tru2[m].sum(), 1e-30))), "n": int(m.sum()),
                         "share_of_sq_err": float(err2[m].sum() / err2.sum()), "share_of_points": float(m.mean())}
        out[f] = res
    return out


def main():
    result = {"date": date.today().isoformat(), "bands": [b[0] for b in BANDS], "arms": {}}
    for arm, (root, runs) in RUNS.items():
        per_run = {}
        for run in runs:
            pred_dir = os.path.join(T, root, run, run, "predictions")
            if not os.path.isdir(pred_dir):
                print("missing", pred_dir); continue
            cases = sorted(os.listdir(pred_dir))
            acc = {}
            for c in cases:
                cb = case_bands(pred_dir, c)
                for f, res in cb.items():
                    for band, v in res.items():
                        acc.setdefault(f, {}).setdefault(band, []).append(v)
            per_run[run] = {f: {b: {k: statistics.mean(x[k] for x in vs) for k in vs[0] if k != "n"} | {"n_mean": statistics.mean(x["n"] for x in vs)}
                                for b, vs in bands.items()} for f, bands in acc.items()}
            print(arm, run, {f: {b: round(v["rel_l2"], 4) for b, v in per_run[run][f].items()} for f in FIELDS})
        if per_run:
            result["arms"][arm] = {"runs": per_run,
                                   "mean": {f: {b: {k: statistics.mean(r[f][b][k] for r in per_run.values()) for k in next(iter(per_run.values()))[f][b]}
                                                for b in next(iter(per_run.values()))[f]} for f in FIELDS}}
    dst = sys.argv[1] if len(sys.argv) > 1 else f"{T}/transfer/band_decomposition_{result['date']}.json"
    json.dump(result, open(dst, "w"), indent=1)
    print("== eddy viscosity rel L2 by band (two-seed means) ==")
    for arm, a in result["arms"].items():
        print(arm, {b: round(v["rel_l2"], 4) for b, v in a["mean"]["nut"].items()}, "share of sq err near wall:", round(a["mean"]["nut"].get("sdf<0.01L", {}).get("share_of_sq_err", float("nan")), 3))
    print("wrote", dst)


if __name__ == "__main__":
    main()
