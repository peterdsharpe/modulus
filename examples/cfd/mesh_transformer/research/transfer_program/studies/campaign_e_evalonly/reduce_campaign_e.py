"""Reduce campaign E (density-bias probe) and grade against PREREG.md.

Reads ``$T/transfer/campaign_e/<run>/{unif,biased}/**/metrics.jsonl`` (48
DrivAerML validation cars each), computes per-run mean pressure and
wall-shear relative L2 on the uniform control and the 10:1 biased draw, the
per-run degradation factor biased / uniform, per-arm seed means, and the
preregistered verdict. Writes ``campaign_e_density_<date>.json`` next to
this file (or to the path given as argv[1]).
"""
import glob
import json
import statistics
import sys
from datetime import date
from pathlib import Path

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
ARMS = {
    "gt_unit": ["uw_gt_unit_lr1e3_seed42", "uw_gt_unit_lr1e3_seed43", "uw_gt_unit_lr1e3_seed44"],
    "transolver_unit": ["uw_transolver_unit_lr3e3_seed42", "uw_transolver_unit_lr3e3_seed43"],
    "isla_reference": ["iw_mt2_lr1e3_seed42", "iw_mt2_lr1e3_seed43"],
    "isla_gauge": ["iw_mt2_gauge_seed42", "iw_mt2_gauge_seed43"],
    "isla_noweights": ["now_mt2_lr1e3_seed42", "now_mt2_lr1e3_seed43"],
}
KEYS = ("pressure_l2", "wall_shear_l2", "wallshear_l2", "wall_shear_stress_l2")
BARS = {"holds_gt_factor_min": 1.32, "isla_factor_max": 1.24, "not_hold_gt_factor_max": 1.24, "isla_gauge_factor_prior": 1.20}


def run_metrics(run, probe):
    ps = glob.glob(f"{T}/transfer/campaign_e/{run}/{probe}/**/metrics.jsonl", recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0])]
    rows = [r for r in rows if r.get("phase") == "infer_step"]
    keys = [k for k in rows[0]["metrics"] if k.endswith("_l2")]
    per_case = {r["sample_id"]: r["metrics"] for r in rows}
    return {"n": len(rows), "mean": {k: statistics.mean(r["metrics"][k] for r in rows) for k in keys}, "per_case": per_case}


out = {"date": date.today().isoformat(), "bars": BARS, "runs": {}, "arms": {}}
for arm, runs in ARMS.items():
    factors, unif, biased = [], [], []
    for run in runs:
        u, b = run_metrics(run, "unif"), run_metrics(run, "biased")
        if not (u and b):
            out["runs"][run] = None
            continue
        f = b["mean"]["pressure_l2"] / u["mean"]["pressure_l2"]
        # paired per-car factor summary (same 48 cars, different samplings)
        common = sorted(set(u["per_case"]) & set(b["per_case"]))
        per_car = [b["per_case"][c]["pressure_l2"] / u["per_case"][c]["pressure_l2"] for c in common]
        out["runs"][run] = {"n_unif": u["n"], "n_biased": b["n"], "unif": u["mean"], "biased": b["mean"], "factor_pressure": f,
                            "per_car_factor_median": statistics.median(per_car), "cars_worse_under_bias": sum(x > 1 for x in per_car), "n_common": len(common)}
        factors.append(f); unif.append(u["mean"]["pressure_l2"]); biased.append(b["mean"]["pressure_l2"])
    if factors:
        out["arms"][arm] = {"n_seeds": len(factors), "factor_mean": statistics.mean(factors), "factor_seeds": factors,
                            "unif_pressure_mean": statistics.mean(unif), "biased_pressure_mean": statistics.mean(biased)}

# Preregistered bars name "ISLA" as the reference configuration; the gauge arm is graded alongside
# because the book's 1.20 figure (and the derivation of the bars) belongs to it. Both are reported.
g, i = out["arms"].get("gt_unit"), out["arms"].get("isla_reference")
if g and out["arms"].get("isla_gauge"):
    gg = out["arms"]["isla_gauge"]
    out["verdict_gauge_arm"] = {"gt_factor": g["factor_mean"], "isla_gauge_factor": gg["factor_mean"],
                                "holds_by_bars_with_gauge_as_isla": g["factor_mean"] >= BARS["holds_gt_factor_min"] and gg["factor_mean"] <= BARS["isla_factor_max"]}
if g and i:
    if g["factor_mean"] >= BARS["holds_gt_factor_min"] and i["factor_mean"] <= BARS["isla_factor_max"]:
        v = "HOLDS: ISLA's density robustness survives against the input-matched GeoTransolver"
    elif g["factor_mean"] <= BARS["not_hold_gt_factor_max"]:
        v = "DOES NOT HOLD: unit-drive GeoTransolver is as density-robust as ISLA within the noise"
    else:
        v = "BETWEEN: reported as ratio of factors"
    out["verdict"] = {"text": v, "gt_factor": g["factor_mean"], "isla_factor": i["factor_mean"], "ratio_gt_over_isla": g["factor_mean"] / i["factor_mean"]}

print("| arm | seeds | uniform pressure | biased pressure | degradation factor (seeds) |")
print("|---|---|---|---|---|")
for arm, a in out["arms"].items():
    print(f"| {arm} | {a['n_seeds']} | {a['unif_pressure_mean']:.4f} | {a['biased_pressure_mean']:.4f} | {a['factor_mean']:.3f} ({', '.join(f'{x:.3f}' for x in a['factor_seeds'])}) |")
for run, r in out["runs"].items():
    if r:
        print(f"  {run}: unif {r['unif']['pressure_l2']:.4f} biased {r['biased']['pressure_l2']:.4f} factor {r['factor_pressure']:.3f} median per-car {r['per_car_factor_median']:.3f} worse {r['cars_worse_under_bias']}/{r['n_common']}")
print(json.dumps(out.get("verdict"), indent=1))
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name(f"campaign_e_density_{out['date']}.json")
slim = {**out, "runs": {k: ({kk: vv for kk, vv in v.items()} if v else None) for k, v in out["runs"].items()}}
dst.write_text(json.dumps(slim, indent=1)); print("wrote", dst)
