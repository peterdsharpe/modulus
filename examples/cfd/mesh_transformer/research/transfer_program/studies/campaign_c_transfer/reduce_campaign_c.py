"""Reduce campaign C (DrivAerML -> SHIFT-SUV fastback transfer) from the float32 evaluations and grade against PREREG.md.

Reads ``$T/campc_evals_fp32/<run>/{fastback,drivaer}/<run>/metrics.jsonl``:
per-case ``infer_step`` records (pressure_l2, wss_l2) and the recipe's
``infer_forces_summary`` (CD/CL/CS predicted vs reference means and MAE over
the cases; the force error is reported as MAE / |reference mean| per
coefficient, since per-case force records are not written). Seed means per
arm; bars copied from PREREG.md. Writes campaign_c_<date>.json next to this
file (or argv[1]).
"""
import glob
import json
import statistics
import sys
from datetime import date
from pathlib import Path

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
ARMS = {
    # T1 complete conditioning (mixed DrivAerML + estate; fastback carries cond=1 for the conditioned arms)
    "cond0_isla": ["campC_cond0_isla_seed42", "campC_cond0_isla_seed43"],
    "cond1_isla": ["campC_cond1_isla_seed42", "campC_cond1_isla_seed43"],
    "cond0_gt": ["campC_cond0_gt_seed42", "campC_cond0_gt_seed43"],
    "cond1_gt": ["campC_cond1_gt_seed42", "campC_cond1_gt_seed43"],
    # T2 matched-count diversity (218 DrivAerML + 217 estate)
    "mix435_isla": ["campC_mix435_isla_seed42", "campC_mix435_isla_seed43"],
    "mix435_gt": ["campC_mix435_gt_seed42", "campC_mix435_gt_seed43"],
    # T3 few-shot
    "ft20_isla": ["campC_ft20_isla_seed42", "campC_ft20_isla_seed43"],
    "scratch20_isla": ["campC_scratch20_isla_seed42", "campC_scratch20_isla_seed43"],
    "ft20_gt": ["campC_ft20_gt_seed42", "campC_ft20_gt_seed43"],
    "scratch20_gt": ["campC_scratch20_gt_seed42", "campC_scratch20_gt_seed43"],
    "ft20lr3_isla": ["campC_ft20lr3_isla_seed42", "campC_ft20lr3_isla_seed43"],
    "ft20lr3_gt": ["campC_ft20lr3_gt_seed42", "campC_ft20lr3_gt_seed43"],
    # T2 all-DrivAerML single-family references, scored zero-shot on the fastback cases and on DrivAerML in float32
    # by the same eval launcher (lane-table rows 24-27); the bf16 book value for ISLA was 0.99 (three seeds).
    "ref_single_isla": ["iw_mt2_lr1e3_seed42", "iw_mt2_lr1e3_seed43"],
    "ref_single_gt": ["uw_gt_unit_lr1e3_seed42", "uw_gt_unit_lr1e3_seed43"],
    # T2 decomposition: 218 DrivAerML cars alone at the campaign protocol (rows 29-32)
    "dr218_isla": ["campC_dr218_isla_seed42", "campC_dr218_isla_seed43"],
    "dr218_gt": ["campC_dr218_gt_seed42", "campC_dr218_gt_seed43"],
}


def load(run, probe):
    ps = glob.glob(f"{T}/campc_evals_fp32/{run}/{probe}/**/metrics.jsonl", recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0])]
    steps = [r["metrics"] for r in rows if r.get("phase") == "infer_step"]
    forces = next((r["coefficients"] for r in rows if r.get("phase") == "infer_forces_summary"), None)
    out = {"n": len(steps), "pressure_l2": statistics.mean(s["pressure_l2"] for s in steps), "wss_l2": statistics.mean(s["wss_l2"] for s in steps)}
    if forces:
        for c in ("CD", "CL", "CS"):
            if c in forces and forces[c].get("true_mean"):
                out[f"{c}_rel_mae"] = forces[c]["mae"] / abs(forces[c]["true_mean"])
                out[f"{c}_true_mean"] = forces[c]["true_mean"]; out[f"{c}_pred_mean"] = forces[c]["pred_mean"]
    return out


res = {"date": date.today().isoformat(), "runs": {}, "arms": {}}
for arm, runs in ARMS.items():
    per = {}
    for run in runs:
        fb, dr = load(run, "fastback"), load(run, "drivaer")
        if fb:
            per[run] = {"fastback": fb, "drivaer": dr}
            res["runs"][run] = per[run]
    if per:
        agg = {"n_seeds": len(per)}
        for probe in ("fastback", "drivaer"):
            vals = [v[probe] for v in per.values() if v.get(probe)]
            if vals:
                agg[probe] = {k: statistics.mean(v[k] for v in vals) for k in vals[0] if isinstance(vals[0][k], (int, float)) and k != "n"}
                agg[probe]["seeds_pressure"] = [v["pressure_l2"] for v in vals]
        res["arms"][arm] = agg

A = res["arms"]
def fb(arm, k="pressure_l2"):
    return A.get(arm, {}).get("fastback", {}).get(k)
def dr(arm, k="pressure_l2"):
    return A.get(arm, {}).get("drivaer", {}).get(k)

verdict = {}
# T1: conditioning matters if conditioned fastback error is >= 15% below the unconditioned twin (either architecture); null if within 5%.
for a in ("isla", "gt"):
    c0, c1 = fb(f"cond0_{a}"), fb(f"cond1_{a}")
    if c0 and c1 and A[f"cond0_{a}"]["n_seeds"] == 2 and A[f"cond1_{a}"]["n_seeds"] == 2:
        gain = 1 - c1 / c0
        v = "MATTERS" if gain >= 0.15 else "NULL" if abs(gain) <= 0.05 else "BETWEEN"
        d0, d1 = dr(f"cond0_{a}"), dr(f"cond1_{a}")
        verdict[f"T1_{a}"] = {"fastback_cond0": c0, "fastback_cond1": c1, "gain": gain, "verdict": v,
                              "drivaer_cond0": d0, "drivaer_cond1": d1, "infamily_cost": (d1 / d0 - 1) if d0 and d1 else None}
# T2: diversity helps if fastback error falls >= 30% vs the all-DrivAerML reference at <= 10% in-family cost.
for a in ("isla", "gt"):
    m, ref, dref = fb(f"mix435_{a}"), fb(f"ref_single_{a}"), dr(f"ref_single_{a}")
    if m and ref:
        cost = (dr(f"mix435_{a}") / dref - 1) if (dr(f"mix435_{a}") and dref) else None
        drop = 1 - m / ref
        v = ("HELPS at <= 10% in-family cost" if (drop >= 0.30 and cost is not None and cost <= 0.10)
             else "HELPS on the target, at an in-family cost above 10% (trade reported)" if drop >= 0.30 else "REPORTED")
        verdict[f"T2_{a}"] = {"fastback_mix435": m, "fastback_reference_zeroshot": ref, "drop": drop,
                              "drivaer_mix435": dr(f"mix435_{a}"), "drivaer_reference": dref, "infamily_cost": cost, "verdict": v}
    elif m:
        verdict[f"T2_{a}"] = {"fastback_mix435": m, "fastback_reference_zeroshot": None, "note": "reference pending"}
    # decomposition of the in-family cost: rung (218 DrivAerML alone / 435 DrivAerML) x mixing (218 + 217 estate / 218 alone)
    d218 = dr(f"dr218_{a}")
    if d218 and dref and dr(f"mix435_{a}") and f"T2_{a}" in verdict:
        verdict[f"T2_{a}"]["infamily_decomposition"] = {"drivaer_218_alone": d218, "rung_factor": d218 / dref,
                                                        "mixing_factor": dr(f"mix435_{a}") / d218, "fastback_218_alone": fb(f"dr218_{a}")}
# T3: credible if fine-tune <= 0.15 and >= 2x better than scratch; pretraining adds nothing if scratch <= 0.15.
for a in ("isla", "gt"):
    s = fb(f"scratch20_{a}")
    for tag in ("ft20", "ft20lr3"):
        f = fb(f"{tag}_{a}")
        if f and s:
            ratio = s / f
            v = "CREDIBLE" if (f <= 0.15 and ratio >= 2) else "PRETRAINING ADDS NOTHING (scratch <= 0.15)" if s <= 0.15 else "NOT CREDIBLE AT THIS BUDGET"
            if tag == "ft20lr3":
                rel = f / s - 1
                v += "; " + ("within 5% of scratch: pretraining worth nothing at this budget" if abs(rel) <= 0.05 else "≥20% better than scratch: pretraining helps, labels too few" if rel <= -0.20 else "reported")
            verdict[f"T3_{tag}_{a}"] = {"fastback_finetune": f, "fastback_scratch": s, "scratch_over_finetune": ratio, "verdict": v}
res["verdict"] = verdict

print("| arm | seeds | fastback pressure | fastback wall shear | CD rel. MAE | CL rel. MAE | DrivAerML pressure |")
print("|---|---|---|---|---|---|---|")
for arm, a in A.items():
    f = a.get("fastback", {}); d = a.get("drivaer", {})
    print(f"| {arm} | {a['n_seeds']} | {f.get('pressure_l2', float('nan')):.4f} | {f.get('wss_l2', float('nan')):.4f} | {f.get('CD_rel_mae', float('nan')):.3f} | {f.get('CL_rel_mae', float('nan')):.3f} | {d.get('pressure_l2', float('nan')) if d else float('nan'):.4f} |")
print(json.dumps(verdict, indent=1))
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name(f"campaign_c_{res['date']}.json")
dst.write_text(json.dumps(res, indent=1)); print("wrote", dst)
