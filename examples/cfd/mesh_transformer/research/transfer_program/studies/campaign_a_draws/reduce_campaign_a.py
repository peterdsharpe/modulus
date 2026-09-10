"""Reduce Campaign A (data-draw variance of the HiLiftAeroML comparisons).

Reads hl_evals/<run_id>/*/metrics.jsonl for the 72 campaign runs (and the curated-draw
runs of the same arms, as the comparison "draw 0"), and writes
$T/transfer/campaign_a_reduction.json with, per arm and rung:
  * the 9 per-run means (3 draws x 3 seeds) of pressure / velocity / wall-shear rel-L2,
  * per-draw seed means, the between-draw and within-draw standard deviations
    (one-way layout) and their ratio,
  * per-draw ratios GT/ISLA, Transolver/ISLA, ISLA-noweights/ISLA from seed-mean errors,
    with the mean over draws, the t-based 90% interval (n = 3) and the spread,
  * paired per-case counts per draw (descriptive), and the curated draw's values
    with a flag for whether they fall inside the three-draw range.
Bars (PREREG.md): ordering stands if the 90% interval of baseline/ISLA excludes 1.0;
parity if it contains 1.0 with spread < 0.10; weights-off persists if nw/ISLA <= 0.93 on
all draws, fades if any draw >= 0.97. Runs on the cluster login node with the recipe venv.
"""
import glob, json, math, os, statistics as st, sys
T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
# float32 inference is the reporting instrument (CLAIMS 31a069280): campaign evals live in hl_evals_fp32/, the
# curated-draw references in the main session's float32 re-evaluations in the same directory. EVAL_ROOT=hl_evals
# reduces the bf16 side for the offset table.
EVAL_ROOT = os.environ.get("EVAL_ROOT", "hl_evals_fp32")
OUT = os.environ.get("OUT", f"{T}/transfer/campaign_a_reduction_{EVAL_ROOT}.json")
F = ("pressure_l2", "velocity_l2", "tau_wall_l2")
ARMS = ("gt", "transolver", "isla", "islanw")
CURATED = {  # curated-draw run ids for the comparison "draw 0" (unit drive for the baselines)
    ("gt", 35): ["udrv_hl_gt_super_scarce_seed42", "udrv_hl_gt_super_scarce_seed43", "udrv_hl_gt_super_scarce_seed44"],
    ("transolver", 35): ["udrv_hl_transolver_super_scarce_lr3e3_seed42", "udrv_hl_transolver_super_scarce_lr3e3_seed43", "udrv_hl_transolver_super_scarce_lr3e3_seed44"],
    ("isla", 35): ["lad_hl_mt2_super_scarce_seed42", "lad_hl_mt2_super_scarce_seed43", "lad_hl_mt2_super_scarce_seed44"],
    ("islanw", 35): ["a35_hl_mt2_noweights_seed42", "a35_hl_mt2_noweights_seed43", "now_hl_mt2_super_scarce_seed44"],
    ("gt", 210): ["udrv_hl_gt_scarce_seed42", "udrv_hl_gt_scarce_seed43", "udrv_hl_gt_scarce_seed44"],
    ("transolver", 210): ["udrv_hl_transolver_scarce_lr3e3_seed42", "udrv_hl_transolver_scarce_lr3e3_seed43", "udrv_hl_transolver_scarce_lr3e3_seed44"],
    ("isla", 210): ["lad_hl_mt2_scarce_seed42", "lad_hl_mt2_scarce_seed43", "lad_hl_mt2_scarce_seed44"],
    ("islanw", 210): ["a35_hl_mt2_noweights_scarce_seed42", "a35_hl_mt2_noweights_scarce_seed43"],
}
T90 = {2: 2.920, 1: 6.314}  # two-sided 90% t quantile for n-1 degrees of freedom


def load(run):
    ps = glob.glob(f"{T}/{EVAL_ROOT}/{run}/*/metrics.jsonl")
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0])]
    return {r["sample_id"]: r["metrics"] for r in rows if r.get("phase") == "infer_step"}


def seed_mean_percase(dicts):
    ks = sorted(set.intersection(*[set(d) for d in dicts]))
    return {k: st.mean(d[k]["pressure_l2"] for d in dicts) for k in ks}


out = {"arms": {}, "draws": {}, "ratios": {}, "curated": {}, "missing": []}
percase = {}  # (arm, n, draw) -> per-case seed-mean pressure
for n in (35, 210):
    for arm in ARMS:
        per_draw = {}
        for d in (1, 2, 3):
            runs = [f"campA_hl_{arm}_n{n}_d{d}_seed{s}" for s in (42, 43, 44)]
            ds = [load(r) for r in runs]
            got = [(r, x) for r, x in zip(runs, ds) if x is not None]
            out["missing"] += [r for r, x in zip(runs, ds) if x is None]
            if not got:
                continue
            means = {f: [st.mean(v[f] for v in x.values()) for _, x in got] for f in F}
            per_draw[d] = {"runs": [r for r, _ in got], "means": means, "seed_mean": {f: st.mean(means[f]) for f in F}}
            percase[(arm, n, d)] = seed_mean_percase([x for _, x in got])
        # curated draw 0
        cds = [load(r) for r in CURATED[(arm, n)]]
        cgot = [(r, x) for r, x in zip(CURATED[(arm, n)], cds) if x is not None]
        if cgot:
            cm = {f: [st.mean(v[f] for v in x.values()) for _, x in cgot] for f in F}
            per_draw[0] = {"runs": [r for r, _ in cgot], "means": cm, "seed_mean": {f: st.mean(cm[f]) for f in F}, "curated": True}
            percase[(arm, n, 0)] = seed_mean_percase([x for _, x in cgot])
        # variance decomposition over the three campaign draws (pressure)
        dm = [per_draw[d]["seed_mean"]["pressure_l2"] for d in (1, 2, 3) if d in per_draw]
        within = [st.pstdev(per_draw[d]["means"]["pressure_l2"]) for d in (1, 2, 3) if d in per_draw and len(per_draw[d]["means"]["pressure_l2"]) > 1]
        dec = {}
        if len(dm) >= 2:
            dec = {"between_draw_sd": st.stdev(dm), "within_draw_sd_pooled": math.sqrt(st.mean([w ** 2 for w in within])) if within else None,
                   "draw_means": dm}
            if dec["within_draw_sd_pooled"]:
                dec["ratio_between_over_within"] = dec["between_draw_sd"] / dec["within_draw_sd_pooled"]
            if 0 in per_draw:
                c = per_draw[0]["seed_mean"]["pressure_l2"]; dec["curated_mean"] = c
                dec["curated_typical"] = min(dm) <= c <= max(dm)
        out["arms"][f"{arm}_n{n}"] = {"per_draw": per_draw, "pressure_decomposition": dec}
    # ratios per draw
    for a, b, name in (("gt", "isla", "gt_over_isla"), ("transolver", "isla", "transolver_over_isla"), ("islanw", "isla", "noweights_over_isla")):
        vals = {}
        for d in (0, 1, 2, 3):
            ka, kb = f"{a}_n{n}", f"{b}_n{n}"
            if d in out["arms"].get(ka, {}).get("per_draw", {}) and d in out["arms"].get(kb, {}).get("per_draw", {}):
                ea = out["arms"][ka]["per_draw"][d]["seed_mean"]["pressure_l2"]; eb = out["arms"][kb]["per_draw"][d]["seed_mean"]["pressure_l2"]
                pa, pb = percase[(a, n, d)], percase[(b, n, d)]; ks = sorted(set(pa) & set(pb))
                vals[d] = {"ratio": ea / eb, "cases_where_second_lower": sum(pb[k] < pa[k] for k in ks), "n_cases": len(ks),
                           "median_case_ratio": st.median(pa[k] / pb[k] for k in ks) if ks else None}
        camp = [vals[d]["ratio"] for d in (1, 2, 3) if d in vals]
        summ = {"per_draw": vals}
        if len(camp) >= 2:
            m, s = st.mean(camp), st.stdev(camp); t = T90[len(camp) - 1]; h = t * s / math.sqrt(len(camp))
            summ.update({"mean_over_draws": m, "ci90": [m - h, m + h], "spread": max(camp) - min(camp), "n_draws": len(camp)})
            if name != "noweights_over_isla":
                summ["verdict"] = ("ISLA behind" if m + h < 1.0 else "ISLA ahead" if m - h > 1.0 else
                                   "parity" if (max(camp) - min(camp)) < 0.10 else "draw-dependent")
            else:
                summ["verdict"] = ("persists" if all(r <= 0.93 for r in camp) and len(camp) == 3 else
                                   "fades" if any(r >= 0.97 for r in camp) else "between")
        out["ratios"][f"{name}_n{n}"] = summ
out["eval_root"] = EVAL_ROOT
json.dump(out, open(OUT, "w"), indent=1)
for k, v in out["arms"].items():
    print(k, {d: round(x["seed_mean"]["pressure_l2"], 4) for d, x in v["per_draw"].items()}, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v["pressure_decomposition"].items() if kk != "draw_means"})
for k, v in out["ratios"].items():
    print(k, {d: round(x["ratio"], 3) for d, x in v["per_draw"].items()}, {kk: vv for kk, vv in v.items() if kk in ("mean_over_draws", "ci90", "spread", "verdict")})
print("missing:", len(out["missing"]))
