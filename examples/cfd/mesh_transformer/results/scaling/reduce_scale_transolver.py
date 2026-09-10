"""Reduce the Transolver SCALE lanes (notebook #sec-nb-scale-prereg, #sec-nb-scale-transolver-verdict).

Runs on the cluster login node with the recipe venv python (reads metrics.jsonl, the saved
prediction points and train.log only; no compute). Reporting instrument (program-wide rule,
2026-09-09): float32 inference is the headline (hl_evals_fp32/<run>/), the bf16 evaluation
(hl_evals/<run>/) is reported alongside with the per-run shift (mean over cases of the bf16
pressure error minus the fp32 one, and the per-case maximum absolute difference); the two
artifacts' sampled points are asserted bit-identical per case. For every run: validation means
of pressure / velocity / wall-shear relative L2 under both instruments, parameter count, peak
"Mem:" from train.log, median step time (each link's first step excluded), and GPU-hours for
500 epochs (steps x median step time x 4 GPUs); per arm the two-seed means.
References (not rerun): HiLift udrv_hl_transolver_full_lr3e3_seed{42,43}; DrivAerML
uw_transolver_unit_lr3e3_seed{42,43} and uw_transolver_unit_lr1e3_seed{42,43}; their fp32
re-evaluations are read from hl_evals_fp32/ and iw_evals_fp32/ when present, otherwise the
comparison is labelled fp32-vs-bf16.
Writes $T/hl_evals/transolver_scale_reduction.json; copy to
results/scaling/transolver_scale_reduction_2026-09-09.json.
"""
import glob, json, os, re, statistics as st

import numpy as np

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
F = ("pressure_l2", "velocity_l2", "tau_wall_l2")
ARMS = {}
for ds in ("hl", "dr"):
    for arm in ("w384", "w512", "t40k", "t80k", "c384x40k"):
        ARMS[f"{ds}_{arm}"] = [f"scale_transolver_{ds}_{arm}_seed{s}" for s in (42, 43)]
# Coordinator amendment 2026-09-10: the ISLA 512 x 80k pilot (0.0428 fp32) sits below the Transolver 384 x 40k corner,
# so the frontier needs Transolver at the same corner (DrivAerML only; rows 20-21 of the lane table).
ARMS["dr_c512x80k"] = [f"scale_transolver_dr_c512x80k_seed{s}" for s in (42, 43)]
REFS = {"hl_ref_unit_lr3e3": ("hl_evals", ["udrv_hl_transolver_full_lr3e3_seed42", "udrv_hl_transolver_full_lr3e3_seed43"]),
        "dr_ref_unit_lr3e3": ("iw_evals", ["uw_transolver_unit_lr3e3_seed42", "uw_transolver_unit_lr3e3_seed43"]),
        "dr_ref_unit_lr1e3": ("iw_evals", ["uw_transolver_unit_lr1e3_seed42", "uw_transolver_unit_lr1e3_seed43"])}


def rows(root, run):
    ps = glob.glob(f"{T}/{root}/{run}/*/metrics.jsonl") + glob.glob(f"{T}/{root}/{run}/metrics.jsonl")
    if not ps:
        return None
    rs = [json.loads(l) for l in open(ps[0])]
    return {r["sample_id"]: r["metrics"] for r in rs if r.get("phase") == "infer_step"}


def means(d):
    return {f: st.mean(v[f] for v in d.values()) for f in F if f in next(iter(d.values()))} | {"n_cases": len(d)}


def points_identical(root_a, root_b, run):
    """Every case's saved sampled points must be bit-identical between the two evaluations."""
    pa = sorted(glob.glob(f"{T}/{root_a}/{run}/*/predictions/*/_tensordict/interior/_tensordict/points.memmap"))
    if not pa:
        return None
    n_same = 0
    for p in pa:
        q = p.replace(f"/{root_a}/", f"/{root_b}/", 1)
        if not os.path.exists(q):
            return {"checked": len(pa), "identical": n_same, "missing_in_fp32": True}
        if open(p, "rb").read() == open(q, "rb").read():
            n_same += 1
    return {"checked": len(pa), "identical": n_same}


LOADER_HAZARD = ("skipping load", "Could not find valid model file", "Inference summary is inconsistent")  # last: infer.py summary range guard (f9330f317)
HAZARD = {"logs_checked": 0, "hits": []}


def log_ok(path):
    """Interim loader-hazard rule (coordinator, 2026-09-10): refuse a run whose evaluation or probe log shows a
    skipped or missing checkpoint load. Counts every log it reads so the verdict can report the tally."""
    if not os.path.exists(path):
        return True
    HAZARD["logs_checked"] += 1
    txt = open(path, errors="replace").read()
    if any(h in txt for h in LOADER_HAZARD):
        HAZARD["hits"].append(path.replace(T + "/", ""))
        return False
    return True


def cost(run):
    try:
        log = open(f"{T}/runs/{run}/train.log").read()
    except FileNotFoundError:
        return None
    params = re.findall(r"Parameters: ([\d,]+)", log)
    steps = re.findall(r"Epoch \d+ \[(\d+)/(\d+)\] Loss: [\d.e+-]+ Step: ([\d.]+)s Mem: ([\d.]+)GB", log)
    out = {"parameters": int(params[-1].replace(",", "")) if params else None, "trained": "Training completed" in log}
    if not steps:
        return out
    times = [float(s[2]) for s in steps if int(s[0]) > 1]
    med = st.median(times) if times else None
    per_epoch = int(steps[-1][1])
    out.update({"peak_mem_gb": max(float(s[3]) for s in steps), "median_step_s": med, "steps_logged": len(steps),
                "steps_per_epoch": per_epoch, "epochs_started": len(re.findall(r"--- Epoch \d+/\d+ ---", log)),
                "gpu_hours_500ep": (500 * per_epoch * med * 4 / 3600) if med else None})
    return out


def instrument_pair(root, run):
    """Metrics under bf16 (root) and fp32 (root_fp32) plus the shift; fp32 is the headline when present."""
    b = rows(root, run) if log_ok(f"{T}/{root}/{run}.log") else None
    f32 = rows(root + "_fp32", run) if log_ok(f"{T}/{root}_fp32/{run}.log") else None
    rec = {"bf16": means(b) if b else None, "fp32": means(f32) if f32 else None}
    if b and f32:
        ks = sorted(set(b) & set(f32))
        diff = [b[k]["pressure_l2"] - f32[k]["pressure_l2"] for k in ks]
        rec["shift_bf16_minus_fp32_pressure"] = {"mean": st.mean(diff), "max_abs_per_case": max(abs(x) for x in diff), "n": len(ks)}
        rec["points_identical"] = points_identical(root, root + "_fp32", run)
    rec["headline"] = rec["fp32"] or rec["bf16"]
    rec["headline_instrument"] = "fp32" if f32 else ("bf16" if b else None)
    return rec


out = {"instrument_rule": "fp32 headline, bf16 alongside with shift; points asserted bit-identical", "runs": {}, "arms": {}, "refs": {}}
for arm, runs in ARMS.items():
    recs = []
    for r in runs:
        rec = instrument_pair("hl_evals", r); rec["cost"] = cost(r); out["runs"][r] = rec
        if rec["headline"]:
            recs.append(rec)
    if recs:
        a = {"n_seeds": len(recs), "headline_instrument": sorted({r["headline_instrument"] for r in recs})}
        for inst in ("fp32", "bf16"):
            ms = [r[inst] for r in recs if r[inst]]
            if ms:
                a[inst] = {f: st.mean(m[f] for m in ms) for f in F if all(f in m for m in ms)} | {"seed_pressure": [m["pressure_l2"] for m in ms]}
        sh = [r["shift_bf16_minus_fp32_pressure"] for r in recs if r.get("shift_bf16_minus_fp32_pressure")]
        if sh:
            a["shift_bf16_minus_fp32_pressure"] = {"mean": st.mean(x["mean"] for x in sh), "max_abs_per_case": max(x["max_abs_per_case"] for x in sh)}
        costs = [r["cost"] for r in recs if r.get("cost")]
        if costs:
            a["parameters"] = costs[0]["parameters"]
            mems = [c["peak_mem_gb"] for c in costs if c.get("peak_mem_gb")]; a["peak_mem_gb"] = max(mems) if mems else None
            meds = [c["median_step_s"] for c in costs if c.get("median_step_s")]; a["median_step_s"] = st.mean(meds) if meds else None
            gh = [c["gpu_hours_500ep"] for c in costs if c.get("gpu_hours_500ep")]; a["gpu_hours_500ep"] = st.mean(gh) if gh else None
        out["arms"][arm] = a
for name, (root, runs) in REFS.items():
    recs = [instrument_pair(root, r) for r in runs]
    recs = [r for r in recs if r["headline"]]
    if not recs:
        out["refs"][name] = None; continue
    ref = {"n_seeds": len(recs), "headline_instrument": sorted({r["headline_instrument"] for r in recs})}
    for inst in ("fp32", "bf16"):
        ms = [r[inst] for r in recs if r[inst]]
        if ms:
            ref[inst] = {f: st.mean(m[f] for m in ms) for f in F if all(f in m for m in ms)} | {"seed_pressure": [m["pressure_l2"] for m in ms]}
    costs = [c for c in (cost(r) for r in runs) if c]
    if costs:
        ref["parameters"] = costs[0]["parameters"]
        meds = [c["median_step_s"] for c in costs if c.get("median_step_s")]; ref["median_step_s"] = st.mean(meds) if meds else None
        gh = [c["gpu_hours_500ep"] for c in costs if c.get("gpu_hours_500ep")]; ref["gpu_hours_500ep"] = st.mean(gh) if gh else None
        mems = [c["peak_mem_gb"] for c in costs if c.get("peak_mem_gb")]; ref["peak_mem_gb"] = max(mems) if mems else None
    out["refs"][name] = ref
# Density-bias probe (program redirect 2026-09-10): biased (10:1 sampling density, drivaer_probe_biased3) over
# uniform (drivaer_probe_unif2) pressure error, float32, each model at its own training resolution. Reference
# probes come from the transfer campaign (transfer/campaign_e_fp32/<run>/{unif,biased}); the 512 x 80k corner
# arm from hl_evals_probe_fp32/<run>/{unif,biased}.
PROBES = {"dr_ref_unit_lr3e3": ("transfer/campaign_e_fp32", ["uw_transolver_unit_lr3e3_seed42", "uw_transolver_unit_lr3e3_seed43"]),
          "dr_c512x80k": ("hl_evals_probe_fp32", ["scale_transolver_dr_c512x80k_seed42", "scale_transolver_dr_c512x80k_seed43"])}
out["density_probe"] = {}
for name, (root, runs) in PROBES.items():
    per = []
    for r in runs:
        if not (log_ok(f"{T}/{root}/{r}/unif.log") and log_ok(f"{T}/{root}/{r}/biased.log")):
            continue
        u = rows(f"{root}/{r}/unif", ""); b = rows(f"{root}/{r}/biased", "")
        if u and b:
            mu, mb = means(u)["pressure_l2"], means(b)["pressure_l2"]
            per.append({"run": r, "uniform": mu, "biased": mb, "ratio": mb / mu})
    out["density_probe"][name] = {"runs": per, "ratio_mean": st.mean(x["ratio"] for x in per) if per else None,
                                  "uniform_mean": st.mean(x["uniform"] for x in per) if per else None,
                                  "biased_mean": st.mean(x["biased"] for x in per) if per else None}
out["loader_hazard_check"] = HAZARD
json.dump(out, open(f"{T}/hl_evals/transolver_scale_reduction.json", "w"), indent=1)
print("LOADER_HAZARD", HAZARD)
for k, v in out["density_probe"].items():
    print("PROBE", k, json.dumps(v)[:300])
for k, v in out["refs"].items():
    print("REF", k, json.dumps(v)[:300] if v else None)
for k, v in out["arms"].items():
    print(k, json.dumps(v)[:400])
