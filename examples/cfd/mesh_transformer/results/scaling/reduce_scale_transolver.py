"""Reduce the Transolver SCALE lanes (notebook #sec-nb-scale-prereg, #sec-nb-scale-transolver-verdict).

Runs on the cluster login node with the recipe venv python (reads metrics.jsonl and train.log
only; no compute). For every run: validation means of pressure / velocity / wall-shear relative
L2, parameter count, peak "Mem:" from train.log, median step time (first step of each link
excluded), steps and GPU-hours (steps x median step time x 4 GPUs); per arm the two-seed means.
Reference arms (not rerun): HiLift udrv_hl_transolver_full_lr3e3_seed{42,43};
DrivAerML uw_transolver_unit_lr3e3_seed{42,43} and uw_transolver_unit_lr1e3_seed{42,43}.
Writes $T/hl_evals/transolver_scale_reduction.json; copy to
results/scaling/transolver_scale_reduction_2026-09-09.json.
"""
import glob, json, re, statistics as st

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
F = ("pressure_l2", "velocity_l2", "tau_wall_l2")
ARMS = {}
for ds in ("hl", "dr"):
    for arm in ("w384", "w512", "t40k", "t80k", "c384x40k"):
        ARMS[f"{ds}_{arm}"] = [f"scale_transolver_{ds}_{arm}_seed{s}" for s in (42, 43)]
REFS = {"hl_ref_unit_lr3e3": ("hl_evals", ["udrv_hl_transolver_full_lr3e3_seed42", "udrv_hl_transolver_full_lr3e3_seed43"]),
        "dr_ref_unit_lr3e3": ("iw_evals", ["uw_transolver_unit_lr3e3_seed42", "uw_transolver_unit_lr3e3_seed43"]),
        "dr_ref_unit_lr1e3": ("iw_evals", ["uw_transolver_unit_lr1e3_seed42", "uw_transolver_unit_lr1e3_seed43"])}


def metrics(root, run):
    ps = glob.glob(f"{T}/{root}/{run}/*/metrics.jsonl") + glob.glob(f"{T}/{root}/{run}/metrics.jsonl")
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0])]
    rows = [r["metrics"] for r in rows if r.get("phase") == "infer_step"]
    return {f: st.mean(r[f] for r in rows) for f in F if f in rows[0]} | {"n_cases": len(rows)}


def cost(run):
    try:
        log = open(f"{T}/runs/{run}/train.log").read()
    except FileNotFoundError:
        return None
    params = re.findall(r"Parameters: ([\d,]+)", log)
    steps = re.findall(r"Epoch \d+ \[(\d+)/(\d+)\] Loss: [\d.e+-]+ Step: ([\d.]+)s Mem: ([\d.]+)GB", log)
    if not steps:
        return {"parameters": int(params[-1].replace(",", "")) if params else None}
    times = [float(s[2]) for s in steps if int(s[0]) > 1]  # drop each link's first (warm-up) step
    mem = max(float(s[3]) for s in steps)
    per_epoch = int(steps[-1][1]); n_steps = len(steps)
    med = st.median(times) if times else None
    epochs_done = len(re.findall(r"--- Epoch \d+/\d+ ---", log))
    return {"parameters": int(params[-1].replace(",", "")) if params else None, "peak_mem_gb": mem,
            "median_step_s": med, "steps_logged": n_steps, "steps_per_epoch": per_epoch, "epochs_started": epochs_done,
            "gpu_hours": (500 * per_epoch * med * 4 / 3600) if med else None,
            "trained": "Training completed" in log}


out = {"runs": {}, "arms": {}, "refs": {}}
for arm, runs in ARMS.items():
    per = []
    for r in runs:
        m = metrics("hl_evals", r); c = cost(r)
        out["runs"][r] = {"metrics": m, "cost": c}
        if m:
            per.append(m)
    if per:
        out["arms"][arm] = {f: st.mean(p[f] for p in per) for f in F if all(f in p for p in per)} | {
            "n_seeds": len(per), "seed_pressure": [p["pressure_l2"] for p in per]}
        costs = [out["runs"][r]["cost"] for r in runs if out["runs"][r]["cost"]]
        if costs:
            out["arms"][arm]["parameters"] = costs[0]["parameters"]
            out["arms"][arm]["peak_mem_gb"] = max(c["peak_mem_gb"] for c in costs if c.get("peak_mem_gb"))
            meds = [c["median_step_s"] for c in costs if c.get("median_step_s")]
            out["arms"][arm]["median_step_s"] = st.mean(meds) if meds else None
            gh = [c["gpu_hours"] for c in costs if c.get("gpu_hours")]
            out["arms"][arm]["gpu_hours_500ep"] = st.mean(gh) if gh else None
for name, (root, runs) in REFS.items():
    per = [m for m in (metrics(root, r) for r in runs) if m]
    out["refs"][name] = ({f: st.mean(p[f] for p in per) for f in F if all(f in p for p in per)} | {"n_seeds": len(per)}) if per else None
    costs = [c for c in (cost(r) for r in runs) if c]
    if costs and out["refs"][name] is not None:
        out["refs"][name]["parameters"] = costs[0]["parameters"]; out["refs"][name]["median_step_s"] = st.mean(c["median_step_s"] for c in costs if c.get("median_step_s"))
        out["refs"][name]["gpu_hours_500ep"] = st.mean(c["gpu_hours"] for c in costs if c.get("gpu_hours")); out["refs"][name]["peak_mem_gb"] = max(c["peak_mem_gb"] for c in costs if c.get("peak_mem_gb"))
json.dump(out, open(f"{T}/hl_evals/transolver_scale_reduction.json", "w"), indent=1)
for k, v in out["refs"].items():
    print("REF", k, v)
for k, v in out["arms"].items():
    print(k, {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in v.items()})
