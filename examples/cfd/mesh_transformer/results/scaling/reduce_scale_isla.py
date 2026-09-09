"""Reduce the ISLA arm of the SCALE study (notebook #sec-nb-scale-prereg, verdict #sec-nb-scale-isla-verdict).

Per run: validation means of pressure / velocity / wall-shear relative L2 from hl_evals/<run>/**/metrics.jsonl,
parameter count, peak logged memory and median step time from runs/<run>/train.log, and GPU-hours as the sum of
logged step times times four GPUs. Per arm: two-seed means. Runs on the cluster login node with the recipe venv.
Usage: python reduce_scale_isla.py <out.json>
"""
import glob, json, re, statistics as st, sys

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
F = ("pressure_l2", "velocity_l2", "tau_wall_l2")
ARMS = {
    "hl": {"ref": ["mt2_hl_lr1_seed42", "mt2_hl_lr1_seed43"],
           "w384": ["scale_isla_hl_w384_seed42", "scale_isla_hl_w384_seed43"],
           "w512": ["scale_isla_hl_w512_seed42", "scale_isla_hl_w512_seed43"],
           "t20k": ["floor_hl_mt2_full_20k_seed42", "floor_hl_mt2_full_20k_seed43"],
           "t40k": ["scale_isla_hl_t40k_seed42", "scale_isla_hl_t40k_seed43"],
           "t80k": ["scale_isla_hl_t80k_seed42", "scale_isla_hl_t80k_seed43"],
           "c384x40k": ["scale_isla_hl_c384x40k_seed42", "scale_isla_hl_c384x40k_seed43"],
           "w384nw": ["scale_isla_hl_w384nw_seed42", "scale_isla_hl_w384nw_seed43"]},
    "dr": {"ref": ["iw_mt2_lr1e3_seed42", "iw_mt2_lr1e3_seed43"],
           "w384": ["scale_isla_dr_w384_seed42", "scale_isla_dr_w384_seed43"],
           "w512": ["scale_isla_dr_w512_seed42", "scale_isla_dr_w512_seed43"],
           "t40k": ["scale_isla_dr_t40k_seed42", "scale_isla_dr_t40k_seed43"],
           "t80k": ["scale_isla_dr_t80k_seed42", "scale_isla_dr_t80k_seed43"],
           "c384x40k": ["scale_isla_dr_c384x40k_seed42", "scale_isla_dr_c384x40k_seed43"],
           "w384nw": ["scale_isla_dr_w384nw_seed42", "scale_isla_dr_w384nw_seed43"]},
}
STEP = re.compile(r"Epoch (\d+) \[(\d+)/(\d+)\] Loss: ([0-9.eE+-]+|nan) Step: ([0-9.]+)s Mem: ([0-9.]+)GB")


def metrics(run):
    ps = glob.glob(f"{T}/hl_evals/{run}/**/metrics.jsonl", recursive=True) or glob.glob(f"{T}/iw_evals/{run}/**/metrics.jsonl", recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0])]
    rows = [r["metrics"] for r in rows if r.get("phase") == "infer_step"]
    return {f: st.mean(r[f] for r in rows) for f in F if f in rows[0]} | {"n_cases": len(rows)}


def training(run):
    try:
        txt = open(f"{T}/runs/{run}/train.log").read()
    except FileNotFoundError:
        return None
    params = re.findall(r"Parameters: ([\d,]+)", txt)
    steps = STEP.findall(txt)
    if not steps:
        return {"params": int(params[-1].replace(",", "")) if params else None}
    times = [float(s[4]) for s in steps]; mems = [float(s[5]) for s in steps]
    last_epoch = max(int(s[0]) for s in steps)
    nan_steps = sum(1 for s in steps if s[3] == "nan")
    return {"params": int(params[-1].replace(",", "")) if params else None, "peak_mem_gb": max(mems),
            "median_step_s": st.median(times), "gpu_hours": sum(times) * 4 / 3600, "logged_steps": len(steps),
            "last_epoch": last_epoch, "nan_steps": nan_steps, "completed": "Training completed" in txt}


out = {"runs": {}, "arms": {}}
for ds, arms in ARMS.items():
    for arm, runs in arms.items():
        per = []
        for r in runs:
            m, t = metrics(r), training(r)
            out["runs"][r] = {"metrics": m, "training": t}
            if m:
                per.append((m, t))
        if per:
            a = {f: st.mean(m[f] for m, _ in per) for f in F if all(f in m for m, _ in per)}
            a["n_seeds"] = len(per); a["seed_pressure"] = [m["pressure_l2"] for m, _ in per]
            for k in ("params", "peak_mem_gb", "median_step_s", "gpu_hours"):
                v = [t[k] for _, t in per if t and t.get(k) is not None]
                a[k] = st.mean(v) if v else None
            out["arms"][f"{ds}_{arm}"] = a
for ds in ARMS:
    ref = out["arms"].get(f"{ds}_ref")
    if ref:
        for arm in ARMS[ds]:
            a = out["arms"].get(f"{ds}_{arm}")
            if a:
                a["pressure_over_ref"] = a["pressure_l2"] / ref["pressure_l2"]
json.dump(out, open(sys.argv[1], "w"), indent=1)
for k, v in out["arms"].items():
    print(k, {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in v.items() if kk != "seed_pressure"})
