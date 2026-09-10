"""Reduce the ISLA arm of the SCALE study (notebook #sec-nb-scale-prereg, verdict #sec-nb-scale-isla-verdict).

Per run: validation means of pressure / velocity / wall-shear relative L2 from the FLOAT32 evaluation
(hl_evals_fp32/<run>/**/metrics.jsonl, the program's reporting instrument since #sec-nb-snapshot-ladder-verdict) with the
bf16 evaluation (hl_evals/<run>/) alongside and the per-run fp32-to-bf16 shift; the two artifacts' sampled points are
asserted bit-identical per case. Reference arms use their fp32 re-evaluations when present (hl_evals_fp32 / iw_evals_fp32),
otherwise their bf16 numbers, labelled. Also
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
           "w384nw": ["scale_isla_hl_w384nw_seed42", "scale_isla_hl_w384nw_seed43"],
           "c512x80k": ["scale_isla_hl_c512x80k_seed42", "scale_isla_hl_c512x80k_seed43"]},
    "dr": {"ref": ["iw_mt2_lr1e3_seed42", "iw_mt2_lr1e3_seed43"],
           "w384": ["scale_isla_dr_w384_seed42", "scale_isla_dr_w384_seed43"],
           "w512": ["scale_isla_dr_w512_seed42", "scale_isla_dr_w512_seed43"],
           "t40k": ["scale_isla_dr_t40k_seed42", "scale_isla_dr_t40k_seed43"],
           "t80k": ["scale_isla_dr_t80k_seed42", "scale_isla_dr_t80k_seed43"],
           "c384x40k": ["scale_isla_dr_c384x40k_seed42", "scale_isla_dr_c384x40k_seed43"],
           "w384nw": ["scale_isla_dr_w384nw_seed42", "scale_isla_dr_w384nw_seed43"],
           "c512x80k": ["scale_isla_dr_c512x80k_seed42", "scale_isla_dr_c512x80k_seed43"],
           "kernelctrl": ["scale_isla_dr_kernelctrl_seed42", "scale_isla_dr_kernelctrl_seed43"]},
}
# Numerics families (coordinator amendment): t80k, c512x80k and kernelctrl were trained from code_perf (fast
# point-softmax kernel; bf16-roundoff-level differences from the reference checkpoints); everything else from code_isla5.
FAST_KERNEL = {"t80k", "c512x80k", "kernelctrl"}
STEP = re.compile(r"Epoch (\d+) \[(\d+)/(\d+)\] Loss: ([0-9.eE+-]+|nan) Step: ([0-9.]+)s Mem: ([0-9.]+)GB")


def _metrics_in(root, run):
    ps = glob.glob(f"{T}/{root}/{run}/*/metrics.jsonl") or glob.glob(f"{T}/{root}/{run}/metrics.jsonl")
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0])]
    rows = [r["metrics"] for r in rows if r.get("phase") == "infer_step"]
    return {f: st.mean(r[f] for r in rows) for f in F if f in rows[0]} | {"n_cases": len(rows)}


def _points_identical(run):
    """Assert the fp32 and bf16 artifacts sampled the same points (bit-identical) per case."""
    import numpy as np, os
    a = glob.glob(f"{T}/hl_evals/{run}/*/predictions") + glob.glob(f"{T}/iw_evals/{run}/*/predictions")
    b = glob.glob(f"{T}/hl_evals_fp32/{run}/*/predictions") + glob.glob(f"{T}/iw_evals_fp32/{run}/*/predictions")
    if not a or not b:
        return None
    cases = sorted(set(os.listdir(a[0])) & set(os.listdir(b[0])))
    for cid in cases:
        pa = np.memmap(f"{a[0]}/{cid}/_tensordict/interior/_tensordict/points.memmap", dtype=np.float32, mode="r")
        pb = np.memmap(f"{b[0]}/{cid}/_tensordict/interior/_tensordict/points.memmap", dtype=np.float32, mode="r")
        if pa.shape != pb.shape or not np.array_equal(pa, pb):
            raise AssertionError(f"{run} {cid}: fp32 and bf16 artifacts sampled different points")
    return len(cases)


def metrics(run):
    """fp32 headline with bf16 alongside. Returns None if neither evaluation exists."""
    fp32 = _metrics_in("hl_evals_fp32", run) or _metrics_in("iw_evals_fp32", run)
    bf16 = _metrics_in("hl_evals", run) or _metrics_in("iw_evals", run)
    if fp32 is None and bf16 is None:
        return None
    head = dict(fp32 if fp32 else bf16)
    head["instrument"] = "float32" if fp32 else "bf16 (fp32 re-evaluation not yet available)"
    if fp32 and bf16:
        head["bf16"] = {f: bf16[f] for f in F if f in bf16}
        head["fp32_over_bf16_pressure"] = fp32["pressure_l2"] / bf16["pressure_l2"]
        head["points_identical_cases"] = _points_identical(run)
    return head


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
            a["instrument"] = sorted({m["instrument"] for m, _ in per})
            if all("bf16" in m for m, _ in per):
                a["bf16_pressure_l2"] = st.mean(m["bf16"]["pressure_l2"] for m, _ in per)
                a["fp32_over_bf16_pressure"] = st.mean(m["fp32_over_bf16_pressure"] for m, _ in per)
            for k in ("params", "peak_mem_gb", "median_step_s", "gpu_hours"):
                v = [t[k] for _, t in per if t and t.get(k) is not None]
                a[k] = st.mean(v) if v else None
            a["numerics"] = "fast point-softmax (code_perf)" if arm in FAST_KERNEL else "reference kernel (code_isla5 / reference lanes)"
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
