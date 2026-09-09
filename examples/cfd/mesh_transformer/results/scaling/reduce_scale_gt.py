"""Reduce the GeoTransolver arm of the SCALE study (notebook #sec-nb-scale-prereg, #sec-nb-scale-gt-verdict).

For every run scale_gt_<hl|dr>_<arm>_seed<s> (arms w384, w512, t40k, t80k, c384x40k) and the reference
arms, read the validation metrics (hl_evals/<run>/**/metrics.jsonl or iw_evals for the DrivAerML
references), the training log (peak "Mem:" figure, median step time, parameter count) and report
per-run values, two-seed means per arm, ratios to the reference, and GPU-hours. Runs on the cluster
login node with the recipe venv python (reads only). Writes $T/hl_evals/scale_gt_reduction.json.
"""
import glob, json, os, re, statistics as st

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
F = ("pressure_l2", "velocity_l2", "tau_wall_l2")
ARMS = ("w384", "w512", "t40k", "t80k", "c384x40k")
REF = {  # unit-drive references (main session) with the physical-drive fallbacks
    "hl": {"unit": ["udrv_hl_gt_full_seed42", "udrv_hl_gt_full_seed43"], "physical": ["gt_hl_lr1_seed42", "gt_hl_lr1_seed43"]},
    "dr": {"unit": ["uw_gt_unit_lr1e3_seed42", "uw_gt_unit_lr1e3_seed43", "uw_gt_unit_lr1e3_seed44"], "physical": ["iw_gt_lr1e3_seed42", "iw_gt_lr1e3_seed43"]},
}
TOKENS = {"w384": 10000, "w512": 10000, "t40k": 40000, "t80k": 80000, "c384x40k": 40000, "ref": 10000}
WIDTH = {"w384": 384, "w512": 512, "t40k": 256, "t80k": 256, "c384x40k": 384, "ref": 256}


def metrics(run):
    ps = glob.glob(f"{T}/hl_evals/{run}/**/metrics.jsonl", recursive=True) or glob.glob(f"{T}/iw_evals/{run}/**/metrics.jsonl", recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0])]
    rows = [r["metrics"] for r in rows if r.get("phase") == "infer_step"]
    return {f: st.mean(r[f] for r in rows) for f in F if f in rows[0]} | {"n_cases": len(rows)}


def train_stats(run):
    p = f"{T}/runs/{run}/train.log"
    if not os.path.exists(p):
        return None
    mem, steps, params, epochs, completed = [], [], None, 0, False
    for line in open(p, errors="ignore"):
        m = re.search(r"Step: ([0-9.]+)s Mem: ([0-9.]+)GB", line)
        if m:
            steps.append(float(m.group(1))); mem.append(float(m.group(2)))
        if params is None:
            mp = re.search(r"([0-9][0-9,]*)\s+(?:trainable\s+)?param", line, re.I)
            if mp:
                params = int(mp.group(1).replace(",", ""))
        me = re.search(r"--- Epoch (\d+)/", line)
        if me:
            epochs = int(me.group(1))
        if "Training completed" in line:
            completed = True
    if not steps:
        return {"completed": completed, "epochs_seen": epochs}
    med = st.median(steps[len(steps) // 10:]) if len(steps) > 20 else st.median(steps)
    return {"completed": completed, "epochs_seen": epochs, "peak_mem_gb": max(mem), "median_step_s": med,
            "n_logged_steps": len(steps), "params": params}


out = {"runs": {}, "arms": {}, "references": {}}
for ds in ("hl", "dr"):
    for kind, runs in REF[ds].items():
        vals = [m for m in (metrics(r) for r in runs) if m]
        out["references"][f"{ds}_{kind}"] = ({f: st.mean(v[f] for v in vals) for f in F if all(f in v for v in vals)} | {"n_seeds": len(vals), "runs": runs}) if vals else {"n_seeds": 0, "runs": runs}
    for arm in ARMS:
        runs = [f"scale_gt_{ds}_{arm}_seed{s}" for s in (42, 43)]
        per = {}
        for r in runs:
            per[r] = {"metrics": metrics(r), "train": train_stats(r)}
            out["runs"][r] = per[r]
        done = [per[r]["metrics"] for r in runs if per[r]["metrics"]]
        rec = {"width": WIDTH[arm], "tokens": TOKENS[arm], "n_seeds_evaluated": len(done), "runs": runs}
        if done:
            rec |= {f: st.mean(v[f] for v in done) for f in F if all(f in v for v in done)}
            ref = out["references"][f"{ds}_unit"] if out["references"][f"{ds}_unit"].get("n_seeds") else out["references"][f"{ds}_physical"]
            rec["reference_used"] = "unit" if out["references"][f"{ds}_unit"].get("n_seeds") else "physical"
            if "pressure_l2" in ref:
                rec["pressure_over_reference"] = rec["pressure_l2"] / ref["pressure_l2"]
        tr = [per[r]["train"] for r in runs if per[r]["train"] and "median_step_s" in per[r]["train"]]
        if tr:
            rec["peak_mem_gb"] = max(t["peak_mem_gb"] for t in tr)
            rec["median_step_s"] = st.mean(t["median_step_s"] for t in tr)
            rec["params"] = next((t["params"] for t in tr if t.get("params")), None)
            steps_per_epoch = {"hl": 1260 // 4, "dr": 435 // 4}[ds]
            rec["gpu_hours_500_epochs"] = 500 * steps_per_epoch * rec["median_step_s"] * 4 / 3600
        out["arms"][f"{ds}_{arm}"] = rec
json.dump(out, open(f"{T}/hl_evals/scale_gt_reduction.json", "w"), indent=1)
for k, v in out["references"].items():
    print("REF", k, {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in v.items() if kk != "runs"})
for k, v in out["arms"].items():
    print(k, {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in v.items() if kk != "runs"})
