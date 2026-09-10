"""Reduce the GeoTransolver arm of the SCALE study (notebook #sec-nb-scale-prereg, #sec-nb-scale-gt-verdict).

Reporting instrument (program-wide protocol, 2026-09-09): float32 inference is the headline; the bf16
evaluation is reported alongside with the per-run shift. For every run scale_gt_<hl|dr>_<arm>_seed<s>
(arms w384, w512, t40k, t80k, c384x40k) and the reference arms, read
  - hl_evals_fp32/<run>/**/metrics.jsonl (float32 inference; headline),
  - hl_evals/<run>/**/metrics.jsonl (bf16; alongside), asserting per case that both artifacts sampled
    bit-identical points (interior/points.memmap),
  - runs/<run>/train.log (peak "Mem:", median step time, parameter count),
and report per-run values, two-seed means per arm, ratios to the reference (fp32 vs fp32 when the
reference's fp32 re-evaluation exists under hl_evals_fp32/ or iw_evals_fp32/, otherwise labelled
fp32-vs-bf16), and GPU-hours. Runs on the login node with the recipe venv python (reads only).
Writes $T/hl_evals/scale_gt_reduction.json.
"""
import glob, json, os, re, statistics as st
import numpy as np

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
F = ("pressure_l2", "velocity_l2", "tau_wall_l2", "wss_x_l2", "wss_y_l2", "wss_z_l2")  # HiLift: pressure/velocity/tau_wall; DrivAerML surface: pressure + wss components
ARMS = ("w384", "w512", "t40k", "t80k", "c384x40k", "c512x80k")  # c512x80k: DrivAerML only (coordinator amendment 2026-09-10)
REF = {  # unit-drive references (main session) with the physical-drive fallbacks
    "hl": {"unit": ["udrv_hl_gt_full_seed42", "udrv_hl_gt_full_seed43"], "physical": ["gt_hl_lr1_seed42", "gt_hl_lr1_seed43"]},
    "dr": {"unit": ["uw_gt_unit_lr1e3_seed42", "uw_gt_unit_lr1e3_seed43", "uw_gt_unit_lr1e3_seed44"], "physical": ["iw_gt_lr1e3_seed42", "iw_gt_lr1e3_seed43"]},
}
TOKENS = {"w384": 10000, "w512": 10000, "t40k": 40000, "t80k": 80000, "c384x40k": 40000, "c512x80k": 80000, "ref": 10000}
WIDTH = {"w384": 384, "w512": 512, "t40k": 256, "t80k": 256, "c384x40k": 384, "c512x80k": 512, "ref": 256}
STEPS_PER_EPOCH = {"hl": 1260 // 4, "dr": 435 // 4}


def find_metrics(run, roots):
    for root in roots:
        ps = glob.glob(f"{T}/{root}/{run}/**/metrics.jsonl", recursive=True)
        if ps:
            return ps[0], root
    return None, None


def per_case(path):
    rows = [json.loads(l) for l in open(path)]
    return {r["sample_id"]: r["metrics"] for r in rows if r.get("phase") == "infer_step"}


def summarize(pc):
    keys = [f for f in F if all(f in v for v in pc.values())]
    return {f: st.mean(v[f] for v in pc.values()) for f in keys} | {"n_cases": len(pc)}


def points_identical(path_a, path_b):
    """Both artifacts must have sampled the same points per case (bit-identical memmaps)."""
    da, db = os.path.dirname(path_a), os.path.dirname(path_b)
    n_checked, n_bad = 0, 0
    for cdir in sorted(glob.glob(f"{da}/predictions/*.pdmsh")):
        cid = os.path.basename(cdir)
        pa = f"{cdir}/_tensordict/interior/_tensordict/points.memmap"
        pb = f"{db}/predictions/{cid}/_tensordict/interior/_tensordict/points.memmap"
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        a = np.memmap(pa, dtype=np.float32, mode="r"); b = np.memmap(pb, dtype=np.float32, mode="r")
        n_checked += 1
        if a.shape != b.shape or not np.array_equal(np.asarray(a), np.asarray(b)):
            n_bad += 1
    return {"cases_checked": n_checked, "cases_with_different_points": n_bad}


def eval_pair(run):
    """fp32 headline, bf16 alongside, per-run shift (mean and per-case max of |fp32 - bf16| / bf16 on pressure)."""
    p32, r32 = find_metrics(run, ("hl_evals_fp32", "iw_evals_fp32"))
    p16, r16 = find_metrics(run, ("hl_evals", "iw_evals"))
    rec = {}
    pc32 = per_case(p32) if p32 else None
    pc16 = per_case(p16) if p16 else None
    if pc32:
        rec["fp32"] = summarize(pc32) | {"root": r32}
    if pc16:
        rec["bf16"] = summarize(pc16) | {"root": r16}
    if pc32 and pc16:
        common = sorted(set(pc32) & set(pc16))
        rel = [(pc32[k]["pressure_l2"] - pc16[k]["pressure_l2"]) / pc16[k]["pressure_l2"] for k in common]
        rec["shift_fp32_minus_bf16_pressure"] = {"mean_rel": st.mean(rel), "max_abs_rel_per_case": max(abs(x) for x in rel), "n_cases": len(common),
                                                  "mean_ratio": rec["fp32"]["pressure_l2"] / rec["bf16"]["pressure_l2"]}
        rec["points_identity"] = points_identical(p32, p16)
    rec["headline"] = "fp32" if pc32 else ("bf16" if pc16 else None)
    return rec


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


def headline_value(ev, f="pressure_l2"):
    h = ev.get("headline")
    return (ev[h][f], h) if h and f in ev[h] else (None, None)


out = {"instrument": "float32 inference headline; bf16 alongside; shift = (fp32 - bf16)/bf16 on per-case pressure", "runs": {}, "arms": {}, "references": {}}
for ds in ("hl", "dr"):
    for kind, runs in REF[ds].items():
        evs = {r: eval_pair(r) for r in runs}
        vals = [(headline_value(e)[0], headline_value(e)[1]) for e in evs.values() if headline_value(e)[0] is not None]
        rec = {"runs": runs, "n_seeds": len(vals), "per_run": evs}
        if vals:
            rec["pressure_l2"] = st.mean(v for v, _ in vals)
            rec["instrument"] = "fp32" if all(h == "fp32" for _, h in vals) else ("bf16" if all(h == "bf16" for _, h in vals) else "mixed")
        out["references"][f"{ds}_{kind}"] = rec
    for arm in ARMS:
        runs = [f"scale_gt_{ds}_{arm}_seed{s}" for s in (42, 43)]
        per = {}
        for r in runs:
            per[r] = {"eval": eval_pair(r), "train": train_stats(r)}
            out["runs"][r] = per[r]
        rec = {"width": WIDTH[arm], "tokens": TOKENS[arm], "runs": runs}
        for inst in ("fp32", "bf16"):
            done = [per[r]["eval"][inst] for r in runs if inst in per[r]["eval"]]
            if done:
                rec[inst] = {f: st.mean(v[f] for v in done) for f in F if all(f in v for v in done)} | {"n_seeds": len(done)}
        shifts = [per[r]["eval"]["shift_fp32_minus_bf16_pressure"] for r in runs if "shift_fp32_minus_bf16_pressure" in per[r]["eval"]]
        if shifts:
            rec["shift_fp32_minus_bf16_pressure"] = {"mean_rel": st.mean(s["mean_rel"] for s in shifts), "max_abs_rel_per_case": max(s["max_abs_rel_per_case"] for s in shifts)}
            rec["points_identity"] = {"cases_with_different_points": sum(per[r]["eval"]["points_identity"]["cases_with_different_points"] for r in runs if "points_identity" in per[r]["eval"])}
        head = "fp32" if "fp32" in rec else ("bf16" if "bf16" in rec else None)
        rec["headline"] = head
        if head:
            ref = out["references"][f"{ds}_unit"] if out["references"][f"{ds}_unit"].get("n_seeds") else out["references"][f"{ds}_physical"]
            rec["reference_used"] = ("unit" if out["references"][f"{ds}_unit"].get("n_seeds") else "physical")
            if "pressure_l2" in ref:
                rec["pressure_over_reference"] = rec[head]["pressure_l2"] / ref["pressure_l2"]
                rec["comparison_instruments"] = f"arm {head} vs reference {ref.get('instrument')}"
        tr = [per[r]["train"] for r in runs if per[r]["train"] and "median_step_s" in per[r]["train"]]
        if tr:
            rec["peak_mem_gb"] = max(t["peak_mem_gb"] for t in tr)
            rec["median_step_s"] = st.mean(t["median_step_s"] for t in tr)
            rec["params"] = next((t["params"] for t in tr if t.get("params")), None)
            rec["gpu_hours_500_epochs"] = 500 * STEPS_PER_EPOCH[ds] * rec["median_step_s"] * 4 / 3600
        out["arms"][f"{ds}_{arm}"] = rec
json.dump(out, open(f"{T}/hl_evals/scale_gt_reduction.json", "w"), indent=1)
for k, v in out["references"].items():
    print("REF", k, {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in v.items() if kk not in ("runs", "per_run")})
for k, v in out["arms"].items():
    print(k, {kk: (round(x, 4) if isinstance(x, float) else x) for kk, x in v.items() if kk != "runs"})
