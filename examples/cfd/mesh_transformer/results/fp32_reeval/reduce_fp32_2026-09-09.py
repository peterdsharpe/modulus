"""FP32-REEVAL reduction (notebook #sec-nb-fp32-reeval-prereg).

For every run that has both a bf16 evaluation (<root>/<run>/) and a float32 evaluation
(<root>_fp32/<run>/) on the cluster task directory, this script
  * pairs the per-case metrics from both metrics.jsonl files (phase == infer_step, keyed by sample_id),
  * computes per field the bf16 mean, the fp32 mean, the relative shift (fp32 / bf16 - 1) of the mean
    and the maximum absolute per-case relative shift,
  * asserts point identity: for up to N_CHECK shared cases, np.array_equal on the saved interior points
    and (where present) the boundary points of the two prediction artifacts,
  * assigns each run an architecture (isla / geotransolver / transolver / mesh_transformer / other)
    and a family from its run-id prefix,
and writes one JSON with per-run rows and per-architecture / per-family summaries (median and range
of the pressure shift, count of runs, count failing the 1% / 5% reporting-equivalence bars).

Usage: python reduce_fp32_2026-09-09.py <out.json> [--only PREFIX,PREFIX,...]
Runs on the login node (metrics files are small; the point check reads two small memmaps per case).
"""
import glob, json, os, statistics as st, sys
import numpy as np

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
ROOTS = ("hl_evals", "iw_evals", "v0_evals")
FIELDS = ("pressure_l2", "velocity_l2", "tau_wall_l2", "nut_l2")
N_CHECK = 6
out_path = sys.argv[1]
only = None
if "--only" in sys.argv:
    only = tuple(sys.argv[sys.argv.index("--only") + 1].split(","))


def metrics(root, run):
    ps = glob.glob(f"{T}/{root}/{run}/*/metrics.jsonl")
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0])]
    return {r["sample_id"]: r["metrics"] for r in rows if r.get("phase") == "infer_step"}


def mm(path):
    d, name = os.path.split(path)
    meta = json.load(open(f"{d}/meta.json"))[name.replace(".memmap", "")]
    dt = {"torch.float32": np.float32, "torch.int64": np.int64}[meta["dtype"]]
    return np.asarray(np.memmap(path, dtype=dt, mode="r", shape=tuple(meta["shape"])))


def points_identical(root, run):
    """Compare interior points (and boundary points where present) of the bf16 and fp32 artifacts."""
    a_dirs = sorted(glob.glob(f"{T}/{root}/{run}/{run}/predictions/*"))
    b_dirs = sorted(glob.glob(f"{T}/{root}_fp32/{run}/{run}/predictions/*"))
    a = {os.path.basename(d): d for d in a_dirs}; b = {os.path.basename(d): d for d in b_dirs}
    shared = sorted(set(a) & set(b))[:N_CHECK]
    if not shared:
        return {"checked": 0, "identical": None}
    ok = True; detail = []
    for cid in shared:
        for sub in ("interior/_tensordict/points.memmap", "boundaries/vehicle/_tensordict/points.memmap"):
            pa, pb = f"{a[cid]}/_tensordict/{sub}", f"{b[cid]}/_tensordict/{sub}"
            if os.path.exists(pa) and os.path.exists(pb):
                same = np.array_equal(mm(pa), mm(pb))
                ok &= same
                if not same:
                    detail.append(f"{cid}:{sub.split('/')[0]}")
    return {"checked": len(shared), "identical": bool(ok), "mismatches": detail}


def arch_of(run):
    if any(k in run for k in ("_mt2_", "_isla_", "isla_", "mt2_")):
        return "mesh_transformer" if "mt1k" in run else "isla"
    if "_gt_" in run or run.startswith("gt_") or "gt_vol" in run or "iw_gt" in run or "uw_gt" in run:
        return "geotransolver"
    if "transolver" in run:
        return "transolver"
    if "mt1k" in run:
        return "mesh_transformer"
    return "other"


def family_of(run):
    for pre in ("lad_hl_", "a35_hl_", "a35b_hl_", "udrv_hl_", "t1_hl_", "mech_hl_", "geo_hl_", "floor_hl_", "defl_hl_", "now_hl_", "ctrl_hl_",
                "iw_", "uw_", "now_mt2", "mom2_", "v0l_", "v0_", "udrv_gt_vol", "mt2_hl_", "gt_hl_", "inv_hl_", "b1_hl_"):
        if run.startswith(pre):
            return pre.rstrip("_")
    return run.split("_seed")[0]


rows = {}
for root in ROOTS:
    for d in sorted(glob.glob(f"{T}/{root}_fp32/*/")):
        run = os.path.basename(d.rstrip("/"))
        if only and not run.startswith(only):
            continue
        if not os.path.exists(f"{d}/.done"):
            continue
        b16, f32 = metrics(root, run), metrics(f"{root}_fp32", run)
        if not b16 or not f32:
            continue
        shared = sorted(set(b16) & set(f32))
        rec = {"root": root, "arch": arch_of(run), "family": family_of(run), "n_cases": len(shared), "fields": {}}
        for f in FIELDS:
            if not all(f in b16[k] and f in f32[k] for k in shared):
                continue
            xb = np.array([b16[k][f] for k in shared]); xf = np.array([f32[k][f] for k in shared])
            good = np.isfinite(xb) & np.isfinite(xf) & (xb > 0)
            rec["fields"][f] = {"bf16_mean": float(xb[good].mean()), "fp32_mean": float(xf[good].mean()),
                                "shift": float(xf[good].mean() / xb[good].mean() - 1.0),
                                "per_case_max_abs_shift": float(np.max(np.abs(xf[good] / xb[good] - 1.0))),
                                "n": int(good.sum())}
        rec["points"] = points_identical(root, run)
        p = rec["fields"].get("pressure_l2")
        rec["reporting_equivalent"] = bool(p and abs(p["shift"]) <= 0.01 and p["per_case_max_abs_shift"] <= 0.05)
        rows[run] = rec
        print(f"{run:55s} p bf16 {p['bf16_mean']:.5f} fp32 {p['fp32_mean']:.5f} shift {100*p['shift']:+.2f}% max|case| {100*p['per_case_max_abs_shift']:.1f}%  points {rec['points'].get('identical')}" if p else f"{run}: no pressure field", flush=True)


def summary(group_key):
    out = {}
    for g in sorted({r[group_key] for r in rows.values()}):
        sh = [r["fields"]["pressure_l2"]["shift"] for r in rows.values() if r[group_key] == g and "pressure_l2" in r["fields"]]
        mx = [r["fields"]["pressure_l2"]["per_case_max_abs_shift"] for r in rows.values() if r[group_key] == g and "pressure_l2" in r["fields"]]
        if sh:
            out[g] = {"n_runs": len(sh), "median_shift": float(st.median(sh)), "min_shift": float(min(sh)), "max_shift": float(max(sh)),
                      "median_per_case_max": float(st.median(mx)), "n_not_reporting_equivalent": int(sum(not r["reporting_equivalent"] for r in rows.values() if r[group_key] == g)),
                      "n_points_mismatch": int(sum(r["points"].get("identical") is False for r in rows.values() if r[group_key] == g))}
    return out


res = {"n_runs": len(rows), "runs": rows, "by_architecture": summary("arch"), "by_family": summary("family"),
       "bars": {"reporting_equivalent": "|shift| <= 1% and per-case max <= 5% (pressure)", "architecture_dependent": "per-architecture medians differ by > 1 pt"}}
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps({"by_architecture": res["by_architecture"]}, indent=1))
