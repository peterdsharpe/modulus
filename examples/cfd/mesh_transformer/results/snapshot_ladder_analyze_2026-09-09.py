"""SNAPSHOT-LADDER analysis (notebook #sec-nb-snapshot-ladder-prereg).
Reads the per-snapshot evaluations of iw_mt2_lr1e3_seed42 (reference kernel) under $T/iw_evals/snapladder_<S>/,
the training-time evaluation under $T/iw_evals/iw_mt2_lr1e3_seed42/, compares metrics and sampled point sets,
and builds the point-identity matrix for the program's cross-snapshot comparison pairs.
Usage: snapshot_ladder_analyze.py <out.json>"""
import glob, json, os, statistics as st, sys
import numpy as np

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
SNAPS = ["code", "code_isla2", "code_isla3", "code_isla4", "code_isla5", "code_perf", "code_recipe_support"]
out = {"ladder": {}, "point_identity_ladder": {}, "pair_matrix": {}}


def metrics(d):
    ps = glob.glob(f"{d}/**/metrics.jsonl", recursive=True)
    if not ps:
        return None
    rows = [json.loads(l) for l in open(ps[0])]
    return {r["sample_id"]: r["metrics"] for r in rows if r.get("phase") == "infer_step"}


def mm(path):
    d, name = os.path.split(path)
    meta = json.load(open(f"{d}/meta.json"))[name.replace(".memmap", "")]
    dt = {"torch.float32": np.float32, "torch.int64": np.int64}[meta["dtype"]]
    return np.asarray(np.memmap(path, dtype=dt, mode="r", shape=tuple(meta["shape"])))


def pred_dirs(root):
    ps = sorted(glob.glob(f"{root}/**/predictions", recursive=True))
    return ps[0] if ps else None


def points(pred_root, case):
    base = f"{pred_root}/{case}/_tensordict"
    it = mm(f"{base}/interior/_tensordict/points.memmap")
    bp = f"{base}/boundaries/vehicle/_tensordict/points.memmap"
    return it, (mm(bp) if os.path.exists(bp) else None)


def identity(a, b):
    if a is None or b is None:
        return "missing"
    if a.shape != b.shape:
        return f"different_shape {a.shape} vs {b.shape}"
    if np.allclose(a, b, atol=1e-6):
        return "identical"
    sa, sb = np.lexsort(a.T[::-1]), np.lexsort(b.T[::-1])
    if np.allclose(a[sa], b[sb], atol=1e-6):
        return "permutation"
    return "different_sample"


# --- ladder metrics
ref_train = metrics(f"{T}/iw_evals/iw_mt2_lr1e3_seed42")
base = None
for s in SNAPS:
    m = metrics(f"{T}/iw_evals/snapladder_{s}")
    if m is None:
        out["ladder"][s] = None; continue
    keys = sorted(m)
    rec = {f: st.mean(m[k][f] for k in keys) for f in m[keys[0]]} | {"n_cases": len(keys)}
    if base is None:
        base = m
    ks = sorted(set(m) & set(base))
    rec["max_per_case_rel_change_vs_code"] = {f: max(abs(m[k][f] - base[k][f]) / max(abs(base[k][f]), 1e-12) for k in ks) for f in m[keys[0]]}
    if ref_train:
        kt = sorted(set(m) & set(ref_train))
        rec["max_per_case_rel_change_vs_training_time"] = {f: max(abs(m[k][f] - ref_train[k][f]) / max(abs(ref_train[k][f]), 1e-12) for k in kt) for f in m[keys[0]] if f in ref_train[kt[0]]}
    out["ladder"][s] = rec
    print(s, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items() if not isinstance(v, dict)}, "max case Δp vs code", round(rec["max_per_case_rel_change_vs_code"]["pressure_l2"], 3), flush=True)
if ref_train:
    keys = sorted(ref_train)
    out["training_time_eval"] = {f: st.mean(ref_train[k][f] for k in keys) for f in ref_train[keys[0]]}
    print("training-time eval", {k: round(v, 5) for k, v in out["training_time_eval"].items()})

# --- point identity along the ladder (two fixed cars) + against the training-time artifact
roots = {s: pred_dirs(f"{T}/iw_evals/snapladder_{s}") for s in SNAPS}
roots["training_time"] = pred_dirs(f"{T}/iw_evals/iw_mt2_lr1e3_seed42")
avail = {s: r for s, r in roots.items() if r}
cases = sorted(os.listdir(roots["training_time"]))[:2] if roots.get("training_time") else sorted(os.listdir(next(iter(avail.values()))))[:2]
for case in cases:
    out["point_identity_ladder"][case] = {}
    ref_pts = points(avail["code"] if "code" in avail else next(iter(avail.values())), case)
    for s, r in avail.items():
        try:
            p = points(r, case)
            out["point_identity_ladder"][case][s] = {"interior_vs_code": identity(ref_pts[0], p[0]), "boundary_vs_code": identity(ref_pts[1], p[1])}
        except Exception as e:  # noqa: BLE001
            out["point_identity_ladder"][case][s] = {"error": str(e)[:200]}
    print(case, out["point_identity_ladder"][case], flush=True)

# --- pair matrix for the program's cross-snapshot comparisons (one shared case each)
PAIRS = [("v0_gt_vol_seed42 (v0_evals, code)", f"{T}/v0_evals/v0_gt_vol_seed42", "v0_isla_qtsdfval_seed42 (v0_evals, code_isla2)", f"{T}/v0_evals/v0_isla_qtsdfval_seed42"),
         ("v0_gt_vol_seed42 (v0_evals, code)", f"{T}/v0_evals/v0_gt_vol_seed42", "v0_isla_qtsdf_h256_seed42 (v0_evals, code_isla3)", f"{T}/v0_evals/v0_isla_qtsdf_h256_seed42"),
         ("v0_gt_vol_seed42 (v0_evals, code)", f"{T}/v0_evals/v0_gt_vol_seed42", "udrv_gt_vol_seed42 (v0_evals, code)", f"{T}/v0_evals/udrv_gt_vol_seed42"),
         ("v0_isla_qtsdfval_seed42 (code_isla2)", f"{T}/v0_evals/v0_isla_qtsdfval_seed42", "v0_isla_qtsdf_surf10k_seed42 (code_isla4)", f"{T}/v0_evals/v0_isla_qtsdf_surf10k_seed42"),
         ("iw_mt2_lr1e3_seed42 (iw_evals, code)", f"{T}/iw_evals/iw_mt2_lr1e3_seed42", "iw_gt_lr1e3_seed42 (iw_evals, code)", f"{T}/iw_evals/iw_gt_lr1e3_seed42"),
         ("iw_mt2_lr1e3_seed42 (iw_evals, code)", f"{T}/iw_evals/iw_mt2_lr1e3_seed42", "uw_gt_unit_lr1e3_seed42 (iw_evals, code)", f"{T}/iw_evals/uw_gt_unit_lr1e3_seed42"),
         ("iw_mt2_lr1e3_seed42 (iw_evals, code)", f"{T}/iw_evals/iw_mt2_lr1e3_seed42", "mom2_mt2_lr1e3_seed42 (iw_evals, code_isla5)", f"{T}/iw_evals/mom2_mt2_lr1e3_seed42"),
         ("iw_mt2_lr1e3_seed42 (iw_evals, code)", f"{T}/iw_evals/iw_mt2_lr1e3_seed42", "now_mt2_lr1e3_seed42 (iw_evals, code)", f"{T}/iw_evals/now_mt2_lr1e3_seed42"),
         ("lad_hl_mt2_super_scarce_seed42 (hl_evals, code)", f"{T}/hl_evals/lad_hl_mt2_super_scarce_seed42", "udrv_hl_gt_super_scarce_seed42 (hl_evals, code)", f"{T}/hl_evals/udrv_hl_gt_super_scarce_seed42")]
for na, ra, nb, rb in PAIRS:
    pa, pb = pred_dirs(ra), pred_dirs(rb)
    key = f"{na}  vs  {nb}"
    if not pa or not pb:
        out["pair_matrix"][key] = "missing evaluation"; print(key, "missing"); continue
    shared = sorted(set(os.listdir(pa)) & set(os.listdir(pb)))
    if not shared:
        out["pair_matrix"][key] = "no shared case"; print(key, "no shared case"); continue
    c = shared[0]
    try:
        A, B = points(pa, c), points(pb, c)
        out["pair_matrix"][key] = {"case": c, "interior": identity(A[0], B[0]), "boundary": identity(A[1], B[1])}
    except Exception as e:  # noqa: BLE001
        out["pair_matrix"][key] = {"case": c, "error": str(e)[:200]}
    print(key, out["pair_matrix"][key], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
