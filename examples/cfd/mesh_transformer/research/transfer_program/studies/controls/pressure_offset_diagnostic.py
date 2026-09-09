"""Pressure-offset versus spatial-error diagnostic (transfer program, controls track).

Question. On the DrivAerML -> SHIFT-SUV fastback zero-shot transfer, ISLA's relative
pressure error is about 0.99 and GeoTransolver's 0.59 (three seeds each), and adding
the SHIFT-SUV estate family to training brings both to 0.11-0.13. A relative error
near 1.0 is what a prediction with the wrong pressure REFERENCE produces: the two
datasets are simulated with different ground/wheel conditions, freestream speeds
and solvers, and their gauge-pressure conventions need not agree. Before any
architectural interpretation of the transfer gap, decompose each arm's error into
a per-case constant offset and a centered spatial remainder, and check whether the
offset cancels from the integrated pressure force on the (nearly) closed surface.

Instrument (eval-only, from saved predictions). For each validation case and arm
(seed-mean predictions), with cell areas a recomputed from the saved triangles and
cell normals n from the saved boundary mesh:
  e        = pred - true                       (per sampled cell, physical pressure
                                                as saved; p_inf subtracted when present)
  offset   = sum(a e) / sum(a)                 (area-weighted mean error)
  spatial  = e - offset
  rel_L2   = ||e|| / ||true - mean_a(true)||   (uniform, on gauge pressure about the
                                                area-weighted case mean, so that the
                                                denominator is the spatial signal)
  frac_off = sum(a offset^2) / sum(a e^2)      (share of area-weighted squared error
                                                carried by the constant)
  force    = sum(a p n) for pred and true; the offset contribution is offset * sum(a n),
             which vanishes on a closed surface with consistent normals; we report
             |sum(a n)| / sum(a) (closedness) and the force error with and without
             the offset.
Also: the true field's own offset relative to p_inf (is the target's reference the
same as the source's?), and the in-family DrivAerML arms as the reference behaviour.

Usage: python pressure_offset_diagnostic.py <out.json>
"""
import json, os, sys
import numpy as np

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
out = sys.argv[1]
# arm -> list of (root, run) whose predictions live at root/run/run/predictions (or root/run/<sub>/predictions)
ARMS = {
    "isla_source_only_fastback": [(f"{T}/smoke_fastback", f"mt2_v3c_seed{s}") for s in (42, 43, 44)],
    "gt_source_only_fastback": [(f"{T}/smoke_fastback", f"gt_fixed_seed{s}") for s in (45, 46)],  # seeds 42-44 of the round3 reduction are no longer on disk; 45/46 are the same configuration
    "isla_with_estate_fastback": [(f"{T}/smoke_fastback", f"mt2_multi_seed{s}") for s in (42, 43, 44)],
    "gt_with_estate_fastback": [(f"{T}/smoke_fastback", f"gt_multi_seed{s}") for s in (42, 43, 44)],
    "isla_infamily_drivaer": [(f"{T}/sweep", f"mt2_v3c_seed{s}/res10000") for s in (42, 43, 44)],
    "gt_infamily_drivaer": [(f"{T}/sweep", f"gt_fixed_seed{s}/res10000") for s in (45, 46)],
}


def mm(path):
    d, name = os.path.split(path)
    meta = json.load(open(f"{d}/meta.json"))[name.replace(".memmap", "")]
    dt = {"torch.float32": np.float32, "torch.int64": np.int64}[meta["dtype"]]
    return np.asarray(np.memmap(path, dtype=dt, mode="r", shape=tuple(meta["shape"])), dtype=np.float64)


def pred_root(root, run):
    for cand in (f"{root}/{run}/{run.split('/')[0]}/predictions", f"{root}/{run}/predictions"):
        if os.path.isdir(cand):
            return cand
    # one level of run_id subdirectory (infer.py writes <output_dir>/<run_id>/predictions)
    base = f"{root}/{run}"
    if os.path.isdir(base):
        for sub in sorted(os.listdir(base)):
            cand = f"{base}/{sub}/predictions"
            if os.path.isdir(cand):
                return cand
    return None


def case(pred_dir, cid):
    base = f"{pred_dir}/{cid}/_tensordict"
    it = f"{base}/interior/_tensordict"
    pts = mm(f"{it}/points.memmap"); pred = mm(f"{it}/point_data/pred_pressure.memmap"); true = mm(f"{it}/point_data/true_pressure.memmap")
    p_inf = 0.0
    if os.path.exists(f"{base}/global_data/p_inf.memmap"):
        p_inf = float(mm(f"{base}/global_data/p_inf.memmap").ravel()[0])
    bdirs = [d for d in os.listdir(f"{base}/boundaries") if os.path.isdir(f"{base}/boundaries/{d}") and d != "_tensordict"]
    bv = f"{base}/boundaries/{bdirs[0]}/_tensordict"
    P = mm(f"{bv}/points.memmap"); C = mm(f"{bv}/cells.memmap").astype(np.int64)
    cr = np.cross(P[C[:, 1]] - P[C[:, 0]], P[C[:, 2]] - P[C[:, 0]])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    nrm = mm(f"{bv}/cell_data/normals.memmap") if os.path.exists(f"{bv}/cell_data/normals.memmap") else cr / (2 * area[:, None] + 1e-30)
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30)
    return pts, pred - p_inf, true - p_inf, area, nrm, p_inf


res = {"arms": {}, "notes": {}}
for arm, runs in ARMS.items():
    dirs = [pred_root(r, run) for r, run in runs]
    if any(d is None for d in dirs):
        res["notes"][arm] = f"missing predictions: {[run for (r, run), d in zip(runs, dirs) if d is None]}"; continue
    cases = sorted(set.intersection(*[set(os.listdir(d)) for d in dirs]))
    rows = []
    for cid in cases:
        preds = []; ref = None
        for d in dirs:
            pts, p, t, a, n, p_inf = case(d, cid)
            if ref is None:
                ref = (pts, t, a, n, p_inf)
            elif not np.allclose(pts, ref[0], atol=1e-5):
                raise SystemExit(f"points differ across seeds: {arm} {cid}")
            preds.append(p)
        pts, t, a, n, p_inf = ref; p = np.mean(preds, axis=0)
        w = a / a.sum(); e = p - t
        off = float((w * e).sum()); sp = e - off
        t_mean = float((w * t).sum()); tc = t - t_mean
        closed = float(np.linalg.norm((a[:, None] * n).sum(0)) / a.sum())
        F_true = (a[:, None] * t[:, None] * n).sum(0); F_pred = (a[:, None] * p[:, None] * n).sum(0)
        F_pred_nooff = (a[:, None] * (p - off)[:, None] * n).sum(0)
        rows.append({
            "case": cid, "p_inf": p_inf, "true_mean_gauge": t_mean, "true_rms_centered": float(np.sqrt((w * tc ** 2).sum())),
            "offset": off, "offset_over_true_rms": off / (np.sqrt((w * tc ** 2).sum()) + 1e-30),
            "rel_l2_uniform_vs_true": float(np.linalg.norm(e) / np.linalg.norm(t)),
            "rel_l2_uniform_vs_centered_true": float(np.linalg.norm(e) / np.linalg.norm(tc)),
            "rel_l2_spatial_vs_centered_true": float(np.linalg.norm(sp) / np.linalg.norm(tc)),
            "frac_sq_err_offset_area": float(off ** 2 / ((w * e ** 2).sum() + 1e-30)),
            "closedness_|sum a n|/sum a": closed,
            "force_rel_err": float(np.linalg.norm(F_pred - F_true) / (np.linalg.norm(F_true) + 1e-30)),
            "force_rel_err_offset_removed": float(np.linalg.norm(F_pred_nooff - F_true) / (np.linalg.norm(F_true) + 1e-30)),
        })
    keys = [k for k in rows[0] if k != "case"]
    summ = {k: {"mean": float(np.mean([r[k] for r in rows])), "median": float(np.median([r[k] for r in rows])),
                "q10": float(np.quantile([r[k] for r in rows], .1)), "q90": float(np.quantile([r[k] for r in rows], .9))} for k in keys}
    summ["offset_sign_fraction_positive"] = float(np.mean([r["offset"] > 0 for r in rows]))
    res["arms"][arm] = {"n_cases": len(rows), "n_seeds": len(dirs), "summary": summ, "per_case": rows}
    print(arm, len(rows), "cases:", {k: round(summ[k]["mean"], 4) for k in ("rel_l2_uniform_vs_true", "rel_l2_uniform_vs_centered_true", "rel_l2_spatial_vs_centered_true", "frac_sq_err_offset_area", "offset_over_true_rms", "closedness_|sum a n|/sum a", "force_rel_err", "force_rel_err_offset_removed")}, flush=True)
json.dump(res, open(out, "w"), indent=1)
print("notes:", res["notes"])
