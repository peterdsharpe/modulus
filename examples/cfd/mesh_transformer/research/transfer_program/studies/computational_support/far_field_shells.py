"""Far-field diagnostic on saved float32 interior predictions: relative L2 by
distance shell (SDF / L) and by azimuth about the drive axis (dataset frame,
diagnostic only), for the reference, wake, LVT and GeoTransolver-volume arms.
Run with the recipe venv on the login node."""
import json, os, sys, statistics
import numpy as np

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
RUNS = {
    "isla_h344": ("v0_evals_fp32", ["v0_isla_qtsdf_h344_seed42", "v0_isla_qtsdf_h344_seed43"]),
    "isla_h344_wake": ("v0_evals_fp32", ["v0_isla_qtsdf_h344_wake_seed42", "v0_isla_qtsdf_h344_wake_seed43"]),
    "isla_h344_lvt": ("v0_evals_fp32", ["v0_isla_qtsdf_h344_lvt_seed42", "v0_isla_qtsdf_h344_lvt_seed43"]),
    "gt_unit": ("v0_evals_support_fp32", ["udrv_gt_vol_seed42", "udrv_gt_vol_seed43"]),
}
SHELLS = [("0.05-0.1L", 0.05, 0.1), ("0.1-0.2L", 0.1, 0.2), ("0.2-0.4L", 0.2, 0.4), (">=0.4L", 0.4, np.inf)]
FIELDS = ("nut", "pressure", "velocity")


def mm(d, name):
    meta = json.load(open(os.path.join(d, "meta.json")))[name]
    return np.memmap(os.path.join(d, f"{name}.memmap"), dtype=np.dtype(meta["dtype"].replace("torch.", "")), mode="r", shape=tuple(meta["shape"]))


def rel(err2, tru2, m):
    return float(np.sqrt(err2[m].sum() / max(tru2[m].sum(), 1e-30))) if m.sum() else float("nan")


def case_stats(pred_dir, case, want_keys=False):
    td = os.path.join(pred_dir, case, "_tensordict", "interior", "_tensordict")
    pd_ = os.path.join(td, "point_data")
    meta = json.load(open(os.path.join(pd_, "meta.json")))
    if want_keys:
        print("point_data keys:", sorted(meta.keys()))
        for up in (td, os.path.join(pred_dir, case, "_tensordict", "interior"), os.path.join(pred_dir, case, "_tensordict")):
            mj = os.path.join(up, "meta.json")
            if os.path.exists(mj):
                print(up.split("/_tensordict")[-1] or "/", "keys:", sorted(json.load(open(mj)).keys()))
    sdf = np.asarray(mm(pd_, "sdf")).ravel().astype(np.float64)
    gd = os.path.join(td, "global_data")
    try:
        L = float(np.asarray(mm(gd, "L_ref")).ravel()[0])
    except Exception:
        L = 1.0
    # point coordinates: try common names
    pts = None
    for root, name in ((td, "points"), (os.path.join(pred_dir, case, "_tensordict", "interior"), "points"), (pd_, "points"), (pd_, "coordinates")):
        try:
            pts = np.asarray(mm(root, name)).astype(np.float64).reshape(len(sdf), 3); break
        except Exception:
            continue
    out = {}
    for f in FIELDS:
        p = np.asarray(mm(pd_, f"pred_{f}")).astype(np.float64).reshape(len(sdf), -1)
        t = np.asarray(mm(pd_, f"true_{f}")).astype(np.float64).reshape(len(sdf), -1)
        err2 = ((p - t) ** 2).sum(-1); tru2 = (t ** 2).sum(-1)
        res = {}
        for name, lo, hi in SHELLS:
            m = (sdf >= lo * L) & (sdf < hi * L)
            res[name] = {"rel_l2": rel(err2, tru2, m), "share_pts": float(m.mean())}
        far = sdf >= 0.05 * L
        if pts is not None:
            # azimuth about the drive axis (dataset frame: drive +x, up +z): angle of (y, z) about the car's centroid line
            c = pts[sdf < 0.01 * L].mean(0) if (sdf < 0.01 * L).sum() else pts.mean(0)
            dy, dz = pts[:, 1] - c[1], pts[:, 2] - c[2]
            az = np.degrees(np.arctan2(dz, np.abs(dy)))  # 90 = straight above, 0 = beside, -90 = below
            for name, lo, hi in (("above (az>60)", 60, 91), ("oblique (20-60)", 20, 60), ("beside (-20-20)", -20, 20), ("below (<-20)", -91, -20)):
                m = far & (az >= lo) & (az < hi)
                res[f"far:{name}"] = {"rel_l2": rel(err2, tru2, m), "share_pts": float(m.mean())}
            # downstream vs upstream/around in the far band
            dx = pts[:, 0] - c[0]
            for name, m in (("far:downstream x>0.5L", far & (dx > 0.5 * L)), ("far:around |x|<=0.5L", far & (np.abs(dx) <= 0.5 * L)), ("far:upstream x<-0.5L", far & (dx < -0.5 * L))):
                res[name] = {"rel_l2": rel(err2, tru2, m), "share_pts": float(m.mean())}
        out[f] = res
    return out


def main():
    result = {"arms": {}, "shells": [s[0] for s in SHELLS]}
    first = True
    for arm, (root, runs) in RUNS.items():
        per_run = {}
        for run in runs:
            pred_dir = os.path.join(T, root, run, run, "predictions")
            if not os.path.isdir(pred_dir):
                print("missing", pred_dir); continue
            cases = sorted(os.listdir(pred_dir))
            acc = {}
            for c in cases:
                cs = case_stats(pred_dir, c, want_keys=first); first = False
                for f, res in cs.items():
                    for k, v in res.items():
                        acc.setdefault(f, {}).setdefault(k, []).append(v)
            per_run[run] = {f: {k: {"rel_l2": float(np.nanmean([x["rel_l2"] for x in vs])), "share_pts": statistics.mean(x["share_pts"] for x in vs)} for k, vs in d.items()} for f, d in acc.items()}
        if per_run:
            keys = next(iter(per_run.values()))["nut"].keys()
            result["arms"][arm] = {"runs": per_run, "mean": {f: {k: {"rel_l2": statistics.mean(r[f][k]["rel_l2"] for r in per_run.values()), "share_pts": statistics.mean(r[f][k]["share_pts"] for r in per_run.values())} for k in keys} for f in FIELDS}}
    dst = sys.argv[1] if len(sys.argv) > 1 else f"{T}/transfer/far_field_shells.json"
    json.dump(result, open(dst, "w"), indent=1)
    for f in FIELDS:
        print(f"== {f}: rel L2 by region (two-seed means) ==")
        keys = list(next(iter(result["arms"].values()))["mean"][f].keys())
        print("region | share | " + " | ".join(result["arms"].keys()))
        for k in keys:
            row = [f"{result['arms'][a]['mean'][f][k]['rel_l2']:.4f}" for a in result["arms"]]
            sh = next(iter(result["arms"].values()))["mean"][f][k]["share_pts"]
            print(f"{k} | {sh:.3f} | " + " | ".join(row))
    print("wrote", dst)


if __name__ == "__main__":
    main()
