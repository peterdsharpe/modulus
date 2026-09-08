import json, os, glob, re, sys

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
E = os.path.join(T, "hl_evals")
runs = []
for arm in ["mt2", "gt"]:
    for s in [42, 43, 44]:
        runs.append(f"lad_hl_{arm}_super_scarce_seed{s}")
        runs.append(f"lad_hl_{arm}_scarce_seed{s}")
    for s in [42, 43]:
        runs.append(f"{arm}_hl_lr1_seed{s}")
        runs.append(f"geo_hl_{arm}_geometry_super_scarce_seed{s}")
        runs.append(f"geo_hl_{arm}_geometry_scarce_seed{s}")
        runs.append(f"mech_hl_{arm}_single_aoa_12_seed{s}")
for s in [42, 43, 44]:
    runs.append(f"t1_hl_transolver_super_scarce_lr3e3_seed{s}")
    runs.append(f"t1_hl_transolver_scarce_lr3e3_seed{s}")
for s in [42, 43]:
    runs.append(f"mech_hl_transolver_single_aoa_12_lr3e3_seed{s}")
    runs.append(f"mech_hl_gt_single_aoa_12_lr3e3_seed{s}")

out = {"task_dir": T, "runs": {}, "missing": [], "multi": {}}
for r in runs:
    d = os.path.join(E, r)
    if not os.path.isdir(d):
        out["missing"].append(r)
        continue
    files = glob.glob(os.path.join(d, "*", "metrics.jsonl")) + glob.glob(os.path.join(d, "metrics.jsonl"))
    if not files:
        out["missing"].append(r + " (no metrics.jsonl)")
        continue
    if len(files) > 1:
        out["multi"][r] = files
    f = files[0]
    rows = []
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            if j.get("phase") != "infer_step":
                continue
            sid = str(j.get("sample_id"))
            m = j.get("metrics", {})
            g = re.search(r"LHC\d+", sid)
            a = re.search(r"AoA_(\d+)", sid)
            rows.append({
                "sample_id": sid,
                "geometry": g.group(0) if g else None,
                "aoa": int(a.group(1)) if a else None,
                "pressure_l2": m.get("pressure_l2"),
                "velocity_l2": m.get("velocity_l2"),
                "tau_wall_l2": m.get("tau_wall_l2"),
            })
    out["runs"][r] = {"metrics_file": f, "n_rows": len(rows), "rows": rows}

dst = os.path.join(E, "audit_percase_rows.json")
with open(dst, "w") as fh:
    json.dump(out, fh)
print("wrote", dst)
print("missing", out["missing"])
print("multi", list(out["multi"].keys()))
for r, v in out["runs"].items():
    print(r, v["n_rows"])
