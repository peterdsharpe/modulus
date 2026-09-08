"""Deliverable 4: per-band excess SSE of ISLA passive-SDF interior over GeoTransolver-volume (DrivAerML V0)."""
import json
import os
R = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(f"{R}/v0_sdf_band_2026-09-05.json"))
scheme = "gstrat_nondim_unit"
bd = d["band_definitions"][scheme]
names = bd["names"]; thr = bd["sdf_thresholds_nondim"]
L_ref_m = d["global_data_first_case"]["L_ref"][0]  # 5.0 m per nondim unit
edges_m = [0.0] + [t * L_ref_m for t in thr] + [None]
band_m = {n: {"nondim_lo": ([0.0] + thr)[i], "nondim_hi": (thr + [None])[i],
              "metres_lo": edges_m[i], "metres_hi": edges_m[i + 1]} for i, n in enumerate(names)}
pcp = d["per_case_parts"]
arms = {"gt_vol": ["v0_gt_vol_seed42", "v0_gt_vol_seed43"],
        "mt2_int_sdf": ["v0_mt2_int_sdf_seed42", "v0_mt2_int_sdf_seed43"]}
# probe the triple structure: [numerator, denominator?, count]
ex = pcp["v0_gt_vol_seed42"][scheme][names[0]]["pressure"][0]
out = {"date": "2026-09-08", "source": "results/v0_sdf_band_2026-09-05.json (per_case_parts)", "scheme": scheme,
       "band_definitions": {"thresholds_nondim": thr, "L_ref_m_per_nondim_unit": L_ref_m, "bands": band_m,
                            "note": bd["note"]},
       "per_case_parts_triple_meaning": "each per-case entry is [sum of squared error over band points, sum of squared target over band points, n points]; inferred from the file (num, den, count) and checked below against the file's pooled per-band rel-L2",
       "example_triple": ex,
       "arms": {k: v for k, v in arms.items()}, "fields": {}}
for field in ["pressure", "velocity", "velocity_pert"]:
    F = {"per_arm": {}, "bands": {}}
    tot = {}
    for arm, runs in arms.items():
        per_band = {}
        for b in names:
            sse = 0.0; sst = 0.0; n = 0; ncases = 0
            for r in runs:
                for trip in pcp[r][scheme][b][field]:
                    sse += trip[0]; sst += trip[1]; n += int(trip[2]); ncases += 1
            per_band[b] = {"sse_sum_over_seeds": sse, "sst_sum_over_seeds": sst, "n_points_sum_over_seeds": n,
                           "n_case_entries": ncases, "rms_error": (sse / n) ** 0.5 if n else None,
                           "rms_target": (sst / n) ** 0.5 if n else None,
                           "pooled_rel_l2": (sse / sst) ** 0.5 if sst else None}
        total_sse = sum(v["sse_sum_over_seeds"] for v in per_band.values())
        total_n = sum(v["n_points_sum_over_seeds"] for v in per_band.values())
        for b in names:
            per_band[b]["share_of_arm_sse"] = per_band[b]["sse_sum_over_seeds"] / total_sse
            per_band[b]["share_of_points"] = per_band[b]["n_points_sum_over_seeds"] / total_n
        F["per_arm"][arm] = {"per_band": per_band, "total_sse": total_sse, "total_points": total_n}
    g = F["per_arm"]["gt_vol"]["per_band"]; m = F["per_arm"]["mt2_int_sdf"]["per_band"]
    excess = {b: m[b]["sse_sum_over_seeds"] - g[b]["sse_sum_over_seeds"] for b in names}
    tot_ex = sum(excess.values())
    cum = 0.0
    for b in names:
        cum += excess[b]
        F["bands"][b] = {"metres": [band_m[b]["metres_lo"], band_m[b]["metres_hi"]],
                         "share_of_points": g[b]["share_of_points"],
                         "gt_sse": g[b]["sse_sum_over_seeds"], "isla_sdf_sse": m[b]["sse_sum_over_seeds"],
                         "excess_sse_isla_minus_gt": excess[b], "excess_fraction": excess[b] / tot_ex,
                         "excess_fraction_cumulative": cum / tot_ex,
                         "gt_rms_error": g[b]["rms_error"], "isla_sdf_rms_error": m[b]["rms_error"],
                         "rms_target": g[b]["rms_target"],
                         "rms_ratio_isla_over_gt": m[b]["rms_error"] / g[b]["rms_error"],
                         "gt_pooled_rel_l2": g[b]["pooled_rel_l2"], "isla_sdf_pooled_rel_l2": m[b]["pooled_rel_l2"]}
    F["total_excess_sse"] = tot_ex
    F["total_sse_ratio_isla_over_gt"] = F["per_arm"]["mt2_int_sdf"]["total_sse"] / F["per_arm"]["gt_vol"]["total_sse"]
    out["fields"][field] = F
# cross-check pooled rel-L2 vs file's per_run pooled numbers for one run/band
chk = {}
for r in ["v0_gt_vol_seed42", "v0_mt2_int_sdf_seed42"]:
    for b in names:
        parts = pcp[r][scheme][b]["pressure"]
        mine = (sum(p[0] for p in parts) / sum(p[1] for p in parts)) ** 0.5
        theirs = d["per_run"][r][scheme][b]["pressure"]["pooled"]
        chk[f"{r}/{b}"] = {"recomputed_pooled": mine, "file_pooled": theirs}
out["pooled_rel_l2_cross_check"] = chk
out["audit_claims"] = {"audit_reported_excess_below_first_band": {"pressure": 0.9458, "velocity": 0.9373},
                       "audit_reported_gt_99pct_within_2m": True,
                       "computed_excess_fraction_first_band": {f: out["fields"][f]["bands"][names[0]]["excess_fraction"] for f in ["pressure", "velocity"]},
                       "computed_cumulative_within_first_three_bands_lt_2.5m": {f: out["fields"][f]["bands"][names[2]]["excess_fraction_cumulative"] for f in ["pressure", "velocity"]},
                       "computed_cumulative_within_first_two_bands_lt_0.5m": {f: out["fields"][f]["bands"][names[1]]["excess_fraction_cumulative"] for f in ["pressure", "velocity"]}}
out["note"] = "Bands are in the recipe's nondimensional frame (1 unit = L_ref = 5.0 m for all DrivAerML cases), so the file's band edges 0.01/0.1/0.5 are 5 cm / 50 cm / 2.5 m physical, not 1 cm / 10 cm / 50 cm. SSE sums both seeds' 48 cases per arm; excess = ISLA passive-SDF minus GeoTransolver-volume."
json.dump(out, open(f"{R}/v0_interior_excess_sse_2026-09-08.json", "w"), indent=1)
print(json.dumps(out["band_definitions"]["bands"]))
print(json.dumps(out["audit_claims"], indent=1))
print(json.dumps(chk, indent=1)[:1500])
for f in ["pressure", "velocity"]:
    print(f, "ratio total", out["fields"][f]["total_sse_ratio_isla_over_gt"])
    for b in names:
        x = out["fields"][f]["bands"][b]
        print(" ", b, x["metres"], "pts %.4f" % x["share_of_points"], "excess %.4f cum %.4f" % (x["excess_fraction"], x["excess_fraction_cumulative"]), "rms ratio %.3f" % x["rms_ratio_isla_over_gt"], "gt rel %.4f isla rel %.4f" % (x["gt_pooled_rel_l2"], x["isla_sdf_pooled_rel_l2"]))
