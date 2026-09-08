"""Deliverable 1: case- and geometry-level paired statistics, ISLA vs GeoTransolver / Transolver on HiLiftAeroML."""
import json, collections, math
import numpy as np
from scipy.stats import binomtest

import os, sys
R = os.path.dirname(os.path.abspath(__file__))
# argv[1]: per-case rows produced on the cluster by hilift_percase_rows_extract_remote_2026-09-08.py
# ($T/hl_evals/audit_percase_rows.json, scp'd back); the rows are embedded in the output artifact.
raw = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "audit_percase_rows.json")))
runs = raw["runs"]
rng = np.random.default_rng(20260908)
NBOOT = 10000

def rows(run):
    return {r["sample_id"]: r for r in runs[run]["rows"]}

comparisons = {
    "35": {"split": "super_scarce (35 training cases), validation = full_val (180 cases)",
           "isla": "lad_hl_mt2_super_scarce_seed{s}", "seeds": [42, 43, 44],
           "baselines": {"gt": "lad_hl_gt_super_scarce_seed{s}", "transolver": "t1_hl_transolver_super_scarce_lr3e3_seed{s}"}},
    "210": {"split": "scarce (210 training cases), validation = full_val (180 cases)",
            "isla": "lad_hl_mt2_scarce_seed{s}", "seeds": [42, 43, 44],
            "baselines": {"gt": "lad_hl_gt_scarce_seed{s}", "transolver": "t1_hl_transolver_scarce_lr3e3_seed{s}"}},
    "1260": {"split": "full (1,260 training cases), validation = full_val (180 cases)",
             "isla": "mt2_hl_lr1_seed{s}", "seeds": [42, 43],
             "baselines": {"gt": "gt_hl_lr1_seed{s}"}},
    "geo40": {"split": "geometry_super_scarce (4 training geometries, 40 cases), validation = geometry_val (18 unseen geometries x 10 angles)",
              "isla": "geo_hl_mt2_geometry_super_scarce_seed{s}", "seeds": [42, 43],
              "baselines": {"gt": "geo_hl_gt_geometry_super_scarce_seed{s}"}},
    "geo210": {"split": "geometry_scarce (21 training geometries, 210 cases), validation = geometry_val (18 unseen geometries x 10 angles)",
               "isla": "geo_hl_mt2_geometry_scarce_seed{s}", "seeds": [42, 43],
               "baselines": {"gt": "geo_hl_gt_geometry_scarce_seed{s}"}},
    "single12": {"split": "single_aoa_12 (126 training cases at AoA 12), validation = single_aoa_12_val (18 unseen geometries, one angle each)",
                 "isla": "mech_hl_mt2_single_aoa_12_seed{s}", "seeds": [42, 43],
                 "baselines": {"gt": "mech_hl_gt_single_aoa_12_seed{s}",
                               "transolver": "mech_hl_transolver_single_aoa_12_lr3e3_seed{s}",
                               "gt_lr3e3_supplementary": "mech_hl_gt_single_aoa_12_lr3e3_seed{s}"}},
}

def sign_test(wins, n):
    return binomtest(wins, n, 0.5, alternative="two-sided").pvalue

def analyze(isla_tmpl, base_tmpl, seeds, metric="pressure_l2"):
    I = {s: rows(isla_tmpl.format(s=s)) for s in seeds}
    B = {s: rows(base_tmpl.format(s=s)) for s in seeds}
    ids = sorted(set.intersection(*[set(v) for v in I.values()], *[set(v) for v in B.values()]))
    per_case_rows = []
    isla_case = {}; base_case = {}
    geo_of = {}; aoa_of = {}
    for cid in ids:
        g = I[seeds[0]][cid]["geometry"]; a = I[seeds[0]][cid]["aoa"]
        geo_of[cid] = g; aoa_of[cid] = a
        iv = [I[s][cid][metric] for s in seeds]; bv = [B[s][cid][metric] for s in seeds]
        isla_case[cid] = float(np.mean(iv)); base_case[cid] = float(np.mean(bv))
        per_case_rows.append({"case": cid, "geometry": g, "aoa": a,
                              "isla_per_seed": iv, "baseline_per_seed": bv,
                              "isla_seed_mean": isla_case[cid], "baseline_seed_mean": base_case[cid],
                              "ratio_baseline_over_isla": base_case[cid] / isla_case[cid]})
    n = len(ids)
    ic = np.array([isla_case[c] for c in ids]); bc = np.array([base_case[c] for c in ids])
    wins = int((ic < bc).sum())
    both = int(all_seed_wins(I, B, ids, seeds, metric))
    ratios = bc / ic
    case_level = {"n_cases": n, "isla_wins_seed_mean": wins, "isla_wins_every_seed": both,
                  "win_fraction": wins / n, "sign_test_p_two_sided": sign_test(wins, n),
                  "median_ratio_baseline_over_isla": float(np.median(ratios)),
                  "q10_ratio": float(np.quantile(ratios, 0.1)), "q90_ratio": float(np.quantile(ratios, 0.9)),
                  "mean_isla": float(ic.mean()), "mean_baseline": float(bc.mean()),
                  "ratio_of_means_baseline_over_isla": float(bc.mean() / ic.mean())}
    # case-level bootstrap of ratio of means
    idx = rng.integers(0, n, size=(NBOOT, n))
    rb = bc[idx].mean(axis=1) / ic[idx].mean(axis=1)
    case_level["bootstrap_ratio_of_means_95"] = [float(np.quantile(rb, 0.025)), float(np.quantile(rb, 0.975))]
    case_level["bootstrap_resamples"] = NBOOT
    # geometry level
    geos = sorted(set(geo_of.values()))
    G = {g: [c for c in ids if geo_of[c] == g] for g in geos}
    ig = np.array([np.mean([isla_case[c] for c in G[g]]) for g in geos])
    bg = np.array([np.mean([base_case[c] for c in G[g]]) for g in geos])
    ng = len(geos)
    gwins = int((ig < bg).sum())
    gr = bg / ig
    angles_per_geo = collections.Counter(len(G[g]) for g in geos)
    geometry_level = {"n_geometries": ng, "angles_per_geometry_histogram": {str(k): v for k, v in sorted(angles_per_geo.items())},
                      "aggregation": "per geometry, mean over its validation angles of the seed-mean case error, per arm",
                      "isla_wins": gwins, "win_fraction": gwins / ng, "sign_test_p_two_sided": sign_test(gwins, ng),
                      "median_ratio_baseline_over_isla": float(np.median(gr)),
                      "q10_ratio": float(np.quantile(gr, 0.1)), "q90_ratio": float(np.quantile(gr, 0.9)),
                      "ratio_of_means_baseline_over_isla": float(bg.mean() / ig.mean()),
                      "per_geometry": [{"geometry": g, "n_angles": len(G[g]), "isla_mean": float(ig[k]), "baseline_mean": float(bg[k]),
                                        "ratio": float(gr[k])} for k, g in enumerate(geos)]}
    # two-stage bootstrap: resample geometries, then angles within geometry (when >1 angle)
    # ratio of means = mean over resampled cases of baseline / mean of isla (case-weighted, as in the headline metric)
    case_i = {g: np.array([isla_case[c] for c in G[g]]) for g in geos}
    case_b = {g: np.array([base_case[c] for c in G[g]]) for g in geos}
    rb2 = np.empty(NBOOT)
    for k in range(NBOOT):
        gsel = rng.integers(0, ng, size=ng)
        si = 0.0; sb = 0.0; cnt = 0
        for gi in gsel:
            g = geos[gi]; m = len(G[g])
            if m > 1:
                asel = rng.integers(0, m, size=m)
                si += case_i[g][asel].sum(); sb += case_b[g][asel].sum()
            else:
                si += case_i[g].sum(); sb += case_b[g].sum()
            cnt += m
        rb2[k] = (sb / cnt) / (si / cnt)
    geometry_level["two_stage_bootstrap_ratio_of_means_95"] = [float(np.quantile(rb2, 0.025)), float(np.quantile(rb2, 0.975))]
    geometry_level["two_stage_bootstrap_note"] = "10,000 resamples; stage 1 resamples geometries with replacement, stage 2 resamples each drawn geometry's angle set with replacement (skipped when a geometry has one angle); statistic = case-weighted mean baseline error / mean ISLA error"
    # geometry-only bootstrap (cluster bootstrap, no within-geometry resampling)
    idxg = rng.integers(0, ng, size=(NBOOT, ng))
    wts = np.array([len(G[g]) for g in geos], dtype=float)
    num = (bg * wts)[idxg].sum(axis=1); den = (ig * wts)[idxg].sum(axis=1)
    geometry_level["cluster_bootstrap_ratio_of_means_95"] = [float(np.quantile(num / den, 0.025)), float(np.quantile(num / den, 0.975))]
    return per_case_rows, case_level, geometry_level

def all_seed_wins(I, B, ids, seeds, metric):
    return sum(all(I[s][c][metric] < B[s][c][metric] for s in seeds) for c in ids)

out = {"date": "2026-09-08", "metric": "pressure_l2 (relative L2 of surface pressure, per case)",
       "source": "per-case rows extracted from $T/hl_evals/<run>/<run>/metrics.jsonl (phase == infer_step) on the cluster; $T = " + raw["task_dir"],
       "missing_runs": raw["missing"],
       "seed_mean_note": "case error for an arm = mean over seeds of that seed's per-case pressure_l2 (no ensemble predictions exist); 'isla_wins_every_seed' counts cases where ISLA's error is lower for every seed pairing seed-to-seed",
       "sign_test": "two-sided exact binomial test of wins against 0.5, at the case level (n = cases) and at the geometry level (n = validation geometries)",
       "comparisons": {}}
for key, spec in comparisons.items():
    C = {"split": spec["split"], "isla_runs": [spec["isla"].format(s=s) for s in spec["seeds"]], "seeds": spec["seeds"], "baselines": {}}
    for bname, btmpl in spec["baselines"].items():
        pcr, cl, gl = analyze(spec["isla"], btmpl, spec["seeds"])
        entry = {"baseline_runs": [btmpl.format(s=s) for s in spec["seeds"]], "case_level": cl, "geometry_level": gl, "per_case_rows": pcr}
        if len(spec["seeds"]) > 2:
            _, cl2, gl2 = analyze(spec["isla"], btmpl, [42, 43])
            gl2.pop("per_geometry", None)
            entry["two_seed_42_43_for_continuity"] = {"case_level": cl2, "geometry_level": gl2}
        C["baselines"][bname] = entry
    out["comparisons"][key] = C

# geometry counts per validation split from rows
out["validation_geometry_counts"] = {k: out["comparisons"][k]["baselines"]["gt"]["geometry_level"]["n_geometries"] for k in comparisons}
json.dump(out, open(f"{R}/hilift_paired_stats_geometry_2026-09-08.json", "w"), indent=1)

for key, C in out["comparisons"].items():
    for bname, e in C["baselines"].items():
        cl = e["case_level"]; gl = e["geometry_level"]
        print(f"{key} isla vs {bname} (seeds {C['seeds']}): cases n={cl['n_cases']} wins={cl['isla_wins_seed_mean']} every-seed={cl['isla_wins_every_seed']} p={cl['sign_test_p_two_sided']:.2e} "
              f"median ratio {cl['median_ratio_baseline_over_isla']:.3f} [{cl['q10_ratio']:.3f},{cl['q90_ratio']:.3f}] ratio-of-means {cl['ratio_of_means_baseline_over_isla']:.3f} case-boot {cl['bootstrap_ratio_of_means_95']}")
        print(f"    geometries n={gl['n_geometries']} angles/geo {gl['angles_per_geometry_histogram']} wins={gl['isla_wins']} p={gl['sign_test_p_two_sided']:.2e} median ratio {gl['median_ratio_baseline_over_isla']:.3f} [{gl['q10_ratio']:.3f},{gl['q90_ratio']:.3f}] "
              f"two-stage boot {gl['two_stage_bootstrap_ratio_of_means_95']} cluster boot {gl['cluster_bootstrap_ratio_of_means_95']}")
        if "two_seed_42_43_for_continuity" in e:
            c2 = e["two_seed_42_43_for_continuity"]["case_level"]; g2 = e["two_seed_42_43_for_continuity"]["geometry_level"]
            print(f"    [seeds 42,43] case wins={c2['isla_wins_seed_mean']} p={c2['sign_test_p_two_sided']:.2e} median {c2['median_ratio_baseline_over_isla']:.3f}; geo wins={g2['isla_wins']} p={g2['sign_test_p_two_sided']:.2e} two-stage {g2['two_stage_bootstrap_ratio_of_means_95']}")
print(out["validation_geometry_counts"])
