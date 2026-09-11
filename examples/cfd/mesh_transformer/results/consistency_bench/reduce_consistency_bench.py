#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reduce the consistency benchmark (2026-09-10) evaluation tree to one JSON.

Question: do ISLA (MeshTransformer2), GeoTransolver and Transolver give the
same field when the SAME validation surface is sampled differently (uniform
over cells / area-proportional / 10:1 front-back biased), and do they
converge to one field as the sample count grows?

Input tree (written by $T/bench/consistency_bench_aga.sbatch, float32):
    <evals_root>/<run>/<sampler>_<count>/<run>/metrics.jsonl   (infer.py rows)
    <evals_root>/<run>/<sampler>_<count>/.done                  (lane marker)
    <evals_root>/<run>/<sampler>_<count>.log                    (lane log)
Runs are `<arm>_seed<S>`; two seeds per arm. Per arm x sampler x count the
error is the mean over cases of the seed-mean per case (cases paired by
sample_id, which embeds the validation index).

Preregistered readouts (implemented exactly as registered):
  1. D_area(n)   = err(area,n)/err(unif,n) - 1 and
     D_biased(n) = err(biased,n)/err(unif,n) - 1 at every n; headline n=40k.
     Bars: |D| <= 3% "consistent"; >= 10% "reads the sampling distribution";
     between: "inconclusive".
  2. Convergence: err(unif,n), err(area,n) vs n; err(2.5k)/err(40k);
     monotone within the two-seed spread; fine-mesh gap
     |err(area,40k) - err(unif,40k)|.
  3. Per-case paired counts (area < unif etc.) with an exact two-sided
     sign-test p.
Checks: two seeds must differ (identical errors = weights did not load);
10k-unif reproduces the campaign-E float32 reference per run; lane logs
must not carry the checkpoint-loader warnings.

Usage (login node, stdlib only):
    python3 reduce_consistency_bench.py --evals-root $T/bench_evals \
        --lanes $T/bench/bench_lanes.tsv --walltimes $T/bench/lane_walltimes.log \
        --campaign-e-root $T/transfer/campaign_e_fp32 \
        --out consistency_bench_2026-09-10.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

COUNTS = (2500, 5000, 10000, 20000, 40000)
SAMPLERS = ("unif", "area", "biased")
CONSISTENT_BAR = 0.03
READS_BAR = 0.10
REPRO_TOL = 0.002  # 10k-unif vs campaign-E float32 (a different uniform draw)
LOAD_FAILURE_RE = re.compile(r"skipping load|Could not find valid model file")

PREREGISTERED_PREDICTIONS = [
    "ISLA with measure weights (constant or similarity gauge): |D_area| <= 3% at all n on both datasets.",
    "ISLA similarity gauge: D_biased <= 30%.",
    "ISLA constant gauge fails D_biased (known 12x at 10k) but passes D_area.",
    "GeoTransolver and Transolver (no measure): |D_area| >= 10% on HiLift (heterogeneous multi-element meshes), smaller on DrivAerML; D_biased >= 200% (known 9.5x / 3.9x).",
    "ISLA weights-off ablation: fails D_area on HiLift.",
]

# Arm labels: run-name prefix (seed stripped) -> (architecture, measure handling)
ARM_INFO = {
    "iw_mt2_lr1e3": ("ISLA (MeshTransformer2), constant gauge, measure weights on", "drivaer_ml_surface"),
    "iw_mt2_gauge": ("ISLA (MeshTransformer2), similarity gauge, measure weights on", "drivaer_ml_surface"),
    "now_mt2_lr1e3": ("ISLA (MeshTransformer2), weights-off ablation", "drivaer_ml_surface"),
    "uw_gt_unit_lr1e3": ("GeoTransolver, unit drive, no measure", "drivaer_ml_surface"),
    "uw_transolver_unit_lr3e3": ("Transolver, unit drive, no measure", "drivaer_ml_surface"),
    "lad_hl_mt2_super_scarce": ("ISLA (MeshTransformer2), constant gauge, measure weights on", "highlift_surface_super_scarce"),
    "a35_hl_mt2_noweights": ("ISLA (MeshTransformer2), weights-off ablation (stand-in for now_hl_mt2_super_scarce)", "highlift_surface_super_scarce"),
    "udrv_hl_gt_super_scarce_lr3e3": ("GeoTransolver, unit drive, no measure", "highlift_surface_super_scarce"),
    "udrv_hl_transolver_super_scarce_lr3e3": ("Transolver, unit drive, no measure", "highlift_surface_super_scarce"),
    "mech_hl_mt2_single_aoa_12": ("ISLA (MeshTransformer2), constant gauge, measure weights on", "highlift_surface_single_aoa_12"),
    "now_hl_mt2_single_aoa_12": ("ISLA (MeshTransformer2), weights-off ablation", "highlift_surface_single_aoa_12"),
    "udrv_hl_gt_single_aoa_12": ("GeoTransolver, unit drive, no measure", "highlift_surface_single_aoa_12"),
    "udrv_hl_transolver_single_aoa_12_lr3e3": ("Transolver, unit drive, no measure", "highlift_surface_single_aoa_12"),
    # Follow-up wave (coordinator, 2026-09-10): measure-centering checkpoints, counts 10k / 40k only.
    "cg_dr_mt2_mcenter": ("ISLA (MeshTransformer2), constant gauge, measure weights on, center_mode=measure", "drivaer_ml_surface"),
    "cg_hl_mt2_mcenter_super_scarce": ("ISLA (MeshTransformer2), constant gauge, measure weights on, center_mode=measure", "highlift_surface_super_scarce"),
    # Second follow-up wave: measure-weighted baselines (snapshot code_mw, *_mw model yamls), counts 10k / 40k only.
    "mw_dr_gt_unit_lr1e3": ("GeoTransolver, unit drive, measure-weighted pooling", "drivaer_ml_surface"),
    "mw_dr_transolver_unit_lr3e3": ("Transolver, unit drive, measure-weighted pooling", "drivaer_ml_surface"),
    "mw_hl_gt_super_scarce_lr3e3": ("GeoTransolver, unit drive, measure-weighted pooling", "highlift_surface_super_scarce"),
    "mw_hl_transolver_super_scarce_lr3e3": ("Transolver, unit drive, measure-weighted pooling", "highlift_surface_super_scarce"),
}


def arm_of(run: str) -> tuple[str, int]:
    m = re.fullmatch(r"(.*)_seed(\d+)", run)
    if not m:
        raise ValueError(f"run {run!r} has no _seed<S> suffix")
    return m.group(1), int(m.group(2))


def read_lanes(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            if not line.strip():
                continue
            rows.append(dict(zip(header, line.rstrip("\n").split("\t"))))
    for r in rows:
        r["count"] = int(r["count"])
        r["idx"] = int(r["idx"])
    return rows


def read_lane(evals_root: Path, run: str, sampler: str, count: int) -> dict:
    """Parse one lane: per-case metrics, summary, status."""
    lane_dir = evals_root / run / f"{sampler}_{count}"
    log = evals_root / run / f"{sampler}_{count}.log"
    metrics = lane_dir / run / "metrics.jsonl"
    out = {
        "status": "missing",
        "done_marker": (lane_dir / ".done").exists(),
        "cases": {},
        "summary": None,
        "load_failure": False,
        "n_points": [],
        "frac_at_min_weight": [],
        "weight_ratio": [],
    }
    if log.exists() and LOAD_FAILURE_RE.search(log.read_text(errors="replace")):
        out["load_failure"] = True
    if not metrics.exists():
        if log.exists():
            out["status"] = "failed" if "FAILED" in log.read_text(errors="replace") else "running"
        return out
    with metrics.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("phase") == "infer_step":
                out["cases"][rec["sample_id"]] = rec["metrics"]
                if "n_points" in rec:
                    out["n_points"].append(int(rec["n_points"]))
                mw = rec.get("measure_weights")
                if mw is not None:
                    out["frac_at_min_weight"].append(float(mw["frac_at_min"]))
                    out["weight_ratio"].append(float(mw["max"]) / float(mw["min"]))
            elif rec.get("phase") == "infer_summary":
                out["summary"] = rec
    if out["load_failure"]:
        out["status"] = "load_failure"
    elif out["summary"] is not None and out["done_marker"]:
        out["status"] = "done"
    elif out["summary"] is not None:
        out["status"] = "done_no_marker"
    else:
        out["status"] = "partial"
    return out


def l2_fields(metrics: dict) -> list[str]:
    """Aggregate relative-L2 keys (drop per-component _x/_y/_z keys)."""
    return sorted(
        k for k in metrics if k.endswith("_l2") and not re.search(r"_[xyz]_l2$", k)
    )


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sign_test_p(n_less: int, n_greater: int) -> float:
    """Exact two-sided sign test on the non-tied pairs."""
    n = n_less + n_greater
    if n == 0:
        return float("nan")
    k = min(n_less, n_greater)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def verdict(d: float) -> str:
    if d != d:  # nan
        return "n/a"
    if abs(d) <= CONSISTENT_BAR:
        return "consistent"
    if abs(d) >= READS_BAR:
        return "reads the sampling distribution"
    return "inconclusive"


def reduce_arm(arm: str, runs: dict[int, str], lanes: dict) -> dict:
    """lanes: (run, sampler, count) -> parsed lane."""
    info = ARM_INFO.get(arm, ("unknown", "unknown"))
    result = {
        "label": info[0],
        "dataset": info[1],
        "runs": {str(s): r for s, r in sorted(runs.items())},
        "fields": [],
        "table": {},
        "D": {"area": {}, "biased": {}},
        "verdicts": {"area": {}, "biased": {}},
        "convergence": {},
        "fine_mesh_gap_40k": {},
        "paired": {"area_vs_unif": {}, "biased_vs_unif": {}},
        "checks": {"seeds_differ": True, "identical_seed_pairs": [], "load_failures": []},
        "lane_status": {},
    }
    # per (sampler, count): per-case seed-mean, per-seed means
    percase: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    for sampler in SAMPLERS:
        result["table"][sampler] = {}
        for count in COUNTS:
            parsed = {s: lanes.get((r, sampler, count)) for s, r in runs.items()}
            result["lane_status"][f"{sampler}_{count}"] = {
                runs[s]: (p["status"] if p else "missing") for s, p in parsed.items()
            }
            for s, p in parsed.items():
                if p and p["load_failure"]:
                    result["checks"]["load_failures"].append(f"{runs[s]}/{sampler}_{count}")
            usable = {s: p for s, p in parsed.items() if p and p["status"] in ("done", "done_no_marker")}
            if not usable:
                continue
            fields = sorted(set().union(*(l2_fields(next(iter(p["cases"].values()))) for p in usable.values() if p["cases"])))
            for f in fields:
                if f not in result["fields"]:
                    result["fields"].append(f)
            # cases present in every usable seed
            common = set.intersection(*(set(p["cases"]) for p in usable.values()))
            entry: dict = {"n_seeds": len(usable), "n_cases": len(common)}
            pc: dict[str, dict[str, float]] = {f: {} for f in fields}
            for f in fields:
                seed_means = {}
                for s, p in usable.items():
                    seed_means[runs[s]] = mean(p["cases"][c][f] for c in common)
                for c in sorted(common):
                    pc[f][c] = mean(p["cases"][c][f] for p in usable.values())
                entry[f] = {"mean": mean(pc[f].values()), "seeds": seed_means}
                vals = list(seed_means.values())
                if len(vals) == 2 and abs(vals[0] - vals[1]) < 1e-9:
                    result["checks"]["seeds_differ"] = False
                    result["checks"]["identical_seed_pairs"].append(f"{sampler}_{count}:{f}")
            npts = [n for p in usable.values() for n in p["n_points"]]
            if npts:
                entry["n_points"] = {"mean": mean(npts), "min": min(npts), "max": max(npts)}
            fam = [x for p in usable.values() for x in p["frac_at_min_weight"]]
            if fam:
                ### area mode: fraction of kept cells clamped to pi=1; biased
                ### mode: fraction of kept cells in the front (10x) half.
                entry["frac_at_min_weight_mean"] = mean(fam)
            wr = [x for p in usable.values() for x in p["weight_ratio"]]
            if wr:
                entry["ht_weight_max_over_min_mean"] = mean(wr)
            result["table"][sampler][str(count)] = entry
            percase[(sampler, count)] = pc

    fields = result["fields"]

    def err(sampler, count, f):
        e = result["table"].get(sampler, {}).get(str(count))
        return e[f]["mean"] if e and f in e else float("nan")

    # Readout 1
    for alt in ("area", "biased"):
        for count in COUNTS:
            dd = {}
            for f in fields:
                u, a = err("unif", count, f), err(alt, count, f)
                dd[f] = a / u - 1.0 if (u == u and a == a and u > 0) else float("nan")
            result["D"][alt][str(count)] = dd
            result["verdicts"][alt][str(count)] = {f: verdict(dd[f]) for f in fields}
    result["headline_40k"] = {
        alt: {
            f: {"D": result["D"][alt].get("40000", {}).get(f, float("nan")),
                "verdict": result["verdicts"][alt].get("40000", {}).get(f, "n/a")}
            for f in fields
        }
        for alt in ("area", "biased")
    }

    # Readout 2
    for sampler in ("unif", "area"):
        result["convergence"][sampler] = {}
        for f in fields:
            curve = {str(c): err(sampler, c, f) for c in COUNTS}
            vals = [curve[str(c)] for c in COUNTS]
            have = all(v == v for v in vals)
            spread = {}
            for c in COUNTS:
                e = result["table"].get(sampler, {}).get(str(c))
                sv = list(e[f]["seeds"].values()) if e and f in e else []
                spread[c] = (max(sv) - min(sv)) / 2.0 if len(sv) >= 2 else 0.0
            strict = have and all(vals[i + 1] <= vals[i] for i in range(len(vals) - 1))
            within = have and all(
                vals[i + 1] <= vals[i] + max(spread[COUNTS[i]], spread[COUNTS[i + 1]])
                for i in range(len(vals) - 1)
            )
            result["convergence"][sampler][f] = {
                "curve": curve,
                "seed_half_range": {str(c): spread[c] for c in COUNTS},
                "ratio_2500_over_40000": (vals[0] / vals[-1]) if have and vals[-1] > 0 else float("nan"),
                "monotone_strict": strict if have else None,
                "monotone_within_seed_spread": within if have else None,
            }
    for f in fields:
        u, a = err("unif", 40000, f), err("area", 40000, f)
        result["fine_mesh_gap_40k"][f] = abs(a - u) if (u == u and a == a) else float("nan")

    # Readout 3
    for alt, key in (("area", "area_vs_unif"), ("biased", "biased_vs_unif")):
        for count in COUNTS:
            pu, pa = percase.get(("unif", count)), percase.get((alt, count))
            if not pu or not pa:
                continue
            result["paired"][key][str(count)] = {}
            for f in fields:
                if f not in pu or f not in pa:
                    continue
                common = sorted(set(pu[f]) & set(pa[f]))
                n_less = sum(1 for c in common if pa[f][c] < pu[f][c])
                n_greater = sum(1 for c in common if pa[f][c] > pu[f][c])
                n_tie = len(common) - n_less - n_greater
                result["paired"][key][str(count)][f] = {
                    "n_cases": len(common),
                    f"n_{alt}_less_than_unif": n_less,
                    f"n_{alt}_greater_than_unif": n_greater,
                    "n_tie": n_tie,
                    "sign_test_p": sign_test_p(n_less, n_greater),
                }
    return result


def campaign_e_reference(root: Path, runs: list[str]) -> dict[str, dict]:
    """Campaign-E float32 10k-uniform reference (pressure_l2 / wss_l2) per run."""
    ref = {}
    for run in runs:
        f = root / run / "unif" / run / "metrics.jsonl"
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("phase") == "infer_summary":
                ref[run] = {k: v for k, v in rec["metrics"].items() if k in ("pressure_l2", "wss_l2")}
    return ref


def read_walltimes(path: Path | None) -> dict:
    out = {"lanes": {}, "lane_gpu_hours": 0.0, "allocation_gpu_hours": 0.0, "by_count_mean_seconds": {}}
    if path is None or not path.exists():
        return out
    per_task: dict[str, int] = defaultdict(int)
    by_count: dict[str, list[int]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.startswith("LANE-WALL"):
            continue
        kv = dict(tok.split("=", 1) for tok in line.split()[1:])
        key = f"{kv['run']}/{kv['sampler']}_{kv['count']}"
        secs = int(kv["seconds"])
        out["lanes"][key] = {"seconds": secs, "rc": int(kv["rc"]), "job": kv.get("job")}
        out["lane_gpu_hours"] += secs / 3600.0
        if kv.get("job"):
            per_task[kv["job"]] = max(per_task[kv["job"]], secs)
        by_count[f"{kv['run'].split('_seed')[0]}@{kv['count']}"].append(secs)
    # Whole-node allocations: each array task holds 4 GPUs for its longest lane.
    out["allocation_gpu_hours"] = sum(per_task.values()) * 4 / 3600.0
    out["by_count_mean_seconds"] = {k: mean(v) for k, v in sorted(by_count.items())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals-root", type=Path, required=True)
    ap.add_argument("--lanes", type=Path, required=True)
    ap.add_argument("--walltimes", type=Path, default=None)
    ap.add_argument("--campaign-e-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path(f"consistency_bench_{date.today().isoformat()}.json"))
    ap.add_argument("--note", action="append", default=[], help="free-text note recorded under 'notes' (repeatable)")
    args = ap.parse_args()

    lane_rows = read_lanes(args.lanes)
    lanes = {}
    status_counts: dict[str, int] = defaultdict(int)
    failed, missing = [], []
    for r in lane_rows:
        p = read_lane(args.evals_root, r["run"], r["sampler"], r["count"])
        lanes[(r["run"], r["sampler"], r["count"])] = p
        status_counts[p["status"]] += 1
        key = f"{r['run']}/{r['sampler']}_{r['count']}"
        if p["status"] in ("failed", "load_failure"):
            failed.append(key)
        elif p["status"] in ("missing", "running", "partial"):
            missing.append(key)

    arms: dict[str, dict[int, str]] = defaultdict(dict)
    snapshots: dict[str, set[str]] = defaultdict(set)
    for r in lane_rows:
        arm, seed = arm_of(r["run"])
        arms[arm][seed] = r["run"]
        snapshots[arm].add(r["snapshot"])
    reduced = {arm: reduce_arm(arm, runs, lanes) for arm, runs in arms.items()}
    for arm, res in reduced.items():
        res["code_snapshot"] = sorted(snapshots[arm])

    ref = campaign_e_reference(args.campaign_e_root, [r for a in arms.values() for r in a.values()]) if args.campaign_e_root else {}
    for arm, res in reduced.items():
        repro = {}
        for seed, run in arms[arm].items():
            p = lanes.get((run, "unif", 10000))
            if not p or p["status"] not in ("done", "done_no_marker") or run not in ref:
                continue
            bench = mean(p["cases"][c]["pressure_l2"] for c in p["cases"])
            r = ref[run]["pressure_l2"]
            repro[run] = {"bench_10k_unif": bench, "campaign_e_reference": r, "diff": bench - r, "ok": abs(bench - r) <= REPRO_TOL}
        res["checks"]["reproduction_10k_unif_pressure_l2"] = repro

    out = {
        "benchmark": "consistency_bench",
        "date": date.today().isoformat(),
        "precision": "float32",
        "evals_root": str(args.evals_root),
        "counts": list(COUNTS),
        "samplers": list(SAMPLERS),
        "bars": {"consistent_abs_D_le": CONSISTENT_BAR, "reads_abs_D_ge": READS_BAR},
        "preregistered_predictions": PREREGISTERED_PREDICTIONS,
        "notes": args.note,
        "lanes_total": len(lane_rows),
        "lane_status_counts": dict(status_counts),
        "lanes_failed": failed,
        "lanes_missing_or_incomplete": missing,
        "arms": reduced,
        "walltimes": read_walltimes(args.walltimes),
    }
    args.out.write_text(json.dumps(out, indent=1, allow_nan=True))
    # Console table: pressure_l2 per arm x sampler x count
    print(f"lanes: {dict(status_counts)}  -> {args.out}")
    for arm, res in reduced.items():
        f = "pressure_l2"
        print(f"\n== {arm}  [{res['label']}; {res['dataset']}]")
        print("sampler   " + "".join(f"{c:>10d}" for c in COUNTS))
        for s in SAMPLERS:
            row = [res["table"].get(s, {}).get(str(c), {}).get(f, {}).get("mean", float("nan")) for c in COUNTS]
            print(f"{s:<10}" + "".join(f"{v:>10.4f}" for v in row))
        for alt in ("area", "biased"):
            row = [res["D"][alt].get(str(c), {}).get(f, float("nan")) for c in COUNTS]
            print(f"D_{alt:<8}" + "".join(f"{100 * v:>+9.1f}%" for v in row) + f"   40k: {res['verdicts'][alt].get('40000', {}).get(f, 'n/a')}")
        conv = res["convergence"].get("unif", {}).get(f, {})
        if conv:
            print(f"unif err(2.5k)/err(40k) = {conv['ratio_2500_over_40000']:.3f}  monotone(strict/within-spread) = {conv['monotone_strict']}/{conv['monotone_within_seed_spread']}")
        if not res["checks"]["seeds_differ"]:
            print(f"!! identical seed pairs: {res['checks']['identical_seed_pairs']}")
        if res["checks"]["load_failures"]:
            print(f"!! checkpoint load failures: {res['checks']['load_failures']}")
        for run, rp in res["checks"].get("reproduction_10k_unif_pressure_l2", {}).items():
            print(f"10k repro {run}: bench {rp['bench_10k_unif']:.5f} vs ref {rp['campaign_e_reference']:.5f} ({'ok' if rp['ok'] else 'MISMATCH'})")


if __name__ == "__main__":
    main()
