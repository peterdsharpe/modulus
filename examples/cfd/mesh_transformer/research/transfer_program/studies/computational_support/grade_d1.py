"""Grade D1 (computational support with passive readout) and campaign D
(interior carriers) against their preregistered bars.

Input: the cluster reducer's output ``hl_evals/a35_v0_reduction.json``
(written by ``highlift/reduce_a35_v0.py``; per-run 48-car mean relative L2 of
interior pressure, velocity and eddy viscosity for ``v0_<arm>_seed{42,43}``).
Output: a markdown table on stdout and ``d1_campd_verdict_<date>.json`` beside
this file. Bars are copied verbatim from PREREG.md (D1) and the notebook's
campaign entry (campaign D); nothing here is tuned after the fact.

Usage: python grade_d1.py /path/to/a35_v0_reduction.json
"""
import json
import statistics
import sys
from datetime import date
from pathlib import Path

KEYS = ("pressure_l2", "velocity_l2", "nut_l2")
ARMS = {
    "isla_qtsdfval": "reference: ISLA query tokens + SDF, 10k interacting queries",
    "isla_support12": "SUPPORT-12: 5k interacting support + 5k passive queries, 12 read blocks",
    "isla_support4": "SUPPORT-4: same, 4 read blocks",
    "isla_passive12": "PASSIVE-12: no support, 5k passive queries, 12 read blocks",
    "isla_qtsdf_h344": "campaign D: ISLA QT+SDF, hidden 344 (27.6M params)",
    "gt_vol_nolocal": "campaign D: GeoTransolver-volume without local features (unit drive)",
    "gt_vol": "GeoTransolver-volume reference",
}
# D1 bars (PREREG.md, "Bars, in readout units"), absolute readout values.
D1_SUPPORTED = {"pressure_l2": 0.0584, "velocity_l2": 0.0879, "nut_l2": 0.1251}
D1_FALSIFIED_PRESSURE = 0.0624
# Campaign D bars (99-notebook.qmd, #sec-nb-campaign), eddy-viscosity readout.
D_GT_NOLOCAL_LOCAL_CARRIES = 0.103   # GT-volume nu_t >= this without local features -> local features carry the lead
D_H344_SUPPORTED = 0.103             # ISLA h344 nu_t <= this -> parameters carry
D_H344_FALSIFIED = 0.117


def load(path):
    red = json.loads(Path(path).read_text())["v0"]
    per_arm = {}
    for arm in ARMS:
        runs = {s: red[f"v0_{arm}_seed{s}"] for s in (42, 43) if f"v0_{arm}_seed{s}" in red}
        if runs:
            per_arm[arm] = {
                "seeds": sorted(runs),
                "per_seed": runs,
                "mean": {k: statistics.mean(r[k] for r in runs.values()) for k in KEYS if all(k in r for r in runs.values())},
            }
    return per_arm


def grade(per_arm):
    v = {"d1": {}, "campaign_d": {}}
    ref = per_arm.get("isla_qtsdfval", {}).get("mean")
    s12 = per_arm.get("isla_support12", {}).get("mean")
    if s12 and len(per_arm["isla_support12"]["seeds"]) == 2:
        within = {k: s12[k] <= D1_SUPPORTED[k] for k in KEYS}
        if s12["pressure_l2"] >= D1_FALSIFIED_PRESSURE:
            acc = "FALSIFIED"
        elif all(within.values()):
            acc = "SUPPORTED (accuracy half; sentinel study still required)"
        else:
            acc = "BETWEEN"
        v["d1"] = {"support12_mean": s12, "within_bar": within, "accuracy_verdict": acc,
                   "ratios_vs_reference": {k: s12[k] / ref[k] for k in KEYS} if ref else None}
        for other, label in (("isla_support4", "depth (SUPPORT-12 vs SUPPORT-4)"), ("isla_passive12", "interaction on a support (SUPPORT-12 vs PASSIVE-12)")):
            m = per_arm.get(other, {}).get("mean")
            if m:
                v["d1"][label] = {k: m[k] / s12[k] for k in KEYS}
    else:
        v["d1"]["accuracy_verdict"] = "PENDING (SUPPORT-12 needs both seeds evaluated)"
    nl = per_arm.get("gt_vol_nolocal", {}).get("mean")
    if nl and len(per_arm["gt_vol_nolocal"]["seeds"]) == 2:
        v["campaign_d"]["gt_vol_nolocal_nut"] = nl["nut_l2"]
        v["campaign_d"]["local_features_carry_the_lead"] = nl["nut_l2"] >= D_GT_NOLOCAL_LOCAL_CARRIES
    h = per_arm.get("isla_qtsdf_h344", {}).get("mean")
    if h and len(per_arm["isla_qtsdf_h344"]["seeds"]) == 2:
        v["campaign_d"]["isla_h344_nut"] = h["nut_l2"]
        v["campaign_d"]["h344_verdict"] = ("SUPPORTED: parameters carry" if h["nut_l2"] <= D_H344_SUPPORTED
                                            else "FALSIFIED" if h["nut_l2"] >= D_H344_FALSIFIED else "BETWEEN")
    return v


def table(per_arm):
    print("| arm | seeds | pressure | velocity | eddy viscosity |")
    print("|---|---|---|---|---|")
    for arm, label in ARMS.items():
        if arm in per_arm:
            m = per_arm[arm]["mean"]
            print(f"| {label} | {len(per_arm[arm]['seeds'])} | {m.get('pressure_l2', float('nan')):.4f} | {m.get('velocity_l2', float('nan')):.4f} | {m.get('nut_l2', float('nan')):.4f} |")
        else:
            print(f"| {label} | 0 | pending | pending | pending |")


if __name__ == "__main__":
    per_arm = load(sys.argv[1])
    table(per_arm)
    verdict = grade(per_arm)
    print(json.dumps(verdict, indent=1))
    out = Path(__file__).with_name(f"d1_campd_verdict_{date.today().isoformat()}.json")
    out.write_text(json.dumps({"source": sys.argv[1], "per_arm": per_arm, "verdict": verdict}, indent=1))
    print("wrote", out)
