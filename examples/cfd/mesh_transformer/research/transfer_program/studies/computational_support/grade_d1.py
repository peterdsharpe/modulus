"""Grade D1 (computational support with passive readout) and campaign D
(interior carriers) against their preregistered bars, in float32.

Reads per-run ``metrics.jsonl`` (48-car ``infer_step`` rows: interior
pressure, velocity and eddy-viscosity relative L2) from
  ``$T/v0_evals_fp32/<run>/<run>/``            (D1 and campaign-D arms, float32)
  ``$T/v0_evals_support_fp32/<run>/<run>/``    (same-snapshot float32 references)
and optionally the main session's float32 unit-drive GeoTransolver-volume
reference (``--gt-unit-nut``) for campaign D's eddy-viscosity bars.

Bars are ratios copied from PREREG.md (D1) and the notebook's campaign entry
(campaign D), evaluated at the float32 reference at run time:
  D1 supported  : SUPPORT-12 <= 1.03 x reference on pressure, velocity AND eddy viscosity
  D1 falsified  : SUPPORT-12 pressure >= 1.10 x reference
  D  local carry: GT-volume without local features nu_t >= 1.10 x unit-drive GT-volume nu_t
  D  h344       : ISLA hidden-344 nu_t <= 1.10 x GT-volume nu_t supported; >= 1.25 x falsified

Usage: python grade_d1.py [--gt-unit-nut 0.0xx] [--out path.json]
Run on the cluster (stdlib only).
"""
import argparse
import glob
import json
import statistics
from datetime import date
from pathlib import Path

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
KEYS = ("pressure_l2", "velocity_l2", "nut_l2")
ARMS = {
    "isla_qtsdfval": ("v0_evals_support_fp32", "reference: ISLA query tokens + SDF, 10k interacting queries"),
    "gt_vol": ("v0_evals_support_fp32", "GeoTransolver-volume reference (physical drive)"),
    "udrv_gt_vol": ("v0_evals_support_fp32", "GeoTransolver-volume reference (unit drive; campaign D's eddy-viscosity reference)"),
    "isla_support12": ("v0_evals_fp32", "SUPPORT-12: 5k interacting support + 5k passive queries, 12 read blocks"),
    "isla_support4": ("v0_evals_fp32", "SUPPORT-4: same, 4 read blocks"),
    "isla_passive12": ("v0_evals_fp32", "PASSIVE-12: no support, 5k passive queries, 12 read blocks"),
    "isla_qtsdf_h344": ("v0_evals_fp32", "campaign D: ISLA QT+SDF, hidden 344 (27.6M params)"),
    "gt_vol_nolocal": ("v0_evals_fp32", "campaign D: GeoTransolver-volume without local features (unit drive)"),
}


def run_metrics(root, run):
    ps = glob.glob(f"{T}/{root}/{run}/**/metrics.jsonl", recursive=True)
    if not ps:
        return None
    rows = [json.loads(l)["metrics"] for l in open(ps[0]) if '"infer_step"' in l]
    if not rows:
        return None
    return {"n": len(rows), **{k: statistics.mean(r[k] for r in rows) for k in KEYS if k in rows[0]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-unit-nut", type=float, default=None, help="float32 nu_t of the unit-drive GeoTransolver-volume reference (main session)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    per_arm = {}
    for arm, (root, label) in ARMS.items():
        prefix = "" if arm.startswith("udrv_") else "v0_"
        runs = {s: run_metrics(root, f"{prefix}{arm}_seed{s}") for s in (42, 43)}
        runs = {s: r for s, r in runs.items() if r}
        if runs:
            per_arm[arm] = {"label": label, "seeds": sorted(runs), "per_seed": runs,
                            "mean": {k: statistics.mean(r[k] for r in runs.values()) for k in KEYS if all(k in r for r in runs.values())}}
    # bf16 side (same runs, same points; v0_evals/ for the arms, v0_evals_support/ for the references) for the
    # per-architecture offset table and the direct test of whether GeoTransolver-volume's bf16 inflation comes
    # from its local features (compare gt_vol_nolocal's inflation with gt_vol's / udrv_gt_vol's).
    bf16 = {}
    for arm, (root, _) in ARMS.items():
        prefix = "" if arm.startswith("udrv_") else "v0_"
        broot = root.replace("_fp32", "")
        runs = {s: run_metrics(broot, f"{prefix}{arm}_seed{s}") for s in (42, 43)}
        runs = {s: r for s, r in runs.items() if r}
        if runs and arm in per_arm:
            bmean = {k: statistics.mean(r[k] for r in runs.values()) for k in KEYS if all(k in r for r in runs.values())}
            bf16[arm] = {"mean": bmean, "bf16_over_fp32": {k: bmean[k] / per_arm[arm]["mean"][k] for k in bmean if k in per_arm[arm]["mean"]}}
    ref = per_arm.get("isla_qtsdfval", {}).get("mean")
    v = {"d1": {}, "campaign_d": {}, "precision": "float32", "bf16_offset": bf16}
    if ref:
        v["d1"]["bars_at_reference"] = {"supported_max": {k: 1.03 * ref[k] for k in KEYS}, "falsified_pressure_min": 1.10 * ref["pressure_l2"]}
    s12 = per_arm.get("isla_support12")
    if ref and s12 and len(s12["seeds"]) == 2:
        m = s12["mean"]; ratios = {k: m[k] / ref[k] for k in KEYS}
        if ratios["pressure_l2"] >= 1.10:
            acc = "FALSIFIED"
        elif all(r <= 1.03 for r in ratios.values()):
            acc = "SUPPORTED (accuracy half; sentinel-query study still required)"
        else:
            acc = "BETWEEN"
        v["d1"].update({"support12_mean": m, "ratios_vs_reference": ratios, "accuracy_verdict": acc})
        gt = per_arm.get("gt_vol", {}).get("mean")
        if gt:
            v["d1"]["support12_over_gt_vol_pressure"] = m["pressure_l2"] / gt["pressure_l2"]
        for other, label in (("isla_support4", "depth: SUPPORT-4 / SUPPORT-12"), ("isla_passive12", "interaction on a support: PASSIVE-12 / SUPPORT-12")):
            o = per_arm.get(other, {}).get("mean")
            if o:
                v["d1"][label] = {k: o[k] / m[k] for k in KEYS}
    elif ref:
        v["d1"]["accuracy_verdict"] = "PENDING (SUPPORT-12 needs both seeds evaluated)"
    gt_unit_nut = a.gt_unit_nut
    if gt_unit_nut is None and per_arm.get("udrv_gt_vol") and len(per_arm["udrv_gt_vol"]["seeds"]) == 2:
        gt_unit_nut = per_arm["udrv_gt_vol"]["mean"]["nut_l2"]  # same-snapshot float32 re-evaluation of the unit-drive reference
    v["campaign_d"]["gt_unit_nut_reference"] = gt_unit_nut
    nl = per_arm.get("gt_vol_nolocal")
    if nl and len(nl["seeds"]) == 2 and gt_unit_nut:
        v["campaign_d"]["gt_vol_nolocal_nut"] = nl["mean"]["nut_l2"]
        v["campaign_d"]["nolocal_over_unit_gt"] = nl["mean"]["nut_l2"] / gt_unit_nut
        v["campaign_d"]["local_features_carry_the_lead"] = nl["mean"]["nut_l2"] >= 1.10 * gt_unit_nut
    h = per_arm.get("isla_qtsdf_h344")
    if h and len(h["seeds"]) == 2 and gt_unit_nut:
        r = h["mean"]["nut_l2"] / gt_unit_nut
        v["campaign_d"]["isla_h344_nut"] = h["mean"]["nut_l2"]; v["campaign_d"]["h344_over_unit_gt"] = r
        v["campaign_d"]["h344_verdict"] = "SUPPORTED: parameters carry" if r <= 1.10 else "FALSIFIED" if r >= 1.25 else "BETWEEN"
    print("| arm | seeds | pressure | velocity | eddy viscosity |")
    print("|---|---|---|---|---|")
    for arm, (root, label) in ARMS.items():
        if arm in per_arm:
            m = per_arm[arm]["mean"]
            print(f"| {label} | {len(per_arm[arm]['seeds'])} | {m.get('pressure_l2', float('nan')):.4f} | {m.get('velocity_l2', float('nan')):.4f} | {m.get('nut_l2', float('nan')):.4f} |")
        else:
            print(f"| {label} | 0 | pending | pending | pending |")
    print(json.dumps(v, indent=1))
    out = Path(a.out) if a.out else Path(__file__).with_name(f"d1_campd_verdict_{date.today().isoformat()}.json")
    out.write_text(json.dumps({"per_arm": per_arm, "verdict": v}, indent=1)); print("wrote", out)


if __name__ == "__main__":
    main()
