"""Reduce POSE-BENCH (campaign B): posed-validation errors per arm against canonical references.

Run on the cluster login node (stdlib only):
  python3 reduce_campaign_b.py <out.json>
Reads $T/iw_evals/<run>/*/metrics.jsonl (infer_step rows) for the campB_* runs and, when
present, the main session's canonical unit-drive references uw_gt_unit_lr1e3_seed*,
uw_transolver_unit_lr{1e3,3e3}_seed* (same layout). ISLA's canonical reference is fixed
(iw_mt2_lr1e3_seed{42,43}: 0.0588 / 0.0567). Prints the PREREG bars' verdict.
"""
import glob, json, os, statistics as st, sys

T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
ARMS = {
    "gt_aug": ["campB_dr_gt_aug_seed42", "campB_dr_gt_aug_seed43"],
    "transolver_aug": ["campB_dr_transolver_aug_seed42", "campB_dr_transolver_aug_seed43"],
    "isla": ["campB_dr_isla_seed42", "campB_dr_isla_seed43"],
    "gt_noaug": ["campB_dr_gt_noaug_seed42"],
    "isla_aug": ["campB_dr_isla_aug_seed42"],
}
CANON = {
    "gt": ["uw_gt_unit_lr1e3_seed42", "uw_gt_unit_lr1e3_seed43", "uw_gt_unit_lr1e3_seed44"],
    "transolver_lr1e3": ["uw_transolver_unit_lr1e3_seed42", "uw_transolver_unit_lr1e3_seed43"],
    "transolver_lr3e3": ["uw_transolver_unit_lr3e3_seed42", "uw_transolver_unit_lr3e3_seed43"],
}
ISLA_CANON = {"seeds": [0.0588, 0.0567], "mean": 0.05775}


def run_mean(run):
    ps = glob.glob(f"{T}/iw_evals/{run}/*/metrics.jsonl")
    if not ps or not os.path.exists(f"{T}/iw_evals/{run}/.done"):
        return None
    rows = [json.loads(l) for l in open(ps[0]) if l.strip()]
    rows = [r for r in rows if r.get("phase") == "infer_step"]
    out = {}
    for k in ("pressure_l2", "wss_l2"):
        v = [r["metrics"][k] for r in rows if k in r.get("metrics", {})]
        if v:
            out[k] = st.mean(v)
    out["n_cases"] = len(rows)
    return out


res = {"arms": {}, "canonical": {"isla": ISLA_CANON}}
for arm, runs in ARMS.items():
    per = {r: run_mean(r) for r in runs}
    got = [p for p in per.values() if p]
    res["arms"][arm] = {"runs": per, "pressure_mean": st.mean(p["pressure_l2"] for p in got) if got else None,
                        "wss_mean": st.mean(p["wss_l2"] for p in got if "wss_l2" in p) if any("wss_l2" in p for p in got) else None,
                        "n_seeds_done": len(got)}
for arm, runs in CANON.items():
    per = {r: run_mean(r) for r in runs}
    got = [p for p in per.values() if p]
    res["canonical"][arm] = {"runs": per, "pressure_mean": st.mean(p["pressure_l2"] for p in got) if got else None, "n_seeds_done": len(got)}

A = res["arms"]; C = res["canonical"]
lines = ["| arm | posed pressure L2 (mean) | seeds | canonical | posed ÷ canonical |", "|---|---|---|---|---|"]
def row(name, arm, canon_key):
    p = A[arm]["pressure_mean"]; c = C[canon_key]["pressure_mean"] if canon_key in C else None
    lines.append(f"| {name} | {p:.4f} | {A[arm]['n_seeds_done']} | {('%.4f' % c) if c else 'pending'} | {('%.3f' % (p / c)) if (p and c) else '—'} |" if p else f"| {name} | pending | {A[arm]['n_seeds_done']} | | |")
row("GeoTransolver + SO(3) aug", "gt_aug", "gt")
row("Transolver + SO(3) aug (lr 3e-3)", "transolver_aug", "transolver_lr3e3")
row("ISLA (no aug)", "isla", "isla")
row("GeoTransolver, no aug (control)", "gt_noaug", "gt")
row("ISLA + SO(3) aug (control)", "isla_aug", "isla")
print("\n".join(lines))

# verdict against PREREG bars
verdict = {}
pi = A["isla"]["pressure_mean"]; pg = A["gt_aug"]["pressure_mean"]; pt = A["transolver_aug"]["pressure_mean"]
cg = C["gt"]["pressure_mean"]; ct = C["transolver_lr3e3"]["pressure_mean"]
if pi and pg and pt:
    verdict["isla_within_3pct_of_canonical"] = pi <= 1.03 * ISLA_CANON["mean"]
    verdict["isla_leads_gt_by_5pct"] = pg / pi >= 1.05
    verdict["isla_leads_transolver_by_5pct"] = pt / pi >= 1.05
    if cg and ct:
        verdict["gt_posed_over_canonical"] = pg / cg; verdict["transolver_posed_over_canonical"] = pt / ct
        verdict["baselines_lose_ge_10pct"] = pg / cg >= 1.10 and pt / ct >= 1.10
        verdict["augmentation_suffices"] = pg / cg <= 1.03 and pt / ct <= 1.03
        if verdict["baselines_lose_ge_10pct"] and verdict["isla_within_3pct_of_canonical"] and verdict["isla_leads_gt_by_5pct"] and verdict["isla_leads_transolver_by_5pct"]:
            verdict["call"] = "EQUIVARIANCE EARNS ACCURACY"
        elif verdict["augmentation_suffices"]:
            verdict["call"] = "AUGMENTATION SUFFICES"
        else:
            verdict["call"] = "BETWEEN (report per arm)"
    else:
        verdict["call"] = "posed numbers complete; canonical GT/Transolver references pending (uw_*)"
else:
    verdict["call"] = "pending"
res["verdict"] = verdict
print(json.dumps(verdict, indent=1))
json.dump(res, open(sys.argv[1], "w"), indent=1)
