"""Deliverable 3: log-log slopes of pressure_l2 vs training cases, two- and three-seed conventions."""
import json, math
import os
R = os.path.dirname(os.path.abspath(__file__))
lad = json.load(open(f"{R}/hilift_ladder_reduction_2026-09-03.json"))
w2 = json.load(open(f"{R}/w2_reduction_2026-09-07b.json"))["runs"]

def val(run):
    if run in lad:
        return lad[run]["pressure_l2"]
    return w2[run]["pressure_l2"]

N = {"35": 35, "210": 210, "1260": 1260}
runs = {
    "gt": {"35": "lad_hl_gt_super_scarce_seed{s}", "210": "lad_hl_gt_scarce_seed{s}", "1260": "gt_hl_lr1_seed{s}"},
    "isla": {"35": "lad_hl_mt2_super_scarce_seed{s}", "210": "lad_hl_mt2_scarce_seed{s}", "1260": "mt2_hl_lr1_seed{s}"},
}
seedsets = {"two_seed_42_43": {"35": [42, 43], "210": [42, 43], "1260": [42, 43]},
            "three_seed_42_43_44_where_available": {"35": [42, 43, 44], "210": [42, 43, 44], "1260": [42, 43]}}

def slope(n1, e1, n2, e2):
    return -(math.log(e2) - math.log(e1)) / (math.log(n2) - math.log(n1))

out = {"date": "2026-09-08", "metric": "pressure_l2 (validation mean over 180 full_val cases; seed mean of per-run means)",
       "sources": {"seeds_42_43": "results/hilift_ladder_reduction_2026-09-03.json", "seed_44": "results/w2_reduction_2026-09-07b.json (runs)"},
       "slope_definition": "slope = -d ln(error) / d ln(n_cases) between adjacent rungs; positive = error falls with data",
       "conventions": {}}
for conv, ss in seedsets.items():
    c = {"per_arm": {}}
    for arm, tmpl in runs.items():
        per = {}
        for rung, t in tmpl.items():
            vals = [val(t.format(s=s)) for s in ss[rung]]
            per[rung] = {"seeds": ss[rung], "per_seed": vals, "mean": sum(vals) / len(vals)}
        s1 = slope(35, per["35"]["mean"], 210, per["210"]["mean"])
        s2 = slope(210, per["210"]["mean"], 1260, per["1260"]["mean"])
        c["per_arm"][arm] = {"rungs": per, "slope_35_to_210": s1, "slope_210_to_1260": s2}
    # data multiplier: cases GT needs to hit ISLA's 35-case error, log-linear interpolation on GT curve
    g = c["per_arm"]["gt"]["rungs"]; i35 = c["per_arm"]["isla"]["rungs"]["35"]["mean"]
    pts = [(35, g["35"]["mean"]), (210, g["210"]["mean"]), (1260, g["1260"]["mean"])]
    target = i35
    n_needed = None; segment = None
    for (na, ea), (nb, eb) in zip(pts[:-1], pts[1:]):
        if (ea >= target >= eb) or (ea <= target <= eb):
            t = (math.log(target) - math.log(ea)) / (math.log(eb) - math.log(ea))
            n_needed = math.exp(math.log(na) + t * (math.log(nb) - math.log(na)))
            segment = [na, nb]
            break
    c["data_multiplier"] = {
        "definition": "training cases GeoTransolver needs to reach ISLA's 35-case seed-mean pressure_l2, log-log linear interpolation between adjacent GT rungs",
        "isla_35_error": i35, "gt_curve": pts, "interpolation_segment": segment,
        "gt_cases_needed": n_needed, "multiplier_vs_35": (n_needed / 35) if n_needed else None,
    }
    out["conventions"][conv] = c

out["audit_claim_check"] = {
    "audit_reported_gt_three_seed": [0.553, 0.675],
    "computed_gt_three_seed": [out["conventions"]["three_seed_42_43_44_where_available"]["per_arm"]["gt"]["slope_35_to_210"],
                               out["conventions"]["three_seed_42_43_44_where_available"]["per_arm"]["gt"]["slope_210_to_1260"]],
    "book_gt_two_seed": [0.50, 0.74],
    "computed_gt_two_seed": [out["conventions"]["two_seed_42_43"]["per_arm"]["gt"]["slope_35_to_210"],
                             out["conventions"]["two_seed_42_43"]["per_arm"]["gt"]["slope_210_to_1260"]],
    "book_isla_two_seed": [0.43, 0.25],
    "computed_isla_two_seed": [out["conventions"]["two_seed_42_43"]["per_arm"]["isla"]["slope_35_to_210"],
                               out["conventions"]["two_seed_42_43"]["per_arm"]["isla"]["slope_210_to_1260"]],
    "computed_isla_three_seed": [out["conventions"]["three_seed_42_43_44_where_available"]["per_arm"]["isla"]["slope_35_to_210"],
                                 out["conventions"]["three_seed_42_43_44_where_available"]["per_arm"]["isla"]["slope_210_to_1260"]],
}
json.dump(out, open(f"{R}/hilift_slopes_threeseed_2026-09-08.json", "w"), indent=1)
print(json.dumps(out["audit_claim_check"], indent=1))
for conv in out["conventions"]:
    print(conv, json.dumps(out["conventions"][conv]["data_multiplier"]))
    for arm in ["gt", "isla"]:
        a = out["conventions"][conv]["per_arm"][arm]
        print(" ", arm, {k: round(v["mean"], 5) for k, v in a["rungs"].items()}, round(a["slope_35_to_210"], 4), round(a["slope_210_to_1260"], 4))
