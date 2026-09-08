"""Deliverable 2: HiLiftAeroML exposure ledger by case and by geometry."""
import json, re, collections, hashlib
import os, sys
R = os.path.dirname(os.path.abspath(__file__))
# argv[1]: pinned public HiLiftAeroML split manifest (dataset revision bbec30b),
# sha256 0a8bb68654bc67ba705598c0530b8edfe4bc17263788a71f340f697e5c5fb557
MP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "hilift_manifest_bbec30b.json")
sha = hashlib.sha256(open(MP, "rb").read()).hexdigest()
m = json.load(open(MP))

trained = ["full_train", "medium_train", "scarce_train", "super_scarce_train", "geometry_scarce_train",
           "geometry_super_scarce_train", "single_aoa_12_train", "aoa_train", "stall_train", "deflection_train"]
validated = ["full_val", "medium_val", "scarce_val", "super_scarce_val", "geometry_val", "geometry_scarce_val",
             "geometry_super_scarce_val", "geometry_medium_val", "single_aoa_12_val", "aoa_val", "stall_val", "deflection_val"]
test_evaluated = ["aoa_test", "stall_test"]
never_read_tests = ["full_test", "geometry_test", "deflection_test", "single_aoa_12_test"]
status_note = {"medium_train": "training in progress (2026-09-08); counted as trained", "deflection_train": "training in progress (2026-09-08); counted as trained",
               "geometry_medium_val": "identical case set to geometry_val; no geometry_medium run exists, listed for completeness"}
# validated splits that are identical sets
def same(a, b): return set(m[a]) == set(m[b])
identities = {"full_val == medium_val == scarce_val == super_scarce_val": all(same("full_val", x) for x in ["medium_val", "scarce_val", "super_scarce_val"]),
              "geometry_val == geometry_scarce_val == geometry_super_scarce_val == geometry_medium_val": all(same("geometry_val", x) for x in ["geometry_scarce_val", "geometry_super_scarce_val", "geometry_medium_val"]),
              "full_test == medium_test == scarce_test == super_scarce_test": all(same("full_test", x) for x in ["medium_test", "scarce_test", "super_scarce_test"]),
              "geometry_test == geometry_scarce_test == geometry_super_scarce_test == geometry_medium_test": all(same("geometry_test", x) for x in ["geometry_scarce_test", "geometry_super_scarce_test", "geometry_medium_test"])}

all_cases = sorted(set(c for v in m.values() for c in v))
def geo(c): return re.search(r"LHC\d+", c).group(0)
def aoa(c): return int(re.search(r"AoA_(\d+)", c).group(1))
geos = sorted(set(geo(c) for c in all_cases))

by_case = {}
for c in all_cases:
    cls = set(); splits = collections.defaultdict(list)
    for s in trained:
        if c in set(m[s]): cls.add("trained"); splits["trained"].append(s)
    for s in validated:
        if c in set(m[s]): cls.add("validated"); splits["validated"].append(s)
    for s in test_evaluated:
        if c in set(m[s]): cls.add("test_evaluated"); splits["test_evaluated"].append(s)
    unread = [s for s in never_read_tests if c in set(m[s])]
    if not cls: cls.add("unread")
    by_case[c] = {"geometry": geo(c), "aoa": aoa(c), "exposure_classes": sorted(cls), "splits": dict(splits),
                  "member_of_unread_test_sets": unread}

order = {"trained": 3, "validated": 2, "test_evaluated": 1, "unread": 0}
by_geo = {}
for g in geos:
    cs = [c for c in all_cases if geo(c) == g]
    counts = collections.Counter()
    for c in cs:
        for k in by_case[c]["exposure_classes"]: counts[k] += 1
    maximal = max((k for c in cs for k in by_case[c]["exposure_classes"]), key=lambda k: order[k])
    trained_splits = sorted(set(s for c in cs for s in by_case[c]["splits"].get("trained", [])))
    by_geo[g] = {"n_cases": len(cs), "maximal_exposure_class": maximal, "cases_per_class": dict(counts),
                 "n_cases_with_any_exposure": sum(1 for c in cs if by_case[c]["exposure_classes"] != ["unread"]),
                 "trained_in_splits": trained_splits,
                 "in_unread_test_sets": sorted(set(s for c in cs for s in by_case[c]["member_of_unread_test_sets"]))}

def inter(a, b): return len(set(m[a]) & set(m[b]))
evaluated_union = set(m["aoa_test"]) | set(m["stall_test"])
intersections = {}
for t in ["full_test", "geometry_test", "deflection_test", "single_aoa_12_test"]:
    intersections[t] = {"n_cases": len(m[t]), "n_geometries": len(set(geo(c) for c in m[t])),
                        "cap_aoa_test": inter(t, "aoa_test"), "cap_stall_test": inter(t, "stall_test"),
                        "cap_aoa_or_stall_test": len(set(m[t]) & evaluated_union),
                        "cap_full_val": inter(t, "full_val"), "cap_geometry_val": inter(t, "geometry_val"),
                        "cap_any_trained_split": len(set(m[t]) & set(c for s in trained for c in m[s])),
                        "cap_any_validated_split": len(set(m[t]) & set(c for s in validated for c in m[s])),
                        "cases_never_read_at_case_level": sum(1 for c in m[t] if by_case[c]["exposure_classes"] == ["unread"])}

# geometry-level: geometries whose cases are entirely held out of training in the geometry / deflection protocols
def held_out_geos(test_split):
    return sorted(set(geo(c) for c in m[test_split]))
geo_test_geos = held_out_geos("geometry_test"); defl_test_geos = held_out_geos("deflection_test")
def trained_geos_any():
    return set(geo(c) for s in trained for c in m[s])
tg = trained_geos_any()
summary = {
    "n_cases_total": len(all_cases), "n_geometries_total": len(geos),
    "geometries_with_zero_exposure_of_any_kind": [g for g in geos if by_geo[g]["n_cases_with_any_exposure"] == 0],
    "n_geometries_with_zero_exposure": sum(1 for g in geos if by_geo[g]["n_cases_with_any_exposure"] == 0),
    "geometries_covered_by_aoa_or_stall_test": len(set(geo(c) for c in evaluated_union)),
    "geometry_test_geometries": {"n": len(geo_test_geos), "list": geo_test_geos,
                                  "trained_on_in_any_split": sorted(set(geo_test_geos) & tg),
                                  "n_trained_on_in_any_split": len(set(geo_test_geos) & tg),
                                  "trained_on_in_geometry_protocol_splits": sorted(set(geo_test_geos) & set(geo(c) for s in ["geometry_scarce_train", "geometry_super_scarce_train"] for c in m[s]))},
    "deflection_test_geometries": {"n": len(defl_test_geos), "list": defl_test_geos,
                                    "trained_on_in_any_split": sorted(set(defl_test_geos) & tg),
                                    "n_trained_on_in_any_split": len(set(defl_test_geos) & tg),
                                    "trained_on_in_deflection_train": sorted(set(defl_test_geos) & set(geo(c) for c in m["deflection_train"]))},
    "case_class_counts": collections.Counter(tuple(by_case[c]["exposure_classes"]) for c in all_cases),
    "geometry_maximal_class_counts": collections.Counter(by_geo[g]["maximal_exposure_class"] for g in geos),
    "unread_confirmation_sets_at_case_level": {t: intersections[t]["cases_never_read_at_case_level"] for t in intersections},
}
summary["case_class_counts"] = {"+".join(k): v for k, v in summary["case_class_counts"].items()}
summary["geometry_maximal_class_counts"] = dict(summary["geometry_maximal_class_counts"])
per_split_geos = {s: len(set(geo(c) for c in m[s])) for s in m}

out = {"date": "2026-09-08", "dataset": "HiLiftAeroML", "dataset_revision": "bbec30bcfc6103309c1375c5228b3ad0a586bfaf",
       "manifest_sha256": sha, "manifest_sha256_expected": "0a8bb68654bc67ba705598c0530b8edfe4bc17263788a71f340f697e5c5fb557",
       "manifest_sha256_matches": sha == "0a8bb68654bc67ba705598c0530b8edfe4bc17263788a71f340f697e5c5fb557",
       "exposure_definitions": {"trained": trained, "validated": validated, "test_evaluated": test_evaluated + ["(2026-09-04, results/hilift_sealed_test_2026-09-04.json)"],
                                "never_read": never_read_tests, "status_notes": status_note},
       "split_identities_verified": identities, "split_sizes": {s: len(m[s]) for s in m}, "split_geometry_counts": per_split_geos,
       "summary": summary, "intersections": intersections, "by_geometry": by_geo, "by_case": by_case}
json.dump(out, open(f"{R}/hilift_exposure_ledger_2026-09-08.json", "w"), indent=1)
print("sha match", out["manifest_sha256_matches"])
print(json.dumps(identities))
print(json.dumps(summary, indent=1, default=str)[:4000])
print(json.dumps(intersections, indent=1))
print({s: per_split_geos[s] for s in ["full_val", "geometry_val", "single_aoa_12_val", "aoa_test", "stall_test", "full_test", "geometry_test", "deflection_test"]})
