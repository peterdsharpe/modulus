"""Realized wake-token positions for one car, from a saved prediction artifact: ell = measure-weighted RMS extent of the
vehicle surface along the drive; tokens at s_mean + c*ell (c = 1, 2, 4) on the drive axis; expressed in body lengths and
relative to the car's rear."""
import json, os, sys
import numpy as np
T = "/scratch/fsw/portfolios/coreai/projects/coreai_modulus_cae/users/psharpe/agents/2026-08-09-mt2-stage0"
run = "v0_isla_qtsdf_h344_seed42"
pred = f"{T}/v0_evals_fp32/{run}/{run}/predictions"
def mm(d, name):
    m = json.load(open(os.path.join(d, "meta.json")))[name]
    return np.memmap(os.path.join(d, f"{name}.memmap"), dtype=np.dtype(m["dtype"].replace("torch.", "")), mode="r", shape=tuple(m["shape"]))
for case in sorted(os.listdir(pred))[:3]:
    td = f"{pred}/{case}/_tensordict"
    bd = f"{td}/boundaries/vehicle/_tensordict"
    P = np.asarray(mm(bd, "points"), dtype=np.float64); C = np.asarray(mm(bd, "cells")).astype(np.int64)
    gd = f"{td}/interior/_tensordict/global_data"
    keys = json.load(open(os.path.join(gd, "meta.json")))
    name = next((k for k in ("U_inf_dir", "U_inf", "freestream", "velocity_inf") if k in keys), None)
    if name is None:
        d = np.array([1.0, 0.0, 0.0])  # DrivAerML convention: freestream along +x
        print(f"  (global_data keys {[k for k in keys if isinstance(keys[k], dict)]}; using +x as the drive)")
    else:
        d = np.asarray(mm(gd, name), dtype=np.float64).ravel(); d = d / np.linalg.norm(d)
    tri = P[C]; cen = tri.mean(1); area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    cdm = json.load(open(f"{bd}/cell_data/meta.json")); w = area
    if "_measure_weights" in cdm:
        w = area * np.asarray(mm(f"{bd}/cell_data", "_measure_weights"), dtype=np.float64).ravel()
    s = cen @ d; wn = w / w.sum(); s_mean = (wn * s).sum(); ell = np.sqrt((wn * (s - s_mean) ** 2).sum())
    L_body = s.max() - s.min(); rear = s.max()  # drive points downstream: the rear has the largest drive-aligned coordinate
    print(f"{case[:22]}: cells {len(C)}, body length along drive {L_body:.3f}, weighted RMS extent ell {ell:.3f} (= {ell/L_body:.3f} L), "
          f"weighted centre at {(s_mean - s.min())/L_body:.2f} L from the nose; rear at {(rear - s_mean)/ell:.2f} ell behind the centre")
    for c in (1.0, 2.0, 4.0):
        pos = s_mean + c * ell
        print(f"   token c={c}: {(pos - s_mean)/L_body:.2f} L behind the centre, {(pos - rear)/L_body:+.2f} L relative to the rear ({'downstream of' if pos > rear else 'inside'} the body)")
