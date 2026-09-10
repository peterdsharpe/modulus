"""Check that two saved prediction artifacts were scored on the same interior
points (protocol rule: compare only same-point evaluations).

Usage:
  python check_eval_points.py REF_RUN_DIR CAND_RUN_DIR [n_cases]

Each RUN_DIR is the recipe's output directory for one run
(``.../<run>/<run>``, containing ``metrics.jsonl`` and ``predictions/``).
For every common case the interior ``points`` memmaps are compared:

* same count  -> ``allclose`` (atol 1e-6 after centering is NOT applied; the
  saved points are what the metric saw) and the per-case pressure relative L2
  from both ``metrics.jsonl`` files;
* candidate smaller (D1 arms score 5,000 queries against a 10,000-point
  reference sample) -> the fraction of candidate rows found exactly in the
  reference, and whether they equal the reference's second half in order
  (the support transform takes the first ``n_support`` rows as support).

Prints one line per case and a summary; exits 1 if any case fails.
"""
import json
import os
import sys

import numpy as np


def _memmap(td_dir, name):
    meta = json.loads(open(os.path.join(td_dir, "meta.json")).read())
    m = meta[name]
    dtype = np.dtype(m["dtype"].replace("torch.", ""))
    return np.memmap(os.path.join(td_dir, f"{name}.memmap"), dtype=dtype, mode="r", shape=tuple(m["shape"]))


def interior_points(run_dir, case):
    td = os.path.join(run_dir, "predictions", case, "_tensordict", "interior", "_tensordict")
    return np.asarray(_memmap(td, "points"), dtype=np.float64)


def per_case_metric(run_dir, key="pressure_l2"):
    out = {}
    for line in open(os.path.join(run_dir, "metrics.jsonl")):
        r = json.loads(line)
        if r.get("phase") == "infer_step" and "sample_id" in r:
            out[r["sample_id"]] = r["metrics"].get(key)
    return out


def main(ref, cand, n=None):
    cases = sorted(set(os.listdir(os.path.join(ref, "predictions"))) & set(os.listdir(os.path.join(cand, "predictions"))))
    if n:
        cases = cases[: int(n)]
    mref, mcand = per_case_metric(ref), per_case_metric(cand)
    ok_all = True
    for c in cases:
        P, Q = interior_points(ref, c), interior_points(cand, c)
        if P.shape == Q.shape:
            same = np.allclose(P, Q, atol=1e-6, rtol=0)
            verdict = "IDENTICAL" if same else f"DIFFER max|dP|={np.abs(P - Q).max():.3e}"
            ok = same
        else:
            # row membership via exact match on a structured view
            Pv = np.ascontiguousarray(P).view([("", P.dtype)] * P.shape[1]).ravel()
            Qv = np.ascontiguousarray(Q).view([("", Q.dtype)] * Q.shape[1]).ravel()
            frac = np.isin(Qv, Pv).mean()
            tail = P.shape[0] - Q.shape[0]
            second_half = tail >= 0 and np.allclose(P[tail:], Q, atol=1e-6, rtol=0)
            verdict = f"SUBSET frac={frac:.4f} second_half_in_order={second_half}"
            ok = frac > 0.999
        ok_all &= ok
        print(f"{c}: ref {P.shape[0]} pts, cand {Q.shape[0]} pts, {verdict}; pressure_l2 ref {mref.get(c)} cand {mcand.get(c)}")
    print("ALL_OK" if ok_all else "MISMATCH")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
