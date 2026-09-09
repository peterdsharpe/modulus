"""POST HOC (not preregistered): how much rank does the frozen source-fitted basis need to reach
1% projection error on each held-out block, compared with the block's own oracle rank? Working
resolution N only (no 2N pass). Writes results_posthoc.json."""
import json, time
from pathlib import Path

import numpy as np

from run_study import K_COMMON, analyse, family_cases, fit_basis, proj_err, rank_at

HERE = Path(__file__).parent
t0 = time.time()
cases = family_cases()
blocks_all = {c["name"]: analyse(c, c["N"])[1] for c in cases}
print("blocks built", round(time.time() - t0), "s", flush=True)
out = {}
for pairing in ("L2", "energy"):
    for obj in ("full", "nullfree"):
        src = [blocks_all[c["name"]][pairing][(obj, i, i)] for c in cases if c["family"].startswith("source") for i in range(len(c["body"]))]
        U, V = fit_basis(src)
        rows = {}
        for c in cases:
            if c["family"].startswith("source"):
                continue
            for (name, i, j), M in blocks_all[c["name"]][pairing].items():
                if name != obj:
                    continue
                ro, _ = rank_at(M, (1e-2,)); r_or = max(int(ro["0.01"]), 1)
                errs = {r: proj_err(M, U, V, r) for r in range(1, U.shape[1] + 1)}
                r_src = next((r for r in range(1, U.shape[1] + 1) if errs[r] <= 1e-2), None)
                rows[f"{c['name']}[{i},{j}]"] = {"family": c["family"], "kind": "self" if i == j else "coupling", "oracle_rank_1pct": r_or,
                                                "source_rank_1pct": r_src, "err_at_2x": errs[min(2 * r_or, U.shape[1])], "err_at_4x": errs[min(4 * r_or, U.shape[1])],
                                                "n_modes": int(U.shape[1])}
        out[f"{pairing}/{obj}"] = rows
        fams = sorted(set(v["family"] for v in rows.values()))
        for fam in fams:
            sub = [v for v in rows.values() if v["family"] == fam]
            print(f"{pairing}/{obj:8s} {fam:16s} oracle med {np.median([v['oracle_rank_1pct'] for v in sub]):.0f} | source-basis rank for 1%: med {np.median([v['source_rank_1pct'] or 999 for v in sub]):.0f} max {max(v['source_rank_1pct'] or 999 for v in sub)} (999 = not reached within {sub[0]['n_modes']} modes) | err at 2x oracle rank med {np.median([v['err_at_2x'] for v in sub]):.3f}, at 4x {np.median([v['err_at_4x'] for v in sub]):.3f}", flush=True)
(HERE / "results_posthoc.json").write_text(json.dumps(out, indent=1))
print("done", round(time.time() - t0), "s")
