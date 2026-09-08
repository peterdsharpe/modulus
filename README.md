# Benchmarks for NVIDIA/physicsnemo PR #1966

Before/after measurements of `Mesh.slice_points` for
https://github.com/NVIDIA/physicsnemo/pull/1966. This branch only holds the
figures, raw results and scripts referenced from the PR comment; it is not
meant to be merged.

- `figures/` PNGs shown in the PR comment.
- `data/hilift_aga.jsonl` HiLiftAeroML reader path, one record per (variant, case, block size).
- `data/synthetic_*.json` synthetic sweeps.
- `scripts/` the benchmark drivers and the table/figure script. Each measurement
  runs in a fresh subprocess; `before` is the lookup-table algorithm from `main`
  reproduced verbatim inside the script, `after` is the branch's `Mesh.slice_points`.
