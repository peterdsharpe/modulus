# Sampled validation points are identical across the frozen cluster snapshots (2026-09-09)

Check performed after the scaling session found that one checkpoint (iw_mt2_lr1e3_seed42)
evaluated under the newly merged code + current recipe scores 0.06235 against the 0.05875
recorded at training time (+6.1%). Question: do the book's cross-arm comparisons, whose
evaluations ran under different frozen snapshots (`code`, `code_isla2`, `code_isla3`,
`code_isla4`, `code_isla5`), share sampled validation points?

Method: for the first six validation cases of each pair, load `interior/points.memmap` from
both runs' saved prediction artifacts ($T/{v0_evals,iw_evals}/<run>/<run>/predictions) and test
shape equality and `allclose(atol=1e-5)`; report the max absolute difference.

| pair (snapshot A vs snapshot B) | cases | identical | max |Δ| |
|---|---|---|---|
| v0_gt_vol_seed42 (`code`) vs v0_isla_qtsdfval_seed42 (`code_isla2`) | 6/6 | yes | 0.0 |
| v0_isla_qtsdfval_seed42 (`code_isla2`) vs v0_isla_qtsdf_h256_seed42 (`code_isla3`) | 6/6 | yes | 0.0 |
| v0_isla_qtsdfval_seed42 (`code_isla2`) vs v0_isla_qtsdf_surf10k_seed42 (`code_isla4`) | 6/6 | yes | 0.0 |
| v0_gt_vol_seed42 (`code`) vs udrv_gt_vol_seed42 (`code`) | 6/6 | yes | 0.0 |
| iw_mt2_lr1e3_seed42 (`code`) vs mom2_mt2_lr1e3_seed42 (`code_isla5`) | 6/6 | yes | 0.0 |
| iw_mt2_lr1e3_seed42 (`code`) vs now_mt2_lr1e3_seed42 (`code`) | 6/6 | yes | 0.0 |
| iw_mt2_lr1e3_seed42 (`code`) vs iw_gt_lr1e3_seed42 (`code`) | 6/6 | yes | 0.0 |

Together with results/measure_metric_35_2026-09-09.json (which asserts identical points across
the 35-case HiLift arms under `code`), every within-family comparison in the book was scored on
the same sampled points. The +6.1% discrepancy therefore arises in the evaluation path of the
newly merged code + recipe (snapshot `code_perf`), which no book number uses; new snapshots must
pass this identity check against a frozen-snapshot artifact before their evaluations are compared
with existing ones.
