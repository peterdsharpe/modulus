# Campaign A notebook: data-draw variance of the HiLiftAeroML comparisons

## 2026-09-09 — Preregistered; manifests, lane table and launchers built

Setting: every HiLiftAeroML ratio in the program (unit-drive GeoTransolver
and Transolver against ISLA at 35 and 210 cases; ISLA with and without
routing measure weights) is measured on one curated training draw with two
or three seeds, so the uncertainty from which cases were drawn is unknown.
Campaign A trains the four arms on three independent, geometry-stratified
draws per rung (build_campaign_a.py; PREREG.md for the stratification rule,
seeds 101–103) with three seeds each, 72 lanes, all validated on the shared
180-case set. The main program's concurrent WAVE-3 stays on the curated draw
(third seeds, other rungs, steel-men), so no lane is duplicated; its curated
third seeds become the comparison "fourth draw".

Instrument details recorded before launch: code snapshot `code_support`
(ISLA defaults bitwise equal to the reference configuration), recipe copy
`recipe_support` carrying the six draw dataset variants
(`highlift_surface_draw{1,2,3}_n{35,210}.yaml`), launcher
`transfer/campaign_a_aga.sbatch` (job names camp-a-<lane>, chain-ahead
singleton links, 3600 s distributed timeout, per-link failure fingerprint),
evaluation launcher `transfer/campaign_a_eval_aga.sbatch` writing to
`hl_evals/<run_id>/`. Launch procedure: one pilot lane per arm, launch
acceptance by a script-file watcher (log advances, no traceback, a `Loss:`
line), then the remaining 68 lanes.

## 2026-09-09 — Launched: 72 chains (camp-a-0..71)

Draws built from the actual pool structure: `full_train` is 180 geometries
× 7 angles (not 126 × 10 as first assumed; the rule for the 210-case draw
was corrected before any lane ran and PREREG.md updated: every geometry
once plus 30 second cases, 21 per angle). Draw summary
(`campaign_a_summary.json`): 35-case draws have 35 geometries each and
pairwise overlaps of 1, 1 and 0 cases; 210-case draws have 180 geometries
each and pairwise overlaps of 36, 33 and 33 cases.

Pilots (lanes 0, 3, 6, 9: unit-drive GeoTransolver, Transolver, ISLA,
ISLA-noweights at 35 cases, draw 1, seed 42; jobs 691732–691735) accepted:
at one hour all four were at epoch 100 of 500 with no error (per-GPU memory
4.6 / 3.2 / 3.9 / 3.9 GB), so 35-case lanes take about five hours. The
remaining 68 chains were then submitted (72 distinct camp-a-<lane> singleton
names; 4 running, 68 pending at submission). Evaluation is submitted per
trained lane by a watcher (`campaign_a_eval_aga.sbatch --array=<lane>`,
short queue) and reduced by `reduce_campaign_a.py` into
`$T/transfer/campaign_a_reduction.json`; the verdicts against PREREG.md are
written here when the reduction covers all 72 runs.
