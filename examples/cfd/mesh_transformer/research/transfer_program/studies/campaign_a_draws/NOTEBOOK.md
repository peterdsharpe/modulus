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
