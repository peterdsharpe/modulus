# Campaign C notebook — generalization on the DrivAerML → SHIFT-SUV fastback shift

## 2026-09-09 — Preregistered, plumbing built and tested, six acceptance lanes submitted

**Setting.** The audit's physical-input prerequisite and evaluation table
(research/cross_dataset_generalization/03-agenda.qmd) asked for three
real-data tests on the one transfer setting the program has: complete
conditioning, matched-count diversity, and few-shot adaptation, each with a
falsifier, force error beside field error, and unit-direction drive for every
model. PREREG.md (this directory) was written before any lane.

**Plumbing added (repository, this branch).** `SetConstantCellField`
(recipe `src/domain_transforms.py`): a constant per-cell scalar written into
`cell_data` per dataset pipeline (0 for DrivAerML, 1 for SHIFT-SUV), read by
ISLA as a boundary scalar and by GeoTransolver as a seventh functional channel.
`training.init_from` (recipe `conf/train.yaml`, hook in `src/train.py`,
function `initialize_from_checkpoint` in `src/utils.py`): weight-only
initialization from another run's checkpoint directory through
`Module.load(strict=True)`, so a checkpoint saved under the legacy
`MeshTransformer2` name loads into `ISLA` while any architecture mismatch
raises; optimizer, scheduler and epoch start fresh; ignored once the run has
its own checkpoints (chained links resume). Tests
`tests/test_campaign_c_hooks.py` (4) pass with the existing transform and
forward-kwargs tests (43 total).

**Cluster deployment** (`build_campaign_c.py`, run on AGA): a recipe copy
`$T/recipe_campc` (from `$T/recipe_support`, so neither the frozen recipe nor
the D1 copy is touched) with the two hooks patched in; manifests under
`$T/transfer/manifests_c/` (DrivAerML 218 of 435, estate 217 of 794, fastback
20 + 10 of the 796 train cases, all with RNG seed 2026; the 99 fastback
validation cases are never used for adaptation); dataset variants
`campc_drivaer_cond0`, `campc_estate_cond1`, `campc_fastback_cond1`,
`campc_drivaer_n218`, `campc_estate_n217`, `campc_fastback_fewshot20`; lane
table `$T/transfer/campaign_c_lanes.tsv` (20 lanes; 0–15 now, 16–19 the
GeoTransolver few-shot arms once `uw_gt_unit_lr1e3_seed42` finishes);
launcher `$T/transfer/campaign_c_aga.sbatch` (clone of hl_udrv: chain-ahead
singleton, timeout 3600, LOG_START guard, per-lane epoch count parsed from
the extra column so T3's 200 epochs are not overridden); eval launcher
`$T/transfer/campaign_c_eval_aga.sbatch` (per lane: fastback 99-case
validation, and the DrivAerML 48-car validation for T1/T2, with the
condition-carrying variants for the conditioned arms; outputs
`$T/campc_evals/<run>/{fastback,drivaer}/`).

**Acceptance lanes submitted** (jobs 691784–691789, names camp-c-{0,2,3,8,12,13}):
cond0 ISLA, cond1 ISLA, cond1 GeoTransolver (exercises the seventh channel),
mix435 ISLA, fine-tune-20 ISLA, scratch-20 ISLA. The remaining ten of lanes
0–15 follow once each of these shows a first training step.

**Commands for whoever finishes this** (coordinator or a later session):
- submit the rest: `cd $T && for i in 1 4 5 6 7 9 10 11 14 15; do sbatch -J camp-c-$i --array=$i --dependency=singleton transfer/campaign_c_aga.sbatch; done`
- GT few-shot arms after `grep -q "Training completed" $T/runs/uw_gt_unit_lr1e3_seed42/train.log`: `for i in 16 17 18 19; do sbatch -J camp-c-$i --array=$i --dependency=singleton transfer/campaign_c_aga.sbatch; done`
- evaluate a trained lane: `sbatch -q short -t 01:30:00 --array=<idx> transfer/campaign_c_eval_aga.sbatch`
- reduce: field relative L2 and pressure-force relative error per case from `$T/campc_evals/<run>/fastback/*/predictions` with the C1 diagnostic's conventions (`studies/controls/pressure_offset_diagnostic.py` shows the loaders: gauge pressure via p_inf, areas from saved triangles, normals from saved cells); two-seed mean predictions; bars in PREREG.md.
