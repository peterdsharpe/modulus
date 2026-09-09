# Campaign B (POSE-BENCH) notebook

## 2026-09-09 — Preregistered and built

Question, arms and bars are in PREREG.md (written before any lane). What was built:

- `MeshReaderWithGlobalData(store_case_key=True)` (recipe `src/merge_global_data.py`):
  a stable per-case 0-dim int64 global field `case_key` = CRC-32 of the case
  directory name (`run_1`, ...). 0-dim fields are invariant under `Mesh.transform`,
  so rotations leave it alone.
- `FixedRandomPose(salt, key_field="case_key")` (recipe `src/domain_transforms.py`):
  a deterministic transform (no generator, never reseeded) drawing a uniform SO(3)
  rotation from a generator seeded by `(case_key, salt)` (unit-quaternion method, the
  same construction as `RandomRotateMesh(mode="uniform")`), applied with
  `transform_point_data/cell_data/global_data=True`, so positions, normals (computed
  after it), wall shear and `U_inf` rotate together; scalars are invariant. Placed
  before `ComputeFreestreamDirection`.
- Tests (`tests/test_fixed_random_pose.py`, 4 pass with the repository venv):
  rotation deterministic in key and salt, orthogonal with determinant +1; positions,
  wall shear and `U_inf` rotate by the same matrix, scalars and the key unchanged,
  two loads of the same case agree; the reader's key is stable across loads and
  differs between cases.
- Dataset variant `drivaer_ml_surface_pose.yaml` (repository copy derived from the
  repository's surface config; the cluster copy is derived by the deploy script from
  the cluster's FROZEN surface config, so that the two differ only in the three
  documented edits). The augmentation block is a uniform SO(3) `RandomRotateMesh`
  and is active only with `augment=true`; the recipe inserts augmentations after
  `CenterMesh`, and in manifest mode builds an un-augmented validation dataset, so
  validation is the fixed posed benchmark for every arm. Decision: one variant
  serves ISLA (`augment=false`) and the augmented baselines (`augment=true`) rather
  than two files; the translation augmentation of the frozen config is dropped for
  every arm (it was never active in the frozen lanes either, which ran with
  `augment=false`).
- Deployment on AGA: recipe sources copied into `$T/recipe_support/src` (the
  cluster's frozen `merge_global_data.py` predates the repository version, so it was
  patched additively in place rather than replaced); lane table
  `$T/transfer/campaign_b_lanes.tsv` (8 lanes); launcher
  `$T/transfer/campaign_b_aga.sbatch` (instwave clone reading the lane table, job
  names camp-b-$i, RECIPE=recipe_support, CODE=code_support, timeout 3600, LOG_START
  guard, `dataset=drivaer_ml_surface_pose`, `augment` per lane); eval launcher
  `$T/transfer/campaign_b_eval_aga.sbatch` (iw_eval clone; every arm scored on
  `drivaer_ml_surface_pose`; outputs `$T/iw_evals/campB_*`).
