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

## 2026-09-09 — Acceptance lanes submitted

Deployed on AGA (the cluster's `merge_global_data.py` predates the repository's, so
its `store_case_key` support was patched in additively; `domain_transforms.py` carries
both the D1 `SplitInteriorSupport` and `FixedRandomPose`; the dataset variant was
derived from the cluster's frozen `drivaer_ml_surface.yaml`). One lane per
architecture submitted first for launch acceptance: camp-b-0 (GeoTransolver +
SO(3) augmentation, seed 42; job 691774), camp-b-2 (Transolver + SO(3) augmentation,
seed 42; 691775), camp-b-4 (ISLA, no augmentation, seed 42; 691776). The remaining
five lanes (seeds 43 of the three arms; the two controls) follow once each arm has
written its first training step without error.

## 2026-09-09 — First acceptance attempt blocked; root cause and fix

All three acceptance lanes failed at training step 0 with
`RuntimeError: Expected all tensors to be on the same device, but got mat2 is on cpu,
different from other tensors on cuda:N (wrapper_CUDA_mm)`, one identical failure per
architecture, so the cause was the posed dataset, not a model. Root cause:
`FixedRandomPose` draws its rotation on a CPU `torch.Generator` (deliberately, so the
draw is a device-independent function of the case key) and moved the matrix only to
the mesh's dtype, not its device; the recipe runs its transform chain on the GPU, so
`mesh.transform(R, ...)` multiplied CUDA positions by a CPU matrix. The CPU-only unit
tests could not see it. Fix: the matrix is moved to the mesh's device and dtype
(`_matrix(global_data, like=mesh.points)`); tests re-run (3 pass); the patched
`domain_transforms.py` copied into `$T/recipe_support/src` (no code-snapshot change:
the transform lives in the recipe copy). The three BLOCKED markers and the runs'
`.last_failure` files were removed and the acceptance lanes resubmitted (jobs
691825/691826/691827); the remaining five lanes wait for their first steps.

## 2026-09-09 — Acceptance passed; all eight lanes submitted

After the device fix the three acceptance lanes trained without error and reached
epoch 72–80 within the first hour (GeoTransolver + SO(3) augmentation 0.28 s/step,
4.6 GB; Transolver + augmentation 0.07 s, 3.2 GB; ISLA 0.25 s, 3.9 GB, on posed
data at 10,000 cells). The remaining five lanes were then submitted: seeds 43 of the
three arms (camp-b-1, -3, -5; jobs 692627–692629) and the two controls,
GeoTransolver without augmentation (camp-b-6; 692630) and ISLA with augmentation
(camp-b-7; 692631). A watcher submits each run's evaluation
(`transfer/campaign_b_eval_aga.sbatch --array=<lane>`, every arm scored on the posed
validation set) when its training completes; outputs land in `$T/iw_evals/campB_*`.
Reduction: per-arm two-seed mean pressure (and wall-shear) relative L2 on the 48
posed validation cars against the canonical references (ISLA 0.0577; unit-drive
GeoTransolver and Transolver from the main session's `uw_*` runs when they land),
judged against the bars in PREREG.md.
