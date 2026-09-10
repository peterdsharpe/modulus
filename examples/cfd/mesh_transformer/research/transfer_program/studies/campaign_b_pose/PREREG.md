# Preregistration: POSE-BENCH (campaign B), written 2026-09-09 before any lane

## Question

Every benchmark in the ISLA program presents geometries in one canonical frame, so
ISLA's exact SE(3) covariance has never been able to earn accuracy. On DrivAerML
surface pressure at 435 training cars, with every training AND validation case in an
independent random SO(3) pose: do GeoTransolver and plain Transolver, trained with
uniform SO(3) rotation augmentation at unit drive, lose accuracy relative to their
canonical-frame runs, while ISLA (invariant by construction) does not, and does ISLA
then lead them?

## Instrument

- Data: `drivaer_ml_surface_pose.yaml` = the frozen DrivAerML surface pipeline with
  (a) a stable per-case integer `case_key` stored by the reader (CRC-32 of the case
  directory name), (b) `FixedRandomPose(salt=0)`: a uniform SO(3) rotation that is a
  deterministic function of `case_key`, applied as a TRANSFORM before the freestream
  direction is computed, rotating positions, normals, wall shear, `U_inf` (hence
  `U_inf_dir`) together, for train and validation alike; (c) the augmentation block
  (active only with `augment=true`) is a uniform SO(3) `RandomRotateMesh` inserted
  after `CenterMesh` (the recipe's insertion point), rotating cell and global vector
  fields. In manifest mode the recipe builds an un-augmented validation dataset, so
  validation is the fixed posed benchmark for every arm. Unit test:
  `tests/test_fixed_random_pose.py` (determinism in key and salt; positions, wall
  shear and `U_inf` rotate by the same matrix; scalars invariant; reader key stable).
- Frozen protocol otherwise: 10,000 sampled cells, 500 epochs, bf16, lr as declared
  per architecture (GeoTransolver 1e-3, Transolver 3e-3, ISLA 1e-3), seeds 42/43,
  final checkpoint, the 48-car validation split; unit-direction drive for every
  model (`forward_kwargs.global_embedding=global_data.U_inf_dir` for GeoTransolver,
  `forward_kwargs.fx.source=global_data.U_inf_dir` for Transolver; ISLA's recipe
  uses the unit direction already).
- Arms (run ids `campB_dr_*`), all trained and scored on posed data:
  1. GeoTransolver + SO(3) augmentation, seeds 42/43 (`campB_dr_gt_aug_seed{42,43}`)
  2. Transolver + SO(3) augmentation, seeds 42/43 (`campB_dr_transolver_aug_seed{42,43}`)
  3. ISLA reference configuration, no augmentation, seeds 42/43 (`campB_dr_isla_seed{42,43}`)
  4. optional control: GeoTransolver WITHOUT augmentation, seed 42 (`campB_dr_gt_noaug_seed42`): what augmentation buys
  5. optional control: ISLA WITH SO(3) augmentation, seed 42 (`campB_dr_isla_aug_seed42`): a contract check at training scale (should equal arm 3 up to sampling)
- Canonical-frame references (not run here): ISLA `iw_mt2_lr1e3_seed{42,43}` =
  0.0588 / 0.0567 (mean 0.0577); unit-drive GeoTransolver and Transolver on canonical
  DrivAerML are the main session's WAVE-3 lanes `uw_gt_unit_lr1e3_seed{42,43,44}`,
  `uw_transolver_unit_lr{1e3,3e3}_seed{42,43}` (read from `$T/iw_evals/uw_*` when they
  land; the canonical column is marked pending until then).
- Readout: 48-car mean surface-pressure relative L2 (the recipe's standardized
  metric), two-seed mean per arm, with wall shear alongside; per-car paired
  comparisons (ISLA vs each augmented baseline; each arm posed vs canonical).

## Bars, in readout units (adopted from the main program's plan, #sec-nb-pose-qcount-plan)

- **Equivariance earns accuracy** if each augmented baseline's posed error is ≥ 10%
  above its own canonical-frame error, ISLA's posed error is within 3% of its
  canonical 0.0577 (i.e. ≤ 0.0594), and posed ISLA is ≥ 5% more accurate than each
  augmented baseline.
- **Augmentation suffices** if both augmented baselines stay within 3% of their
  canonical errors.
- Between: reported per arm. The ISLA contract check (arm 5) must agree with arm 3
  within the two-seed spread of arm 3, or the pose transform is not what it claims
  and every posed number is withdrawn pending repair.
- Derived from: the reference seed spread on canonical DrivAerML is 3.6% for ISLA
  (0.0588 vs 0.0567); 3% is therefore one seed spread; 10% is the smallest
  difference the program treats as a finding; 5% is the lead a two-seed mean can
  resolve against that spread.

## What each outcome changes

Equivariance earns accuracy: ISLA gains its first measured accuracy advantage over
input-matched baselines, in the one setting its construction targets; the program
book's contracts chapter gets a measured value, not a contract. Augmentation
suffices: exact equivariance is a convenience (no augmentation, no canonical frame
needed) with no accuracy value on this task, and the book says so. Either way the
arm-4 control tells how much of the baselines' posed accuracy is augmentation.

## Cost and budget

8 lanes (1 node each), ~2 days at the frozen protocol; job names camp-b-*; code
snapshot `$T/code_support` (its default forward equals the reference ISLA; no
datapipe change was needed, the transform lives in the recipe copy
`$T/recipe_support`).
