# Preregistration: campaign C, generalization on the DrivAerML → SHIFT-SUV fastback shift (written 2026-09-09, before any lane)

## Setting

Zero-shot transfer from DrivAerML (435 training cars; static wheels, fixed
freestream, OpenFOAM hybrid RANS–LES) to the SHIFT-SUV fastback family (99
validation cases; rolling floor, rotating wheels, different freestream speed
and detached-eddy formulation) fails for both architectures: source-only
relative L2 on surface pressure is 0.99 for ISLA and 0.59 for the
physical-drive GeoTransolver (three seeds), and control C1 showed the failure
is a spatial field error (offset share 1.5–3%) with integrated-force errors of
1.6x and 1.4x the true force. Training on DrivAerML plus the SHIFT-SUV estate
family (794 cases) brings both to 0.11–0.13. Three questions, each with its
own falsifier, all under the protocol rule that every model receives the
freestream as the unit direction (GeoTransolver
`forward_kwargs.global_embedding=global_data.U_inf_dir`).

Readouts for every arm: surface-pressure relative L2 and integrated
pressure-force relative error on the 99 fastback validation cases (two-seed
mean predictions; the case is the unit, since each fastback case is a
distinct geometry), plus the in-family DrivAerML validation error (48 cars)
where the arm trains on DrivAerML. Reductions with the C1 diagnostic's
conventions (gauge pressure, recomputed cell areas and normals).

## T1 — complete conditioning (8 lanes)

Train on DrivAerML + SHIFT-SUV estate with and without a per-case
dataset-condition input, for ISLA and unit-drive GeoTransolver, seeds 42/43.
The condition is a constant per-cell scalar written by the new
`SetConstantCellField` transform (0 in the DrivAerML pipeline, 1 in the
SHIFT-SUV pipelines; the fastback evaluation carries 1, since fastback shares
estate's solver, ground and wheel conventions). ISLA reads it as a boundary
scalar (`+model.n_boundary_scalars=1
+forward_kwargs.boundary_scalars=boundaries.vehicle.cell_data.cond`);
GeoTransolver as a seventh functional channel
(`forward_kwargs.local_embedding=[interior.points,boundaries.vehicle.cell_data.normals,boundaries.vehicle.cell_data.cond]
model.functional_dim=7`). Run ids campC_cond{0,1}_{isla,gt}_seed{42,43}
(cond0 = no condition input, same mixed data). lr 1e-3, 500 epochs, bf16,
10,000 cells, frozen protocol otherwise.

Bars. Conditioning MATTERS if the fastback relative L2 of the conditioned
arm is at least 15% below its unconditioned twin for either architecture
(the twins are trained in this campaign at the current protocol; the
historical mixed-family numbers 0.127 / 0.110 are reported alongside, not
used as the reference); NULL if within 5%; between, reported. The in-family
DrivAerML error must not rise by more than 5%, or the conditioning cost is
reported as such.

## T2 — matched-count diversity (4 lanes)

435 training cases mixed (218 DrivAerML + 217 estate, drawn deterministically
with seed 2026 from the two train splits) against 435 all-DrivAerML, for ISLA
and unit-drive GeoTransolver, seeds 42/43. Run ids
campC_mix435_{isla,gt}_seed{42,43}. The all-DrivAerML references are the
existing ISLA runs (iw_mt2_lr1e3_seed{42,43}: fastback 0.99 source-only) and
the main session's unit-drive GeoTransolver runs (uw_gt_unit_lr1e3_seed{42,43},
training; read from $T/iw_evals/uw_* when they land). Bar: diversity HELPS if
the fastback relative L2 falls by at least 30% relative to the all-DrivAerML
reference at a cost of at most 10% on the in-family DrivAerML error;
otherwise reported as the measured trade.

## T3 — few-shot adaptation (8 lanes: 4 now, 4 when the unit-drive GeoTransolver checkpoints exist)

Fine-tune the source-only checkpoints on 20 fastback TRAIN-split cases (with
10 further train-split cases for validation-based reporting; both sets drawn
deterministically with seed 2026 from the fastback train split of 796, case
ids recorded in the manifest), against training from scratch on the same 20
cases for the same number of optimizer steps; evaluate on the 99 fastback
validation cases (untouched by adaptation; identical to the zero-shot
readout). Budget: 200 epochs on 20 cases (4,000 steps at batch 1) at lr 1e-4
for fine-tuning (the `training.init_from` hook loads weights only) and the
same 4,000 steps at lr 1e-3 from scratch. ISLA now (init from
$T/runs/mt2_v3c_seed42, the DrivAerML reference; two seeds of the
fine-tuning RNG, 42/43), GeoTransolver when uw_gt_unit_lr1e3_seed42 completes
(a physical-drive init would confound the speed change with the adaptation;
not used). Run ids campC_ft20_{isla,gt}_seed{s}, campC_scratch20_{isla,gt}_seed{s}.

Bars. Adaptation is the credible fallback if fine-tuning reaches a fastback
relative L2 ≤ 0.15 with 20 labels (against 0.11–0.13 with the full 794-case
estate family) AND beats scratch by ≥ 2x; reported with the label count
(20 + 10) and steps (4,000) as its cost. If scratch on 20 cases already
reaches ≤ 0.15, the pretraining adds nothing at this budget.

## What each outcome changes

T1 matters → the compound physical-condition mismatch is (partly) an
identifiability problem and conditioning is a first-class input for
multi-family training; T1 null → the mismatch is not recoverable from a flag
and the estate family's benefit is coverage, not identification. T2 helps →
diversity at fixed cost is the cheapest transfer lever; T2 hurts in-family →
the trade is stated. T3 → the practical route when a few target labels
exist, with its cost.
