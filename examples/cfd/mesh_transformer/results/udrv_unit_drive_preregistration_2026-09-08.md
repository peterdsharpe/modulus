# Preregistration: UDRV, the unit-drive baseline control (2026-09-08, written before launch)

## Setting

Every surface comparison in the book feeds ISLA the unit freestream direction
`U_inf_dir` (conf/model/isla_surface.yaml) while GeoTransolver and plain
Transolver receive the physical freestream velocity `U_inf`
(geotransolver_surface.yaml `global_embedding`, transolver_surface.yaml
`fx.source`). The dataset pipelines nondimensionalize fields and coordinates
but not this global input, so on HiLiftAeroML the baselines see a vector of
magnitude 2679.5 in/s and on DrivAerML one of magnitude 38.9 m/s. The
independent audit (audit.md item 1) showed, and the coordinator reproduced
(results/geotransolver_drive_conditioning_2026-09-08.json), that
GeoTransolver's untrained global-context projector collapses from about 200
occupied slices per head at unit magnitude to one slice at 2679.5, with a
context vector of magnitude about 1500. The same frozen configurations were
used by every baseline lane on the cluster (verified in the task snapshot).

## Question

Does the baselines' physical-scale drive input carry a material part of
their small-data deficit on HiLiftAeroML? Two settings isolate different
things. At 35 mixed-angle training cases the drive direction varies across
cases and is informative, so a unit direction is the same information at a
benign scale. At the fixed angle (126 geometries at 12 degrees) the drive is
constant across cases, so the change is purely one of a constant input's
scale and the result measures conditioning alone.

Prior: DrivAerML's 38.9 m/s also collapses the projector (1.0 to 3.8
slices) and GeoTransolver still leads ISLA there by 1.2 to 1.3x from 27
cars up, so collapse is not fatal at that magnitude; the HiLift magnitude
is 69 times larger and the context entry 67 times larger. A partial effect
is the most likely outcome.

## Instrument

Eight lanes, frozen ladder protocol (10,000 sampled points, 500 epochs,
bf16, no augmentation, final checkpoint, validation split of the lane's own
split), launcher highlift/hl_udrv_aga.sbatch with lane table
highlift/udrv_lanes.tsv; the only change from the reference baseline lanes is
the hydra override of the drive input:

| Lane | Model | Split | lr | Seeds | Override |
|---|---|---|---|---|---|
| 0-1 | GeoTransolver | super_scarce (35 cases) | 1e-3 | 42, 43 | `forward_kwargs.global_embedding=global_data.U_inf_dir` |
| 2-3 | Transolver | super_scarce (35 cases) | 3e-3 | 42, 43 | `forward_kwargs.fx.source=global_data.U_inf_dir` |
| 4-5 | GeoTransolver | single_aoa_12 (126 cases, one angle) | 1e-3 | 42, 43 | as lanes 0-1 |
| 6-7 | Transolver | single_aoa_12 | 3e-3 | 42, 43 | as lanes 2-3 |

Learning rates are each baseline's declared choice at that rung (GeoTransolver
1e-3, the same as its reference lanes; Transolver 3e-3, its better rate at 35
cases and the rate of its fixed-angle lanes). Readout: mean over validation
cases of the surface-pressure relative L2 (the book's metric), two-seed mean.
Reference values (same protocol, physical `U_inf`): GeoTransolver 35 cases
0.381 (seeds 0.374, 0.409, 0.359); Transolver 35 cases 0.404 (0.406, 0.398,
0.409); GeoTransolver fixed angle 0.179 (0.175, 0.183); Transolver fixed
angle 0.091 (0.087, 0.096). ISLA: 0.138 and 0.036.

## Bars, in readout units

Let g = ln(baseline / ISLA) be the reference log gap: 1.02 at 35 cases for
GeoTransolver, 1.07 for Transolver; 1.61 and 0.93 at the fixed angle.

- **Supported** (the drive scale carries a material part of the deficit): the
  unit-drive two-seed mean closes at least one third of the log gap.
  GeoTransolver 35 cases <= 0.27 (ratio to ISLA <= 1.96x); Transolver 35
  cases <= 0.29; GeoTransolver fixed angle <= 0.105 (<= 2.9x); Transolver
  fixed angle <= 0.067.
- **Falsified** (null): the unit-drive mean lies within the reference seed
  band widened by one band width. GeoTransolver 35 cases >= 0.34; Transolver
  35 cases >= 0.38; GeoTransolver fixed angle >= 0.165; Transolver fixed
  angle >= 0.082.
- Between the two: partial; reported as such with the fraction of the log gap
  closed.

Each of the four arms is graded separately. The verdict that matters most
for the book is GeoTransolver at 35 cases: if supported, the headline 2.8x
contains a baseline input-scaling artifact and the verdict sentence, the
data-multiplier reading and every "2.8x" statement are rewritten with the
unit-drive baseline as the reference; if falsified, the input asymmetry is
recorded as excluded and the fair-comparison chapter gains the control.

## What this does not test

Whether a separately normalized speed channel (needed when speed varies
across cases, which it does not in either dataset) changes anything; whether
GeoTransolver's context projector would benefit from input normalization in
general (a code change to the baseline, out of scope); the DrivAerML surface
gap (GeoTransolver already leads there; a unit-drive lane on DrivAerML is a
follow-up only if the HiLift control is supported).
