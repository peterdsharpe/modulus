# Preregistration: UDRV-L, the HiLiftAeroML ladder with unit-drive baselines (2026-09-09, written before launch)

## Why

The unit-drive control (results/udrv_unit_drive_preregistration_2026-09-08.md,
results/udrv_reduction_2026-09-09.json) was supported far beyond its bar at
35 training cases: GeoTransolver fed the unit freestream direction reaches a
surface-pressure relative L2 of 0.124 (seeds 0.124, 0.125) against 0.381
with the physical velocity, and plain Transolver 0.115 (0.117, 0.113)
against 0.404; ISLA is 0.138. The baselines' small-data deficit on
HiLiftAeroML was an input-scaling artifact of the comparison protocol, and
with the artifact removed both baselines are more accurate than ISLA at 35
cases (per case: GeoTransolver lower on 130 of 180, Transolver on 152 of
180). Every other HiLiftAeroML comparison in the book (210 and 1,260 cases;
4 and 21 training geometries; the fixed angle, whose unit-drive lanes are
still training) used the physical-velocity baselines and is therefore not a
valid comparison until rerun.

## Lanes

Frozen ladder protocol (10,000 sampled points, 500 epochs, bf16, no
augmentation, final checkpoint, each split's own validation set), launcher
highlift/hl_udrv_aga.sbatch, lane table highlift/udrv_lanes.tsv rows 8-17,
the only change from the reference lanes being the drive override
(GeoTransolver `forward_kwargs.global_embedding=global_data.U_inf_dir`,
Transolver `forward_kwargs.fx.source=global_data.U_inf_dir`), learning rates
as declared for the reference lanes:

| Lanes | Model | Split | lr | Seeds | Reference (physical drive) | ISLA |
|---|---|---|---|---|---|---|
| 8-9 | GeoTransolver | scarce (210 cases) | 1e-3 | 42, 43 | 0.141 (three seeds) | 0.065 |
| 10-11 | Transolver | scarce (210 cases) | 3e-3 | 42, 43 | 0.087 | 0.065 |
| 12-13 | GeoTransolver | geometry_super_scarce (4 geometries, 40 cases) | 1e-3 | 42, 43 | 0.449 | 0.250 |
| 14-15 | GeoTransolver | geometry_scarce (21 geometries, 210 cases) | 1e-3 | 42, 43 | 0.143 | 0.102 |
| 16-17 | GeoTransolver | full (1,260 cases) | 1e-3 | 42, 43 | 0.042 | 0.041 |

Lanes 16-17 are launched after the fixed-angle lanes finish, to stay inside
the program's node budget.

## Predictions and bars

Prediction, from the 35-case result: the unit-drive baselines match or beat
ISLA at every rung. Readout per rung: two-seed mean pressure relative L2 of
the unit-drive baseline against ISLA's on the same validation cases, with
the paired per-case and per-geometry counts.

- **ISLA has a data-efficiency advantage at a rung** only if ISLA's error is
  at least 10% below the unit-drive baseline's (ratio baseline ÷ ISLA
  ≥ 1.10) and ISLA is lower on at least two thirds of the validation cases.
  Ten percent is about ten times the seed spread of these arms (under 1%)
  and is the smallest difference the book has treated as a finding.
- **Parity** if the ratio lies within 0.95-1.05.
- **Baseline ahead** if the ratio is ≤ 0.90 (as at 35 cases: 0.90 for
  GeoTransolver, 0.83 for Transolver).
- Between the bands: reported as a small difference in the stated direction.

The unseen-geometry rungs decide whether "ISLA generalizes across wing
geometries from few samples" survives: it does only if the ratio at 4 or 21
training geometries is ≥ 1.10 with the per-geometry count on ISLA's side
(≥ 12 of 18).

## What the outcomes change

If the unit-drive baselines match or beat ISLA at every rung, the book's
HiLiftAeroML chapters become a report of a protocol artifact and its
correction, ISLA's standing on surface pressure becomes "parity to behind on
both datasets", and the program's remaining ISLA-specific claims are the
contracts (exact covariance, passive decode), the interior configuration
(itself awaiting the unit-drive control of GeoTransolver-volume, UDRV-INT),
and cost. If ISLA keeps an advantage at some rung, that rung and its
mechanism become the program's only accuracy claim and are stated with the
unit-drive baseline as the reference. The 510-case rung, the 20,000-token
floor and the deflection lanes still training with physical-velocity
baselines are not evaluated as comparisons; their ISLA arms are kept as
measurements.
