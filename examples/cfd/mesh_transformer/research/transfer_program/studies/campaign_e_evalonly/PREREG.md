# Preregistration: campaign E, density-bias probe on the unit-drive baselines (written 2026-09-09 before any evaluation)

## Question

The program book's remaining robustness claim for ISLA is the 10:1
density-bias probe: the same 48 DrivAerML validation cars sampled with a
Poisson-biased inclusion (dense at one end, sparse at the other, 10:1
expected density ratio, `drivaer_probe_biased3`) against a uniform draw
through the same two-stage 40,000 → 10,000 pipeline (`drivaer_probe_unif2`).
The ISLA similarity-gauge arm degrades by a factor 1.20 (pressure relative
L2 0.0633 uniform → 0.0762 biased, seeds 42 and 43: 1.204 and 1.203). The
audit's objection is that this was never measured on an input-matched
baseline: the GeoTransolver comparison used physical-velocity drive and the
Transolver arm did not exist. The canonical unit-drive baselines now exist
(`uw_gt_unit_lr1e3_seed{42,43,44}`, `uw_transolver_unit_lr3e3_seed{42,43}`,
the main session's WAVE-3 lanes, pressure 0.0556 and 0.0563 in the canonical
frame). Does an input-matched GeoTransolver or Transolver degrade more than
ISLA under the biased sampling?

## Instrument (eval-only, no training)

Each run is scored on both probe datasets at `sampling_resolution=10000`,
`infer_split=val`, with the exact configuration of the main session's
unit-drive evaluation launcher (`uw_eval_aga.sbatch`: model, freestream
drive override, code snapshot), from the final checkpoint in `$T/runs`.
Arms: unit-drive GeoTransolver seeds 42/43/44; unit-drive Transolver at
learning rate 3e-3 seeds 42/43; and the ISLA reference configuration
`iw_mt2_lr1e3_seed{42,43}` (the current ISLA, not only the gauge variant) so
that the ISLA number is from the same protocol and code as the baselines.
Outputs `$T/transfer/campaign_e/<run>/{unif,biased}`; launcher
`$T/transfer/campaign_e_density_aga.sbatch`, job names camp-e-*.

## Readout

Per run: pressure relative L2 on the uniform and the biased draw, and their
ratio (degradation factor). Per arm: seed mean of the factor and of both
levels. Wall shear alongside.

## Bars, in readout units

- **ISLA's density robustness holds against a fair baseline** if the
  unit-drive GeoTransolver's degradation factor is ≥ 1.32 (10% above ISLA's
  1.20; 10% is the smallest difference the program treats as a finding) and
  ISLA's own factor in this protocol stays ≤ 1.24 (within 3%, one seed
  spread of the canonical pressure numbers; the gauge arm's seed spread of
  the factor is 0.1%).
- **The claim does not hold** if GeoTransolver's factor is ≤ 1.24: the
  baseline is as density-robust as ISLA within the noise, and the book's
  claim is withdrawn as an ISLA-specific property.
- Between: reported as a ratio of factors with both seed spreads.
- Absolute levels are reported beside the factors: a model that is robust
  because it is worse everywhere (higher uniform error) is not the claim.
- Transolver is reported on the same footing; its lane has no prior number
  and sets no bar.

## What each outcome changes

Holds: the density-robustness paragraph of the program book gets a fair
comparison and survives; the mechanism question (measure weights) is next.
Does not hold: the claim moves from ISLA property to shared property of
surface transformers at this token count, and the book says so.

## Cost

Seven array tasks, two GPUs each, about 20 minutes per unit on one GB300;
under one node-hour in total.
