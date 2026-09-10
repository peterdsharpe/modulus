# Preregistration: far-field context tokens for ISLA's interior queries (written 2026-09-10, before any lane)

## Question

Campaign D (notebook #sec-nb-campd-verdict) left ISLA's interior eddy-
viscosity error at matched parameters 18% above GeoTransolver-volume's, and
the wall-distance band decomposition (#sec-nb-bands) located the whole
deficit in the outer wake: at signed distance ≥ 0.05 L (10% of the sampled
points) ISLA hidden 344 scores 0.093 against 0.068 while the two are equal
near the wall (0.081 vs 0.080) and ISLA leads in between. Pressure and
velocity show the same far-field pattern. Does giving the encoder context
tokens in the far field remove the deficit?

## Mechanism

ISLA's interior query reads the flow through relational invariants to the
slice anchors, which are measure-weighted means of surface positions. Far
from the body every anchor is at nearly the same distance and in nearly the
same direction, so the invariants that distinguish two far-field queries
collapse toward the same values; GeoTransolver's per-point network on raw
position keeps that resolution. Two constructions add anchors where the
resolution is lost, both built covariantly from the surface and the drive,
both independent of how the surface was sampled, both flag-gated with
defaults unchanged (repository commit `ISLA: wake tokens ...`, tests
`test_context_tokens_with_query_tokens_contracts`,
`test_wake_tokens_extent_is_sampling_invariant`):

- **Wake tokens** (`wake_tokens=true`, offsets 1, 2, 4): three interacting
  tokens on the drive axis through the centering point, downstream at
  c_k × ℓ, where ℓ is the measure-weighted RMS extent of the surface along
  the drive (a body half-length; invariant to the sample by the
  Horvitz–Thompson weights). Their routing weight is a learned fraction of
  the total surface measure. Hypothesis: the wake is organized by the drive
  direction, so anchors placed along it give far-field queries geometry that
  varies where it matters (distance to and direction of the wake axis).
- **Latent volume tokens** (`latent_volume_tokens=true`, offsets 0.5, 1, 2
  slice radii along each slice anchor's normal, plus the centroid): the
  existing construction, previously restricted to the passive decode; now
  allowed with query tokens, with its weight relative to the surface measure
  in that mode. Hypothesis: off-surface context above the body helps the
  intermediate band and the near wake; it is not organized by the flow, so
  it is the control that says whether "any off-surface tokens" or
  "wake-aligned tokens" carry the effect.

## Arms (frozen V0 protocol, DrivAerML volume, 500 epochs, lr 1e-3, bf16 training, 10k cells + 10k interior points, two seeds each; exact activation recompute; hidden 344 so the comparison is at the campaign D parameter count)

| arm | run ids | overrides beyond the hidden-344 QT+SDF configuration |
|---|---|---|
| wake tokens | `v0_isla_qtsdf_h344_wake_seed{42,43}` | `+model.wake_tokens=true` |
| latent volume tokens | `v0_isla_qtsdf_h344_lvt_seed{42,43}` | `+model.latent_volume_tokens=true` |
| reference | existing `v0_isla_qtsdf_h344_seed{42,43}` | none |

Code snapshot `$T/code_wake` (this branch after the commit above);
launcher `v0/v0_wake_aga.sbatch` (job names v0wk-0..3), evaluation
`v0/v0_wake_eval_aga.sbatch` under the program's evaluation snapshot rule
(float32, `code_eval` once it carries this commit, else `code_wake`, stated
in the entry), then `band_decomposition.py` on the saved predictions.

## Readouts

1. Far-band eddy-viscosity relative L2 (SDF ≥ 0.05 L), two-seed mean, from
   the band decomposition (raw-field per-case relative L2 averaged over the
   48 cars, the units of #tbl-bands).
2. Near-wall (SDF < 0.01 L) and intermediate-band errors of all three fields,
   same units: the cost side.
3. The recipe's 48-car metric for pressure, velocity and eddy viscosity
   (float32), for the program book's table.
4. Step time and peak memory from the training logs, against the hidden-344
   reference under the same pinning setting.

## Bars, in readout units (far-band ν_t of the hidden-344 reference: 0.093 with seeds 0.099 / 0.087; GeoTransolver-volume: 0.068 with seeds 0.068 / 0.067)

- **Supported** if the arm's far-band ν_t ≤ 0.074 (within 10% of
  GeoTransolver-volume) with the near-wall band unchanged within 3%
  (ν_t ≤ 0.083, pressure ≤ 0.050, velocity ≤ 0.089).
- **Falsified** if the arm's far-band ν_t ≥ 0.085 (no better than the
  reference's two seeds).
- **Between**: reported as the fraction of the far-band gap closed,
  (0.093 − x) / (0.093 − 0.068).
- Mechanism discrimination: wake tokens supported and latent volume tokens
  not → the flow-aligned placement carries the effect; both supported →
  any far-field anchors do; neither → the deficit is not an anchor-
  resolution effect and the reading of #sec-nb-bands is wrong.
- Cost: the arms add 3 (wake) or 3 × 256 + 1 (latent volume) tokens to
  10,000 + 10,000; a step-time increase above 10% for the wake arm would be
  a surprise and is reported.

## What each outcome changes

Supported: ISLA's interior configuration gains far-field anchors as a
principled default candidate (physical construction, no density read), and
the remaining interior comparison with GeoTransolver-volume is re-run at
matched parameters. Falsified: the far-field deficit is not anchor
resolution; the next hypothesis is the readout itself (the kernel readout
radius is fixed in gauge units and may be too small far from the surface).
