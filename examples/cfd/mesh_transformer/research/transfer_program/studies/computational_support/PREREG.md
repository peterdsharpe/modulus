# Preregistration: D1, computational support with passive readout (written 2026-09-09, before any lane)

## Question

The interacting query-token configuration of ISLA (query tokens + signed
distance, "QT+SDF") leads GeoTransolver-volume on interior pressure and
velocity on DrivAerML, but its prediction at a point depends on which other
points are requested with it (pressure 0.062 at 10,000 queries, 0.077 at
2,000, 0.110 at 1,000 on fixed sentinels). The audit's reading: the queries
supply a rich, geometry-aware computational state, and letting the caller's
query set define that state is an avoidable choice. Replace it with a
problem-derived SUPPORT set that interacts (deep), and decode the requested
points passively from the resulting state with the same signed-distance
inputs and the same depth. Does the accuracy survive?

Known facts designed against: the existing passive arm (4 read blocks, no
SDF scalar on the queries, 365 surface cells) is 2.6x behind on pressure;
UDRV-INT showed the interior ordering is not an input-scale artifact;
V0-SURF10K showed the interior result does not depend on the surface sample
size (365 vs 10,000 cells); the eddy-viscosity gap (1.26x) is open.

## Implementation (committed with this file)

`ISLA(support_tokens=True, query_independent=True, n_query_scalars=1,
query_mass="source_total", n_decoder_layers=K)`; forward takes
`support_points / support_normals / support_scalars` (interacting tokens
with the same invariant seeds, token-type offset and a learned total mass
equal to a learned fraction of the total source measure) and
`query_points / query_normals / query_scalars` (passive, decoded through K
read blocks that see the final slice states, anchors and a measure-invariant
log-space kernel readout of all encoder tokens). Contract tests
(`test/experimental/nn/test_isla_support.py`): exact query independence with
the support fixed; SE(3) covariance; drive degree one; measure-scale and
source-refinement invariance; gauge scale equivariance; geo_checkpoint
exactness; all parameters receive gradients. Recipe: transform
`SplitInteriorSupport` (moves the first `n_support` interior points of the
reader's deterministic per-case sample into `boundaries.support` with `sdf`
and `sdf_normals`), dataset `drivaer_ml_volume_support.yaml`, model
`isla_volume_support.yaml`.

## Arms (DrivAerML volume, 435 training cars, frozen V0 protocol: 500 epochs, lr 1e-3, bf16, no augmentation of pose beyond the recipe's, unit freestream direction for every model, two seeds 42/43)

| arm | run ids | interior tokens | decode | overrides (relative to isla_volume_support.yaml + drivaer_ml_volume_support.yaml) |
|---|---|---|---|---|
| (i) reference QT+SDF, interacting | existing v0_isla_qtsdfval_seed{42,43} | 10,000 queries interacting | in the encoder | none (numbers 0.0567 / 0.0853 / 0.1215) |
| (ii) SUPPORT-12 | v0_isla_support12_seed{42,43} | 5,000 support interacting + 5,000 passive queries | 12 read blocks with SDF | `model=isla_volume_support dataset=drivaer_ml_volume_support model.model.n_decoder_layers=12` |
| (iii) SUPPORT-4 | v0_isla_support4_seed{42,43} | same | 4 read blocks with SDF | `... model.model.n_decoder_layers=4` |
| (iv) PASSIVE-12 (optional) | v0_isla_passive12_seed{42,43} | 0 support; 5,000 passive queries | 12 read blocks with SDF | `model.model.support_tokens=false` plus `~forward_kwargs.support_points ~forward_kwargs.support_normals ~forward_kwargs.support_scalars` (or a model yaml without them) |

The reference arm scores 10,000 queries per car; arms (ii)–(iv) score the
5,000 non-support points. Both are relative L2 over the 48 validation cars
and are comparable as per-point statistics; a 10,000-support + 10,000-query
variant (`sampling_resolution=20000 n_support=10000`) is the cost-unmatched
sensitivity arm if budget allows and is not part of the bars.

Cluster deployment: copy `isla_volume_support.yaml` to
`$T/recipe/conf/model/`, `drivaer_ml_volume_support.yaml` to
`$T/recipe/datasets/`, the patched `domain_transforms.py` to
`$T/recipe/src/`, and take a fresh code snapshot `$T/code_support/physicsnemo`
from the merged branch; launcher `v0/v0_support_aga.sbatch` cloned from
`v0/v0_qt2_aga.sbatch` (job names v0sup-$i, PHYSICSNEMO_DIST_TIMEOUT_S=3600,
LOG_START guard), eval launcher cloned from `v0/v0_qt2_eval_aga.sbatch` with
the new run ids and `dataset=drivaer_ml_volume_support`; reducer arms
`isla_support12`, `isla_support4`, `isla_passive12` in
`highlift/reduce_a35_v0.py`. Before the surface part of the model yaml is
used, diff it against the cluster's `mt2_volume.yaml` so that the surface
tokens (vehicle cell centroids, normals, areas) are identical to the
reference arm's. Not submitted by this track.

## Readouts

1. 48-car relative L2 of pressure, velocity and eddy viscosity, two-seed
   mean, per arm.
2. Sentinel-query study (four cars, one checkpoint, bf16 and fp32): 1,000
   fixed sentinels, companions 1k / 2k / 5k / 10k / 40k, support fixed
   (`eval_fixed_support_skeleton.py`). Readout: max ÷ min sentinel error over
   the companion counts; the bf16 floor is the same statistic on the
   legacy passive arm's checkpoint (structurally independent), reported
   alongside.
3. Cost: step time and peak memory per GPU at 5,000 + 5,000 tokens against
   the reference arm's 0.28 s and 6.3 GiB (eager), from the training logs.
4. Held-out family: SHIFT-SUV volume data are not available to this program;
   recorded as not measured.

## Bars, in readout units

- **Supported** if SUPPORT-12's pressure ≤ 0.0584, velocity ≤ 0.0879 and
  eddy viscosity ≤ 0.1251 (each within 3% of the reference; seed spread of
  the reference under 1% on pressure and velocity) AND the sentinel
  max ÷ min over 1k–40k companions is within the bf16 floor measured on the
  passive path (structural dependence zero by the tests; numerical drift
  reported).
- **Falsified** if SUPPORT-12's pressure ≥ 0.0624 (10% worse than the
  reference, i.e. no better than GeoTransolver-volume's 0.0620).
- **Between**: partial; the SUPPORT-4 comparison then says whether decoder
  depth (ii vs iii) or interaction on a support (ii vs iv) carries the
  difference, each reported as a ratio with the reference's seed spread as
  the noise floor.

## What each outcome changes

Supported: the interior configuration becomes deployable with a stable
field (query batching immaterial), and the audit's first direction has its
first positive; the next node is the held-out-family test when a volume
target family exists, and the cost comparison decides whether the support
size can shrink. Falsified: the accuracy of the interacting arm is
inseparable from letting the requested queries interact, which the audit
flagged as the alternative reading; the deployment repair then goes through
query-count augmentation (QCOUNT, main program) rather than architecture,
and the passive route is closed at matched depth and inputs. Partial: the
depth/interaction decomposition tells which ingredient to keep.
