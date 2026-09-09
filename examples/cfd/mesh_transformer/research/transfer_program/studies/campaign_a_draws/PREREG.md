# Preregistration: Campaign A, data-draw variance of the HiLiftAeroML comparisons (written 2026-09-09, before any lane)

## Question

Every HiLiftAeroML comparison in the program (ISLA against unit-drive
GeoTransolver and Transolver at 35 and 210 training cases; ISLA with and
without routing measure weights) rests on one curated training draw per rung
and two or three initialization seeds. Seeds measure optimization noise; they
do not measure how much the result depends on which 35 or 210 cases an
engineer happened to purchase. This campaign measures that: three
independent, geometry-stratified training draws per rung, three seeds each,
for four arms, all scored on the shared 180-case validation set (`full_val`),
so that every ratio between architectures and the weights-off gain acquire an
interval over draws.

## Draws (build_campaign_a.py)

Source pool: `full_train` of the dataset-root manifest (1,260 cases = 180
geometries × 7 angles; the validation set `full_val` holds 180 cases at the
remaining angles of 118 of those geometries, so training and validation
share geometries as the curated splits do; only cases present on AGA are
used; every `*_test` key stays sealed and unread). Draw ids d1, d2, d3 with RNG seeds 101, 102,
103 (offset by the size, so the 35- and 210-case draws of one id are
independent). Stratification, fixed before drawing: at 35 cases, 35
distinct geometries and balanced angles (four cases at five randomly chosen
angles, three at the other five), mirroring the curated split's character
(33 geometries, 3–7 per angle) without copying it; at 210 cases, every one
of the 180 geometries contributes one case and 30 random geometries a second
at another angle (at most 2 per geometry), 21 cases per angle. Validation is `full_val`
for every draw. The draws' pairwise overlaps are recorded in
`campaign_a_summary.json`.

## Arms (frozen ladder protocol: 10,000 sampled points, 500 epochs, bf16, no augmentation, final checkpoint, lr as declared; drive at unit scale for every architecture)

| arm | model | overrides | lr |
|---|---|---|---|
| GeoTransolver, unit drive | `geotransolver_surface` | `forward_kwargs.global_embedding=global_data.U_inf_dir` | 1e-3 |
| Transolver, unit drive | `transolver_surface` | `forward_kwargs.fx.source=global_data.U_inf_dir` | 3e-3 |
| ISLA, reference | `mt2_surface` | `model.out_scalars=3 model.out_vectors=2` | 1e-3 |
| ISLA, measure weights off | `mt2_surface` | `model.out_scalars=3 model.out_vectors=2 +model.use_measure_weights=false` | 1e-3 |

Seeds 42, 43, 44; rungs 35 and 210; draws d1–d3: 4 × 2 × 3 × 3 = 72 lanes,
run ids `campA_hl_{gt,transolver,isla,islanw}_{n35,n210}_d{1,2,3}_seed{s}`,
lane table `$T/transfer/campaign_a_lanes.tsv`, launcher
`$T/transfer/campaign_a_aga.sbatch` (clone of the program's unit-drive
launcher with job names camp-a-<lane>, recipe copy `recipe_support`, code
snapshot `code_support`), evaluation launcher
`$T/transfer/campaign_a_eval_aga.sbatch`. The curated draw's existing runs
(unit-drive GeoTransolver/Transolver, ISLA, ISLA-noweights at 35 and 210
cases, including the main program's third seeds) serve as a fourth "draw"
for comparison; no curated-draw lane is duplicated here.

## Readouts

Per arm and rung: the 180-case mean pressure relative L2 of each of the 9
runs; a one-way decomposition into between-draw and within-draw (seed)
variance; per-draw means. Ratios: unit-drive GeoTransolver ÷ ISLA,
Transolver ÷ ISLA and ISLA-noweights ÷ ISLA, computed per draw from
seed-mean errors, reported as the mean over the three draws with the 90%
interval over draws (t-based, n = 3) and the per-draw values; paired
per-case counts per draw as descriptive statistics. Velocity and wall-shear
means alongside.

## Bars, in readout units

- **"ISLA behind" (per baseline, per rung) stands** if the 90% interval of
  baseline ÷ ISLA over draws lies entirely below 1.0; **"parity"** if the
  interval contains 1.0 and the total spread (max − min of the per-draw
  ratios) is under 0.10; otherwise the ordering is **draw-dependent** and
  reported as such (the strongest possible qualification of the curated
  result).
- **Weights-off gain persists** if ISLA-noweights ÷ ISLA ≤ 0.93 on all three
  draws at a rung; **fades** if any draw shows ≥ 0.97; between: reported per
  draw. (Curated draw at 35 cases: 0.88.)
- **The curated draw is typical** if each of its arm means lies within the
  range of the three draws' means for that arm; atypical draws are named.
- **Draw versus seed variance**: reported as the ratio of between-draw to
  within-draw standard deviation per arm and rung; a ratio above 2 means the
  program's two-seed protocol understates uncertainty by that factor and
  future rungs need draws, not seeds.

## What each outcome changes

If the orderings stand with intervals excluding 1.0, the book's unit-drive
statements gain a data-draw interval and lose nothing. If any becomes
draw-dependent, the corresponding chapter statements are qualified and the
program stops reporting single-draw comparisons as architecture facts. If
the weights-off gain fades on any draw, the main program's plan to make
weights-off the reference configuration is blocked until the cause is
found. Cost: 72 nodes for about one (35 cases) to two (210 cases) days.
