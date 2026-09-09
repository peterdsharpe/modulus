# Claims ledger: what the program currently knows

This is the single source of truth for every quantitative claim in the
polished chapters. Chapter authors write *from* this ledger; if a chapter
needs a number that is not here, it comes from the named artifact in
`results/` or the named notebook anchor, and is added here. Anything listed
under **Do not state** is deleted from the chapters wherever it appears.

Conventions: "rel-L2" = relative L2 error of a field over the sampled points
of one case, geometric or arithmetic mean over cases as the artifact defines
(state which). "ISLA/GT" = ISLA error divided by GeoTransolver error (< 1 means
ISLA more accurate). Seeds are 42/43 unless stated. Protocol P0 = the frozen
recipe: 10,000 sampled surface points per case, 500 epochs, bf16, no
augmentation, learning rate 1e-3 for ISLA and GeoTransolver (each one's best
on the full split), 3e-3 for Transolver where stated.

## Architectures and resources (chapter 2–4)

- **ISLA**: SE(3)-equivariant soft-slice transformer.
  Inputs per surface point: position, unit normal, cell area (measure
  weight); global unit freestream direction d. Seeds are invariants
  {|r|/L, r̂·d, r̂·n, n·d} relative to a gauge (centroid; reference length L,
  constant 8.0 by default or the measure-weighted RMS radius with the
  similarity gauge on). Encoder: 12 pre-LN soft-slice layers, 256 slices,
  hidden 192; each layer routes points to slices by softmax over points with
  the log-measure bias, forms equivariant anchors (weighted mean position and
  mean normal per slice) and 8 point–anchor relational invariants that refine
  routing and are pooled back. Decoder, reference configuration: the heads
  read directly from the interacting encoder tokens, so a prediction at one
  point depends (weakly, ~7% companion-set sensitivity) on which other points
  are in the sample; the OPTIONAL `query_independent=True` configuration
  replaces this with passive read blocks (queries read slices and boundary
  tokens, never write) and makes predictions at a point exactly independent
  of the other queries, at a measured 2.7–3.9x accuracy cost on DrivAerML.
  Interior (volume) queries in the boundary→interior task are decoded
  passively in the passive-decode configuration and as interacting tokens in
  the promoted query-token configuration (never write "always decoded
  passively"). Every HiLift and DrivAerML surface checkpoint in this book is
  the reference (interacting) configuration. Heads: scalars and vectors
  (vectors in an equivariant basis). Parameters 8.9M (8,866,484 at hidden
  192, 12 layers, 256 slices; 11.0M with query_independent). Peak training memory
  **1.5 GB** at 10,000 tokens with the recipe's activation recompute
  (`geo_checkpoint: true`: the per-slice invariants are rebuilt in backward;
  forward and gradients bitwise identical), forward+backward 330 ms/step on an
  RTX 4090 in bf16; storing the invariants instead is 3.5 GB at 285 ms/step.
  Ratios: 2.3x less memory for 1.16x step time (results/isla_memory_recompute_2026-09-07.json). Contracts verified by fp64 tests: exact rotation and
  translation covariance; query-set independence (in the query_independent
  configuration); geometric-scale equivariance when the similarity gauge is
  on. Source
  `physicsnemo/experimental/nn/isla/model.py`, tests
  `test/experimental/nn/test_isla.py` (33 tests).
- **GeoTransolver**: Transolver-family soft-slice transformer with a
  geometry encoder and ball-query local context (`include_local_features`
  off in this program), raw centered coordinates + normals per point,
  freestream vector as a global feature; 9.0M parameters; 12 layers, hidden
  256, 256 slices; peak training memory **4.6 GB** at 10,000 tokens.
  Not equivariant; recovers exact pose invariance when trained in a
  canonical frame (freestream → +x, principal-axis roll) at 1.01x cost.
- **Transolver**: the same attention family without the geometry encoder;
  8 layers, hidden 256, 256 slices; inputs coordinates + normals + freestream.
- **Exact-kernel MeshTransformer (MT1)**: the original architecture — one
  attention layer with a decoder that evaluates exact singular Laplace
  kernels (the double-layer member and the single-layer member) with learned
  coefficients;
  ~82k–104k parameters in the synthetic suites; larger industrial variants.
  Contracts: similarity equivariance with an intrinsic gauge; drive
  linearity when declared; parity-typed channels. Its accuracy on separated
  3D aerodynamics is governed by the failure of the potential-flow prior
  (below).
- Fair-comparison protocol: matched tokens per case, matched epochs, each
  architecture at its best learning rate chosen on the full split by
  validation before any ladder ran, two or more seeds, shared validation
  cases, paired per-case statistics where reported, sealed test splits
  evaluated once. Parameters: ISLA 8.9M vs GeoTransolver 9.0M. Memory: ISLA
  1.5 GB < GeoTransolver 4.6 GB at 10k tokens;
  the trained ISLA checkpoints were produced by a less memory-efficient but
  function-identical implementation (fp64 max difference 4e-15), so no
  result depended on the extra memory.

## HiLiftAeroML data efficiency (chapter 5) — the primary result

Dataset: 1,800 cases = 180 high-lift wing geometries (LHC-sampled flap/slat
settings) × 10 angles of attack (4°–22°). Curated nested splits
`super_scarce` (35 train) ⊂ `scarce` (210) ⊂ `medium` (510) ⊂ `full`
(1,260), all sharing one 180-case validation set and a sealed 360-case
test set. Metric: pressure rel-L2 on the shared validation set, P0.
Artifacts: `results/hilift_ladder_reduction_2026-09-03.json`,
`results/t1_transolver_control_reduction_2026-09-05.json`,
`results/hilift_paired_stats_2026-09-06.json`,
`results/hilift_stratified_35_2026-09-06.json`,
`results/hilift_reference_predictors_35_2026-09-06.json`,
`results/hilift_ensemble_uq_{35,210,1260}_2026-09-06.json`.

| train cases | GeoTransolver | Transolver (best lr) | ISLA | GT/ISLA | T/ISLA |
|---|---|---|---|---|---|
| 35 | 0.392 (0.375–0.409) | 0.402 (0.398–0.406; lr 3e-3) | 0.138 (0.135–0.141) | 2.9 | 2.9 |
| 210 | 0.141 (0.107–0.160; three seeds, GT seed-fragile) | 0.087 (0.083–0.089; lr 3e-3; three seeds) | 0.065 (0.062–0.067; three seeds) | **2.2** (was 2.5 on two seeds; never write 2.5x again) | 1.3 |
| 1,260 | 0.042 | — | 0.041 | 1.02 | — |

- Paired per-case: at 35 cases ISLA has lower error on 360/360 paired
  validation cases vs each baseline (median ratio 3.2, worst decile 1.8);
  at 210, 354/360 vs GeoTransolver (median 3.1), 330/360 vs Transolver
  (median 1.5); at 1,260, 258/360 vs GeoTransolver (median 1.06).
- Force coefficients (MAE relative to mean |coefficient|, 180 val cases):
  35 cases — ISLA CD 3.3% / CL 3.0%; GeoTransolver 17.1% / 11.0%;
  Transolver 11.9% / 8.1%. 210 — 1.9/1.5%, 3.2/2.5%, 2.0/1.5%.
  1,260 — ISLA 1.4/0.9%, GeoTransolver 1.2/0.8%.
- Steel-man controls for GeoTransolver (all four in, none within 1.2x of
  ISLA): lr 3e-3 at 35 → 0.392/0.480; 1,000 epochs at 35 → 0.373/0.429;
  lr 3e-3 at 210 → 0.220/0.290; 1,000 epochs at 210 → 0.152/0.128.
- Stratified at 35 cases: GT/ISLA between 2.2 and 4.3 at every angle of
  attack, 2.5 at the one angle (18°) absent from training, 3.0 on the 146
  validation geometries never seen in training; the baselines' error is
  flat across strata (0.33–0.47) while ISLA's tracks flow difficulty
  (0.09–0.10 attached, 0.17–0.19 post-stall).
- Reference predictors (gauge pressure, same scale as: ISLA 0.25, GT 0.63,
  Transolver 0.68): constant freestream pressure 1.00; dataset mean 0.94;
  per-case mean 0.93. Median per-case correlation of predicted vs true
  pressure: ISLA 0.97, GeoTransolver 0.76, Transolver 0.69.
- Two-seed disagreement vs per-case error (Spearman): ISLA +0.67 at 35
  cases (baselines −0.20, −0.47); all architectures ≈ +0.6 at 210; 0.15–0.19
  for both at 1,260. Two-seed ensembles gain only 1–5%.
- Scope: a data-efficiency claim for 35–1,260 cases; not a scaling-law
  claim. Segment slopes (log error vs log cases), two-seed fits: GeoTransolver
  0.50 then 0.74; ISLA 0.43 then 0.25; on the three-seed headline means
  GeoTransolver's are 0.553 and 0.675
  (results/hilift_slopes_threeseed_2026-09-08.json). Label every slope with
  its seed count. Shared-floor evidence at 1,260: 70% shared
  residual between architectures (DrivAerML/HiLift noise probe), collapse
  of seed-disagreement correlation, accuracy still rising with tokens per
  case (6,500 → 10,000 tokens worth 8% for ISLA).
- Mechanism (single-factor ablations at 35 cases, reference ISLA 0.138):
  + raw coordinates (breaks SE(3)) 0.143 (1.04x); measure weights off
  0.122 (0.88x); similarity gauge on 0.149 (1.08x); invariant seeds
  replaced by raw [r, n, d] 0.137 (0.99x); relational routing off 0.160
  (1.16x). No single ingredient carries it. Every ISLA variant retains an
  invariant path; the baselines never had drive-relative invariants. The
  transplant test (baselines + [n·d, r̂·d, |r|/L] as per-cell inputs at 35
  cases, 2 seeds, each at its better lr) is DONE: Transolver 0.294/0.301
  (from 0.402; 1.35x better, still 2.15x behind ISLA), GeoTransolver
  0.400/0.388 (from 0.392; no change). Preregistered bars: carrier if both
  ≤ 0.235; falsifier ≥ 0.30. Verdict: the invariant inputs are not the
  carrier; the advantage lies in ISLA's routing/decoder structure as a whole,
  and no single ingredient has been isolated.
- Third seed (44) at 35 cases: ISLA 0.1376, GeoTransolver 0.359, Transolver
  0.409 — inside the two-seed spread ×1.5 for all three (three-seed means
  0.138 / 0.381 / 0.404). Pending (state as running, with preregistered
  bars, no numbers): 510-case rung; 1,260 cases at 20,000
  tokens; single-angle split vs matched mixed-angle control; unseen-geometry
  ladder.

## Accuracy at full data and robustness (chapter 6)

- HiLift full split: ISLA 0.041 vs GeoTransolver 0.042 (0.98), P0.
- DrivAerML surface (435 train / 48 val cars, 10k tokens): best-vs-best
  ISLA 0.0577 (lr 1e-3) vs GeoTransolver 0.0532 (lr 3e-3): ISLA 1.085x
  behind. GeoTransolver at lr 1e-3: 0.0548 (0.0546/0.0550); ISLA at lr 3e-3:
  0.0633. Transolver (surface recipe) has 6.0M parameters.
- Learning-rate robustness (DrivAerML, lr ∈ {1e-3, 3e-3, 1e-2}, 2 seeds):
  GeoTransolver degrades 4–14x at 1e-2; ISLA 1.55x. On HiLift full,
  GeoTransolver at 3e-3 destabilizes (0.148/0.397) where ISLA spans
  0.040–0.049.
- Sampling-density robustness (10:1 biased density, DrivAerML): ISLA with
  similarity gauge 1.20x degradation; ISLA constant gauge 13–14.5x;
  GeoTransolver 3.6x; exact-kernel MT1 about 1.8x (1.6–2.2x over three seeds). In-family accuracy with
  the gauge unchanged (0.0638/0.0629 vs 0.066 bar). Zero-shot cross-family
  transfer unchanged by the gauge.
- Pose: ISLA exactly rotation/translation covariant (fp64). GeoTransolver
  trained in the recipe's frame degrades 26–32x under random SO(3) poses;
  trained in a canonical frame it is exactly pose-invariant at 1.01x cost.
  Therefore equivariance is a contract (no frame estimation, no freestream
  needed at inference), not an in-distribution accuracy advantage on
  vehicle data.
- Query independence: in the reference configuration ISLA's predictions at a
  point depend weakly on the other sampled points (~7% companion-set
  sensitivity vs 2% for GeoTransolver at a shared 5,000-cell prefix). The
  query_independent configuration (passive read blocks) makes them exactly
  independent (fp64 test) at 2.7–3.9x accuracy cost: passive decode
  0.203–0.211 vs 0.0577 in-family pressure on DrivAerML; the anchor-
  conditioned variant with a fixed interacting core 0.209/0.221 vs
  0.0697/0.0671 at matched memory (6.35 vs 6.56 GB).
- Unseen-geometry ladder (W2-F, final): 40 cases/4 geometries → ISLA 0.250
  vs GT 0.449 (1.8x, 172/180 cases); 210 cases/21 geometries → ISLA 0.102 vs
  GT 0.143 (1.40x; velocity 1.37, wall shear 1.34). Random-split comparisons:
  ISLA 0.138/0.065, GT 0.381/0.141. READ AS: ISLA's error responds to the
  number of training geometries, GT's does not. Never write "the advantage
  shrinks on unseen geometries" without the geometry-count explanation.
  Artifact results/w2_reduction_2026-09-08b.json.
- Transolver single angle (2026-09-08): 0.0872/0.0955 → 0.091 on the 18
  held-out geometries; Transolver/ISLA 2.55x, GT/Transolver 1.96x. Bands
  (≤2.0 encoder, ≥3.5 backbone) both missed → write "about two fifths of
  GeoTransolver's fixed-angle deficit is its geometry encoder, the rest the
  shared backbone"; never "the geometry encoder explains the failure".
  Artifact results/w2_reduction_2026-09-08b.json (arm D_single_aoa_12_transolver).
- Single-angle steel-man (2026-09-08): GeoTransolver at lr 3e-3 on the
  single-angle split = 0.296/0.264 (worse than its lr 1e-3 0.175/0.183); ISLA
  0.036. The 5.0x single-angle advantage is stated with GeoTransolver at its
  better rate; the learning-rate axis is closed. Artifact
  results/w2_reduction_2026-09-08.json (arm G_single_aoa_12_gt_lr3e3).
- **Per-case structure (2026-09-07 artifacts):** HiLift per-case error rank
  correlation ISLA vs GeoTransolver 0.57 (35), 0.64 (210), 0.94 (1,260)
  against seed self-consistency 0.87–0.97; DrivAerML 0.86–0.89 (= self).
  READ AS: GeoTransolver's small-data error is nearly case-independent (CV
  0.15 at 35, 0.19 at 210; 0.12 across geometries at a fixed angle) while
  ISLA's varies 3–4x more and tracks the physics; NOT "GeoTransolver fails on
  cases of its own". GeoTransolver gains nothing from having trained on a
  geometry at other angles (0.390 vs 0.392 at 35; 0.158 vs 0.159 at 210);
  ISLA gains 15% at 35. Artifacts: results/hilift_percase_correlation_2026-09-07.json,
  results/drivaer_percase_correlation_2026-09-07.json,
  results/hilift_error_drivers_2026-09-07.json,
  results/hilift_seen_vs_unseen_geometry_2026-09-07.json,
  results/hilift_single_angle_error_spread_2026-09-07.json.
- Memory/compute: ISLA 1.5 GB and 330 ms/step (recipe default, activation
  recompute; 3.5 GB and 285 ms storing activations) vs GeoTransolver 4.6 GB at
  10,000 tokens (ISLA measured on RTX 4090 fwd+bwd bf16; the GeoTransolver
  figure is from the training run on the same protocol).

## Regime extrapolation (chapter 7)

HiLift `aoa` split (train 788 cases at AoA ≤ 12°, val 112 in-regime, test
900 at AoA ≥ 14°) and `stall` split (train 942 pre-stall, val 135, test 723
post-stall), P0, sealed test evaluated once.
Artifacts `results/hilift_sealed_test_2026-09-04.json`.

| split | GeoTransolver val → test | ISLA val → test | d = test/val (GT, ISLA) |
|---|---|---|---|
| aoa | 0.0301 → 0.315 | 0.0207 → 0.244 | 10.5, 11.8 |
| stall | 0.0296 → 0.276 | 0.0207 → 0.260 | 9.3, 12.5 |

- Neither architecture generalizes across a flow-regime boundary: both
  lose ~10x; the OOD errors (0.24–0.33) are worse than ISLA reaches from 35
  in-regime cases (0.138). ISLA keeps the lower absolute OOD error on both
  axes; GeoTransolver has the smaller ratio because its in-regime floor is
  higher. Coverage of the regime, not architecture, governs this axis.
- Mechanism test DONE: ISLA with the similarity gauge ON plus a raw-coordinate
  channel (breaking SE(3)) — two differences from the constant-gauge ladder
  reference, of which the gauge did not move the in-regime floor — on the
  stall split, 2 seeds, one sealed-test evaluation per seed on the full
  723-case list: in-regime val 0.0206/0.0201; sealed test and d in
  `results/b1_raw_coordinate_discriminator_2026-09-07.json` (final values;
  d = 12.7/13.3; sealed test 0.262/0.267 vs the reference 0.261/0.259).
  Preregistered bars: raw coordinates carry the ratio if d ≤ 11.0 and OOD
  pressure < 0.245; they do not if d ≥ 12.0 or ≥ 0.255. Verdict: they do
  not; GeoTransolver's smaller ratio is not explained by its raw-coordinate
  inputs, and no mechanism for it has been isolated.

## Boundary→interior (chapter 8)

DrivAerML volume fields (pressure, velocity, nut) predicted from the
surface alone; 435/48 cars; 10k surface tokens, 10k interior query points;
500 epochs; 2 seeds. Artifacts `results/v0_sdf_band_2026-09-05.json`,
`results/a35_v0_reduction_2026-09-05.json`, notebook
`#sec-nb-v0-verdict`, `#sec-nb-v0-sdf-band`, `#sec-nb-lvt-verdict`.

| arm | interior pressure rel-L2 | vs GT |
|---|---|---|
| GeoTransolver-volume (interior points as interacting tokens; inputs: coordinates, SDF, SDF gradient, U_inf; no surface tokens; 27.6M params, 8.8 GB) | 0.062 | 1.0 |
| ISLA query tokens (interior queries as interacting tokens; SDF-gradient normal; no SDF value; 8.9M params, 6.0 GB) | 0.064 (0.0636/0.0642) | **1.03** (velocity 1.16, ν_t 1.71) |
| ISLA passive interior decode from 10,000 surface tokens, SDF-gradient as query normal (9.9M params, 4.9 GB) | 0.160 | 2.6 |
| ISLA passive interior decode, soft-assigned proxy normal | 0.193 | 3.1 |
| ISLA + 769 equivariant latent volume tokens | 0.228 | 3.7 |
| exact-kernel MT1 (double-layer decoder) | 0.496 | 8.0 |

- GeoTransolver-volume is 2.6x more accurate than ISLA's passive decode on
  interior pressure. Do NOT write "token-level interaction is worth 2.6x":
  the two arms differ in three ways at once (query interaction; the SDF
  value as input; 2.8x parameters with six-radius local features). The
  query-token ISLA arm isolates interaction: at 500 epochs 0.064 vs GT 0.062
  (1.03x), so interaction alone accounts for essentially the whole pressure
  gap (#sec-nb-qt-verdict). ISLA's
  deficit grows with distance from the wall (2.5x near → 3.3–5.4x far),
  i.e. missing volumetric context. Off-surface latent tokens along anchor
  normals make it worse (1.43x). The exact double-layer kernel is not a
  usable prior for separated flow (potential-flow oracle rel-L2 4.8 vs 0.77
  for predicting the case mean), and its "good" far-field pressure is a
  zero-output artefact (far-field velocity 16.7x GT).
- Interior data efficiency (V0-L, 54 cars): GeoTransolver-volume 0.104 vs
  ISLA-QT 0.126 interior pressure (1.21x; velocity 1.26x; ν_t 1.73x); at 435
  cars 1.03 / 1.16 / 1.71; at 109 cars 1.20 / 1.27 / 1.79 (GT 0.078, ISLA-QT
  0.094). Slopes (pressure): 54→109 GT 0.41, ISLA 0.42; 109→435 GT 0.17,
  ISLA 0.28. READ AS: constant 1.2x GT lead through 54–109 cars, GT's curve
  flattens first, parity at 435 is where the curves cross. Do NOT write
  "ISLA's interior error falls faster at every size". Artifacts
  results/v0l_reduction_2026-09-07.json (54), results/v0l_reduction_2026-09-08.json (54+109).
- Local surface-patch features on query tokens (QT-SDF-LF, 2026-09-08):
  FALSIFIED. ν_t 0.1274 (1.36x GT) vs 0.1215 without; pressure +1%, velocity
  +0.2%. Do not describe surface-patch local features as a remedy for the
  eddy-viscosity gap; the remaining candidates are query-to-query local
  aggregation and parameters (untested). Artifact results/a35_v0_reduction_2026-09-08b.json.
- Signed-distance ladder (54/109 cars): QT+SDF / GT = 1.11 / 0.83 / 1.44 at
  54 and 1.06 / 0.73 / 1.49 at 109 (pressure / velocity / ν_t); plain QT
  1.21 / 1.26 / 1.73 and 1.20 / 1.27 / 1.79. Pressure bar inconclusive
  (band 1.05–1.15), velocity closed. Pressure curves cross between 109 and
  435 cars. Artifact results/v0l_reduction_2026-09-08b.json.
- Product statement (2026-09-08): ONE architecture covers surface and
  interior. ISLA with query tokens + SDF value: interior pressure 0.0567
  (0.91x GeoTransolver-volume), velocity 0.0853 (0.64x), ν_t 0.1215 (1.29x),
  at 8.9M vs 27.6M params, 6.0 vs 8.8 GB, inputs matched (both get SDF + SDF
  gradient), exact SE(3) covariance. Query tokens without the SDF value:
  1.03 / 1.16 / 1.71. QT-SDF/QT on pressure = 0.89 = preregistered
  INCONCLUSIVE band (0.85–0.95): say so; do not claim the pressure bar was
  met. Secondary bar (QT-SDF/GT ≤ 1.1) met. ISLA passive decode = the
  configuration when query independence is contractual, at 2.6x on pressure.
  Never write "GeoTransolver for the interior" as the standing
  recommendation. Artifact results/a35_v0_reduction_2026-09-08.json.

## Controlled suites — the exact-kernel line (ARCHIVED: chapters 03/09/10/11 live unrendered in book/archive/; the reader-facing book mentions the exact-kernel design only in the interior chapter's "Why not an exact boundary-integral prior?" section and one footnote in the index)

Keep the results of chapters 07–15 that are not contradicted below, framed
as "what exact operator structure buys on controlled problems and where it
stops". Present-tense, no chronology. Key current statements:
- On 2D Laplace/screened/mixed-BC families with exact labels, the
  exact-kernel decoder generalizes across boundary-condition frequency
  content (rel-L2 0.070 on the held-out frequency band vs ≈1.0 for
  soft-slice and moment-decoder arms), extrapolates amplitude exactly under
  a declared linearity contract (undeclared linearity costs 4.5–12x), and
  its two zero-parameter exact members match eight learned kernels.
- Nonlinear suites: implicit drive-degree is diagnosed then declared;
  fragility ladders as in chapter 06 Part II.
- AirFRANS and DrivAerML MT1 results: keep the measured facts (trace
  formulation dominates the DrivAerML gain; kernel dictionary contributes
  0–8%; within-family OOD behaviour; cross-family zero-shot transfer fails
  for every tested checkpoint and is a property of training-data diversity
  for every architecture tested). Drop all campaign narration.
- Where the line stops: separated 3D aerodynamics (potential-flow oracle
  gate fails); the interior task (8x behind GeoTransolver).


## Review-driven conventions (apply everywhere; these override earlier wording)

- **Ratio convention:** every cross-architecture ratio is written as
  baseline error ÷ ISLA error, so > 1 favours ISLA. HiLift full split:
  GeoTransolver/ISLA = 1.02 (0.042/0.041). DrivAerML full: GeoTransolver/ISLA
  = 0.92 (ISLA is 1.085x behind; write "ISLA 8.5% behind"). At 35 HiLift cases
  the factor is **2.8x** (0.392/0.138 = 2.84; three-seed 2.76) — write 2.8x,
  not 2.9x.
- **Memory — state the current model, not its history:** ISLA's peak training
  memory at 10,000 tokens is **1.5 GB** (forward+backward, bf16, RTX 4090,
  recipe default with activation recompute), against GeoTransolver's 4.6 GB;
  step time 330 ms. The stored-activation alternative (3.5 GB, 285 ms) is
  named only where cost is the topic (architecture cost section, resource
  matching, accuracy-chapter cost section, limits). Passing mentions state 1.5 GB only. The fact that the reported checkpoints were
  produced by a less memory-efficient implementation of the same function
  (fp64 max difference 4e-15) appears ONCE, as one sentence in the
  resource-matching section of the baselines chapter ("the checkpoints in
  this book were trained by an implementation of the same function that
  used more memory; memory was therefore not a controlled variable in the
  comparisons, and no result depends on it"). Never "9.8 GB", never "as
  trained", never "rewrite" outside the cost derivation in chapter 2, which
  may explain WHY the pooling is done before the projection without
  narrating that it was once done the other way. The 6,500-token control is
  a token-count control; give it no memory figure.
- **No path-dependent prose in general.** The chapters describe the current
  state: no "now", "no longer", "previously", "earlier version", "used to",
  "since the fix", "the old implementation", "retired". Where a design
  choice has a reason, state the reason; do not state the alternative that
  was abandoned unless it is a measured result in its own right (an
  ablation).
- **Density robustness is configuration-specific:** the 1.20x figure is
  ISLA *with the similarity gauge*, which is not the configuration behind
  any accuracy number in the book and costs 8% at 35 cases; the
  constant-gauge configuration behind every accuracy result degrades
  13–14.5x under the 10:1 bias, worse than GeoTransolver's 3.6x. Every
  summary sentence names the configuration.
- **Query independence price:** passive decode 0.203–0.211 vs the
  interacting reference 0.0577 at lr 1e-3 → 3.5–3.7x; the anchor-conditioned
  variant 0.209/0.221 vs the 6,500-token control 0.0697/0.0671 → 3.0–3.3x.
  Write **3.0–3.7x** (not 2.7–3.9x).
- **Learning-rate protocol:** each architecture at the rate it prefers on
  the full split of the dataset in question: ISLA 1e-3 on both datasets;
  GeoTransolver 1e-3 on HiLiftAeroML and 3e-3 on DrivAerML; Transolver 3e-3
  on HiLiftAeroML. At a common 1e-3 on DrivAerML, GeoTransolver 0.0548 vs
  ISLA 0.0577 (GeoTransolver 1.05x ahead).
- **Checkpoint selection:** the final-epoch checkpoint is evaluated; no
  checkpoint is selected on validation error. The validation set was used
  for the per-architecture learning-rate choice on the full split, for
  Transolver's learning-rate choice per small-data rung, and it was reused
  by every adaptive ablation and by the narrative selection of which
  results to report; never write "used only for the learning-rate choice".
  HiLift accuracy figures are therefore validation-set figures with no
  checkpoint selection; the sealed test split was spent only on the
  regime-extrapolation and raw-coordinate evaluations.
- **Metric:** one definition (chapter 1): the recipe's relative L2 on the
  standardized field over the sampled points of a case (measure-weighted),
  arithmetic mean over cases. The gauge-pressure form ‖p̂ − p‖/‖p − p_∞‖
  appears ONLY in the reference-predictor and seed-disagreement analyses
  and must be labelled as a different scale (ISLA 0.25 there ↔ 0.138 on the
  recipe metric).
- **Paired statistics:** report the 180 case-level pairs (each case scored
  by each architecture's two-seed mean), not 360 seed-pairs; use the exact
  sign test on n = 180 and also report the stricter "ISLA wins on both seeds"
  count. Artifact: `results/hilift_paired_stats_case_2026-09-07.json`.
- **Third seed (seed 44) at 35 cases:** ISLA 0.1376 (inside the two-seed range
  0.135–0.141); GeoTransolver 0.359 (below the two-seed range 0.375–0.409);
  Transolver 0.409 (just above 0.398–0.406). Three-seed means 0.138 / 0.381 /
  0.404; ratios 2.76 / 2.93. Do not write "inside its two-seed spread" for
  the baselines.
- **Transplant test (invariant inputs to the baselines):** Transolver
  0.294/0.301 (mean 0.297) lands in the preregistered inconclusive band
  (0.235–0.30); GeoTransolver 0.400/0.388 lands in the falsifier band
  (≥ 0.30). Write: "inconclusive for Transolver (a 1.35x gain, a quarter of
  the log-gap), falsified for GeoTransolver; the invariant inputs are not
  the carrier for GeoTransolver and at most a partial contributor for
  Transolver." The test is DONE; nowhere "running".
- **DrivAerML data-efficiency ladder (must be reported with a table):**
  frozen protocol at lr 3e-3 for both architectures, 500 epochs, 10,000
  tokens, two seeds, in-family validation pressure rel-L2
  (`results/ladder_reduction.json`, keys `lad_{gt,mt2}_n{14,27,54,109,218}_seed{42,43}`,
  field `infamily`): n = 14: GT 0.265/0.273, ISLA 0.272/0.273 (GT/ISLA 0.99);
  27: 0.152/0.155 vs 0.179/0.177 (0.86); 54: 0.099/0.102 vs 0.125/0.119
  (0.82); 109: 0.071/0.072 vs 0.091/0.091 (0.78); 218: 0.060/0.060 vs
  0.076/0.074 (0.80). On DrivAerML GeoTransolver is 1.2–1.3x MORE accurate
  than ISLA from 27 cases up (ISLA's preferred rate 1e-3 is worth about 1.1x,
  which does not close this), and the two are tied at 14 cases near the
  trivial-predictor level. The HiLift advantage does not transfer to
  DrivAerML; the book states this wherever the data-efficiency claim is
  summarized.
- **AirFRANS:** the exact-kernel MeshTransformer matches the AirFRANS-paper
  baselines on velocity and is about 10x behind MARIO and 70x behind GLOBE
  in the same table. Do not write "front of the pack".
- **Slope statement (one sentence everywhere):** "a naive power-law
  extrapolation of the two curves puts GeoTransolver at 0.015 and ISLA at
  0.029 at 5,000 cases, a factor of two in GeoTransolver's favour".
- **Reference length:** all recipe coordinates are nondimensionalized by
  the dataset's per-case L_ref (5 m on DrivAerML); ISLA's constant gauge
  8.0 is in those nondimensional units (40 m physical on DrivAerML). The
  interior chapter's distance bands are in metres, converted with L_ref.
- **Resources on the interior task** (results/v0_interior_resources_2026-09-07.json,
  from each lane's train.log): GeoTransolver-volume 27.6M parameters, 8.8 GB
  peak, 10,000 interior tokens only (surface not consumed; inputs coords +
  SDF + SDF gradient + U_inf; include_local_features with six radii); ISLA
  passive decode 9.9M, 4.9 GB; ISLA + latent volume tokens 10.0M, 5.8 GB;
  exact-kernel MeshTransformer 0.76M, 4.0 GB; ISLA query tokens 8.9M, 6.0 GB.
  The earlier "GeoTransolver-volume 9.0M / ISLA 11.0M / 4–9 GB range" was
  wrong (9.0M is the surface configuration) and must not reappear. Write
  "not resource-matched and not input-matched".
- **Status language:** polished chapters say "not yet measured" with a
  pointer to @sec-program-status; never "running", "in progress", "lanes".
- **Transolver on the controlled suites:** any "27x-larger Transolver"
  comparison states the training budget (3,000 updates) and that Transolver
  is still improving at 30,000 updates (0.119), so the matched-budget gap
  is an upper bound on the converged gap. In readout units at 3,000 updates
  on 2D Laplace: 16x against native-capacity Transolver (0.608 vs 0.0375),
  21x against matched-capacity Transolver (0.769 vs 0.0375); 7–9x in the
  fixed-dataset regime (0.126–0.136 vs 0.016–0.017); 34–78x on the
  screened and 3D suites.
- **Datasets to define in chapter 1:** SHIFT-SUV (a second vehicle dataset
  whose two rear-end families, estate and fastback, are the zero-shot
  transfer targets: DrivAerML-only models → estate; DrivAerML+estate models
  → fastback); DrivAerML 484 variants → 435 train / 48 val / 1 excluded
  (state the exclusion).

## Do not state (delete wherever found)

- That ISLA is similarity- or scale-equivariant by default (only with the
  similarity gauge on; default is a constant reference length).
- That ISLA is "measure-complete" as a general property (true only with the
  gauge on; state the measured density robustness instead).
- The "drive-degree" contract for ISLA (vacuous on unit-direction inputs).
- A 1.6x accuracy dividend from equivariance (canonicalized GeoTransolver
  is pose-invariant at 1.01x).
- That the DrivAerML ISLA–GeoTransolver gap is explained by label noise.
- "Equal-or-lower cost" as a property of the reported results (they were
  trained at 9.8 GB); state both memory numbers as above.
- That no architecture generalizes across boundary-condition frequency
  (the exact-kernel decoder does).
- Locality as the carrier of cross-family transfer (refuted).
- Any "provisional" tag on the data-efficiency result (controls are in).
- "2.9x" (write 2.8x); "2.7–3.9x" (write 3.0–3.7x); "360 pairs"; "trains in
  less memory" without the as-trained figure; "most density-robust" without
  naming the gauge configuration.
- Any statement that the interior gap is closable by latent volume tokens
  along anchor normals (falsified).
- Any project code names, dates, "retired", "reframed", "critic".


## Scope decision 2026-09-07

The exact-kernel MeshTransformer is retired as a product candidate for 3D
aerodynamics and removed from the reader-facing book (criterion: keep only
what aids understanding of the future flagship). Its chapters are archived
under book/archive/; its results stay in the lab notebook. The book has one
architecture (ISLA, code class `ISLA`) and two
baselines (GeoTransolver, Transolver) on DrivAerML and HiLiftAeroML, with
SHIFT-SUV as zero-shot target. Do not write "MT1" in polished chapters;
say "the exact-kernel MeshTransformer, an earlier design of this program"
where the interior chapter needs it.

## Program review 2026-09-08: the verdict and how to state it

- **One verdict sentence, stated once in the index (@sec-verdict) and
  referenced everywhere else:** "ISLA is a large improvement over
  GeoTransolver in one regime, scarce training data on a family of
  multi-element wing geometries; elsewhere it ranges from parity to 1.3x
  behind, and it is not a new capability." (Superseded wording, never
  reuse: "and its equal everywhere else".) Chapters support or bound clauses
  of it; no chapter offers a competing verdict paragraph.
- **Calibration words:** "a large improvement" (not "groundbreaking", not
  "a little better") for the HiLift small-data regime; "parity" at full
  data on HiLift; "behind" on DrivAerML (surface 1.2–1.3x below full data,
  8.5% at full data; interior eddy viscosity 1.29x); "no new capability" because regime
  extrapolation and cross-family transfer fail for every architecture and
  the advantage vanishes with data.
- **Data multiplier (the horizontal reading; state alongside every ratio
  headline):** GeoTransolver needs about 6x as many training cases as ISLA
  to reach ISLA's 35-case accuracy (0.141 at 210 vs 0.138 at 35, both
  measured); plain Transolver about 2x at the 0.09 error level (ISLA's case
  count interpolated on its own 35→210 power-law segment, slope 0.42:
  about 106 cases); 1x at 1,260. Table `tbl-data-multiplier` in chapter 5
  computes these live. Never write the Transolver multiplier as measured.
- **Both margins, always:** every statement of the small-data advantage
  names the GeoTransolver margin and the Transolver margin at the same rung
  (2.8x/2.8x at 35; 2.2x/1.3x at 210). Never headline 2.8x at 210.
- **Mechanism status wording:** "located at the level of the dataset
  (parametric multi-element geometry variation), not at the level of the
  surface or of an architectural ingredient". The small-feature hypothesis
  (slat and flap gaps) is STATED AND NOT YET MEASURED; preregistered in
  @sec-nb-hlreg-prereg with bars: supported if near-gap-band GT/ISLA ratio
  ≥ 1.5x the rest-of-surface ratio at 35 cases; falsified if uniform within
  1.15x. Never write it as a finding until the verdict entry exists.
- **Chapter structure:** regime extrapolation is a section of chapter 6
  (`#sec-regime-extrapolation`, `#sec-ood-mechanism`), not a chapter; the
  boundary-condition-content paragraph that belonged to the exact-kernel
  line is gone. The uncertainty-signal section is one paragraph. The
  query-independence section is one paragraph plus its numbers.
- **Interior recommendation:** one architecture covers both tasks;
  GeoTransolver-volume "remains the choice only where eddy viscosity is the
  target field or the training set is small and pressure is the target".
  The wake-aligned probe set and the canonical probe lattice are no longer
  named as roads; the open interior item is the eddy-viscosity gap (width
  control running; query-to-query local aggregation untested).
- **Status chapter:** two architectures plus the Transolver control; the
  exact-kernel MeshTransformer appears only as tree history; the organizing
  open question is the mechanism of the HiLift advantage, not the
  cross-family transfer gap.
- **Do not state:** density robustness in the verdict paragraph (it belongs
  to the similarity-gauge configuration; one sentence in chapter 6 and the
  "does not claim" list); SHIFT-SUV in the thesis sentence (it stays defined
  in chapter 1 and used in the frontier table).

## Audit response 2026-09-08

Rules from the independent audit, verified in-session by the coordinator
(notebook entry @sec-nb-audit-2026-09-08). Each overrides earlier wording.

- **Memory: name both instruments whenever memory is compared.** ISLA's
  1.5 GB (and the 3.5 GB stored-activation figure) is `max_memory_allocated`
  minus resident parameter/optimizer memory on an RTX 4090
  (results/isla_memory_recompute_2026-09-07.json). GeoTransolver's 4.6 GB and
  every "peak memory" value in the interior resources artifact
  (results/v0_interior_resources_2026-09-07.json: 8.8 / 6.0 / 4.9 / 5.8 /
  4.0 GB) are the training logger's `torch.cuda.memory_reserved()`
  (train.py:612), a different and larger quantity. The matched measurement exists
  (results/matched_memory_isla_gt_2026-09-08.json, one GB300, bf16, batch 1, 10,000
  tokens, AdamW step; total peak allocated): GeoTransolver 4.1 GiB and 65 ms
  per step; ISLA current code with recompute 1.6 GiB, without 3.4 GiB, and
  the code that trained every checkpoint in the book 9.3 GiB, all at about
  246 ms per step. Canonical wording: "on one GPU and one instrument, ISLA's
  current code peaks at 1.6 GiB against GeoTransolver's 4.1 GiB; the ISLA
  code that trained the checkpoints peaked at 9.3 GiB; ISLA's step is 3.7x
  slower". Never "3x less memory" without saying which ISLA code; never
  present memory as a settled ISLA advantage in the verdict table (row reads
  "Mixed"). No accuracy number was produced at the smaller footprint. The
  within-model 2.28x recompute reduction stands. Parameter counts are
  unaffected.
- **Width control (QT-SDF-h256).** GeoTransolver-volume's blocks run at an
  effective width of 448 (256 + six 32-channel local features,
  geotransolver.py:542). ISLA at hidden 256 (~15.7M parameters) is therefore
  a *width increase*, still narrower and smaller than the 27.6M comparator:
  write "width control" or "a width increase", never "matched-width control".
  A success shows one width increase helps; a null cannot refute capacity.
  (Status and verdict wording of that node belong to its own entry.)
- **"Held out per checkpoint" vs "program-level confirmation".** The pinned
  HiLift manifest (revision bbec30bcfc6103309c1375c5228b3ad0a586bfaf) puts
  179 of full_test's 360 cases, 182 of geometry_test's and 182 of
  deflection_test's inside the already-evaluated aoa_test/stall_test sets
  (results/hilift_sealed_test_2026-09-04.json), which between them cover all
  180 geometries; geometry_test also shares 37 and deflection_test 40 cases
  with full_val. No HiLift geometry is unexposed at the program level. A
  split is "held out per checkpoint" (no training leakage) unless it has
  been used by no training, tuning, diagnostic or prior test evaluation in
  the program, in which case it is a "program-level confirmation". Removing
  previously evaluated cases post hoc defines a new selected population and
  is never called a replacement confirmatory test. full_val's 180 cases span
  118 geometries; geometry_val's 180 cases span 18 geometries at ten angles.
  Artifact results/hilift_exposure_ledger_2026-09-08.json.
- **Geometry is the independent unit on HiLift unseen-geometry claims.** On
  the geometry ladder (W2-F) and the single-angle split the 180 validation
  cases are 18 geometries × 10 angles; a paired sign test on 180 cases
  overstates the evidence, and the geometry-level test has n = 18 (smallest
  two-sided p 7.6e-6). Report case-win counts as descriptive alongside the
  geometry-level statistic (results/hilift_paired_stats_geometry_2026-09-08.json).
  On the random split the 180 cases represent 118 geometries; the 180/180
  win at 35 cases survives any clustering. All seeds at a rung share one
  training subset, so seed spread does not measure training-set selection
  variance; the data-multiplier reading (GeoTransolver at 210 ≈ ISLA at 35)
  is a measured horizontal comparison, not a minimum CFD-run count under
  optimized sampling.
- **Verdict wording:** "ISLA is a large improvement over GeoTransolver in one
  regime, scarce training data on a family of multi-element wing geometries;
  elsewhere it ranges from parity to 1.3x behind, and it is not a new
  capability." One sentence, one place (@sec-verdict).
- **Baseline input scale.** ISLA's recipe consumes the unit freestream
  direction (isla_surface.yaml:32); GeoTransolver (geotransolver_surface.yaml:41)
  and Transolver (transolver_surface.yaml:34) consume the physical freestream
  velocity, 2679.5 in/s on HiLiftAeroML and 38.9 m/s on DrivAerML, which the
  pipelines do not normalize. GeoTransolver's global-context projector,
  untrained, at production dimensions, collapses from 191–232 effective
  occupied slices per head at drive magnitude 1 to 1.0–3.8 at 38.9 and
  1.00–1.02 at 2679.5 (max context magnitude 0.66 → 22.7 → 1530;
  results/geotransolver_drive_conditioning_2026-09-08.json). Write this as a
  conditioning difference, never as an explanation of the trained gap, until
  the training control (node UDRV, @sec-nb-udrv-prereg) reads out.
- **Query-token measure convention.** In the query-token configuration the
  query weight is exp(qt_logw) × geometric mean of source weights
  (model.py ~756); duplicating every source token at half weight leaves the
  discrete source measure unchanged but changes outputs by 3.7% (query
  tokens) and 9.1% (query tokens + signed distance) in a float64 probe on
  small random models (plain surface model 8e-16). Source refinement, query
  count, query distribution and chunking can therefore change the predicted
  field; the uniform-weight-scaling contract test cannot catch it. Trained
  checkpoints keep this convention. Their dependence on companion-query
  count is measured (results/isla_qt_companion_dependence_2026-09-08.json):
  pressure error at 1,000 fixed points 0.110 alone, 0.077 with 1,000
  companions, 0.062 with 9,000 (training count), 0.056 with 39,000; passive
  configuration exactly 0 in fp32. Canonical wording: "the interior
  advantage holds at or above the training query count of 10,000 and is
  consumed below it". Never describe the query-token configuration as
  measure-refinement invariant or as query-count invariant.
- **Interior surface tokens (node V0-SURF10K):** every ISLA interior arm
  (passive, latent, query tokens, query tokens + SDF, ladders, h256) received
  about 365 vehicle surface cells, not 10,000: the volume recipe applied a
  10,000-cell and then a 10,000-point subsample to the boundary and the
  point cut keeps only fully retained cells. Never write "10,000 surface
  tokens" for the interior arms; write "about 365 surface cells (the
  recipe's nominal 10,000; @sec-interior-task)". GeoTransolver-volume reads
  no boundary and is unaffected. The reader option `boundary_subsample:
  cells` fixes the pipeline; no interior number in the book was produced
  with it.
- **Composition failures (node QMASS):** similarity gauge + surface local
  features is not scale-equivariant (up to 0.79 output change under a 2.7x
  rescale; unnormalized patch log-mass); passive queries + boundary scalars /
  scale conditioning / raw seed features give shape errors when source and
  query counts differ; passive queries + odd vector head collide on normals.
  The plain reference surface configuration is unaffected. State these where
  the optional channels are listed as contracts.
- **First-moment limitation:** two non-congruent five-component arrangements
  with zero transverse first moment about the drive give identical ISLA
  outputs to 4e-15 (default and similarity gauge, two seeds); the
  local-distance channel separates them (0.039–0.063), a second-moment
  contraction distinguishes them (2.41 vs 1.99)
  (results/isla_first_moment_collision_2026-09-08.json). Anchor collapse does
  NOT require rotational symmetry about the drive. Write "not established as
  the cause on cars" for the relationship to any benchmark deficit; never
  "refuted on cars". A documented representation limitation, not a
  demonstrated cause of any measured deficit (benchmark: node MOM2,
  @sec-nb-mom2-prereg).
- **Mechanism claims, narrowed.** (a) Random-split and geometry-ladder
  validation sets share 20 of 180 cases, so 0.138 vs 0.250 and 0.065 vs
  0.102 change both training composition and evaluation population: write
  "supports a geometry-generalization advantage", never "only geometry
  count". (b) Plain Transolver differs from GeoTransolver in depth (8 vs 12
  layers), parameters (6M vs 9M), drive injection and learning rate; the
  ~2/5 encoder / ~3/5 backbone split is a numerical decomposition of the
  log-gap, not a causal attribution to the geometry encoder. (c) Passive ISLA
  decodes through a separate four-block read decoder while query-token ISLA
  routes queries through twelve encoder blocks with token-type and weight
  parameters, and the SDF scalar is unavailable in passive mode: the 2.6x is
  the cost of the complete passive configuration, not a fundamental price of
  query independence (node PASSIVE2 holds query depth and readout fixed and
  disables writes). (d) HLREG locates output error, not input information: a
  uniform ratio does not falsify a gap-information mechanism; grade it as
  localization.
- **Interior interpretation.** Of the passive-SDF arm's excess pressure
  squared error over GeoTransolver-volume, 94.6% lies within the first
  wall-distance band (below 0.01 nondimensional, about 5 cm), 93.7% for
  velocity, over 99% within 2 m (results/v0_interior_excess_sse_2026-09-08.json);
  fractions of the mesh-sampled error, not volume integrals. The far-field
  ratio growth is real but identifies no missing mechanism; never write
  "missing volumetric context" as a finding. Eddy-viscosity rel-L2 is
  computed on the normalized target (ν_t − 4.8e-4)/9.4e-4 before inverse
  normalization (infer.py:590; drivaer_ml_volume.yaml:62): 0.122 is not a
  12.2% physical error; orderings under the frozen metric stand, ratios of
  means need not survive a different centering. DrivAerML's closure is hybrid
  RANS/LES with a grid/filter-length-dependent modeled viscosity, so
  query-neighbour features could recover discretization information: a
  hypothesis, not established.
- **Forces are against sampled-cell integrals until FORCE-REF.** The recipe
  integrates predictions and labels on the same 10,000 sampled cells
  (forces.py:296); the 3% (35-case) and 0.6% (fixed-angle lift) figures are
  errors against the sampled-cell label integral, not the full-surface
  force, and shared quadrature can cancel sampling error. Say so wherever a
  force error is quoted. HiLift dataset revision used:
  bbec30bcfc6103309c1375c5228b3ad0a586bfaf (six force-monitor means replaced
  by surface-field integrations in that revision).
- **Equivariance wording:** "exact by algebraic construction; the
  finite-precision residual on the deployed bf16 path is a measured
  quantity" (CPU autocast probe: 1.2–1.3% scalar rotation residual in bf16
  vs 6e-7 in float32 and 1e-15 in float64; GPU trained-path residual not yet
  measured). Never bare "exact" for the deployed path.
- **Symmetry argument (chapter 1):** homogeneous constant-coefficient
  equations do not imply isotropy or scale invariance (anisotropic
  diffusion, screened Laplace); state the symmetry of the actual PDE,
  boundary data, coefficients and nondimensional parameters.
- **Baselines:** "strongest baseline tested under this protocol", never
  "strongest available mainstream baseline". Candidate nodes: AB-UPT
  (anchored neural-field decoder; TMLR, arXiv 2502.09692), AB-GATr (arXiv
  2605.18816), regularized family adaptation (arXiv 2605.27968),
  Transolver++ (the template's plus flag; flipping it is not reproducing the
  paper).
- **Reader-facing numbers:** "146 cases on 96 geometries" (not "146
  never-seen geometries"; results/hilift_seen_vs_unseen_geometry_2026-09-07.json);
  the single-angle split trains on 126 geometries and validates on 18.

- **Width control result (QT-SDF-h256, 2026-09-08).** ν_t 0.118 (1.26x GT-volume, at
  the 1.25 falsifier), pressure 0.055 (0.88x), velocity 0.084 (0.63x); 2–4% better
  than width 192 on all three. Write "a width increase to 256 does not close the
  eddy-viscosity gap; capacity is not excluded". Never "parameters do not matter".
  Artifact results/a35_v0_reduction_2026-09-08c.json. Like every ISLA interior arm:
  ~365 surface cells.

- **MOM2 (2026-09-08).** The first-moment collision is removable by either cheap
  channel: second-moment (`second_moment_features`) and local-distance features both
  recover the free azimuth (R² ≥ 0.996). Write "removable at known cost, not
  established as a cause of any benchmark gap". The preregistered family-label probe
  was ill-posed (exchangeable labels); the verdict uses the azimuth probes and says so.
  Second-moment separations are ~20x smaller than local-feature separations at radii
  (0.2, 0.4) but linearly decodable; do not write "weaker channel" without that qualifier.

- **HLREG result (2026-09-08).** Facing-surface-distance bands (not connected components:
  HiLift boundary meshes are single bodies). GeoTransolver/ISLA ratio near gaps 2.01 (35) /
  1.87 (210) vs open surfaces 3.00 / 2.57; near ÷ rest 0.72 / 0.78. Write "the error
  advantage is smallest in the gap regions and largest on open surfaces"; never "ISLA does
  not use gap information" (localization only). ISLA's own squared error: 59% (35) / 71%
  (210) in the near band, which holds ~15% of points. HiLift stored cell normals point
  INWARD (sign rule chose −1 on 360/360 cases): a data fact, not a claim about the models.

- **MOM2-T (2026-09-09).** Second-moment channel on DrivAerML surface at 435 cars:
  pressure 0.0583 vs 0.0577 (+0.9%). Write "free at 435 cars on pressure; wall shear not
  measured". Never "improves" (it does not) and never "helps a benchmark".

- **V0-SURF10K (2026-09-09).** QT+SDF with 10,000 vehicle cells: 0.0572 / 0.0848 / 0.1187
  vs 365-cell arm 0.0567 / 0.0853 / 0.1215 — null (all within 3%). Write "the interior
  result does not depend measurably on the surface sample size (365 vs 10,000 cells)";
  never "ISLA ignores the surface" (a zero-surface control was not run). The 365-cell
  numbers remain the chapter's headline numbers; the 10k numbers are the confirmation.
