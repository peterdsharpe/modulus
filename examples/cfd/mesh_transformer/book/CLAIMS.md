# Claims ledger: what the program currently knows

This is the single source of truth for every quantitative claim in the
polished chapters. Chapter authors write *from* this ledger; if a chapter
needs a number that is not here, it comes from the named artifact in
`results/` or the named notebook anchor, and is added here. Anything listed
under **Do not state** is deleted from the chapters wherever it appears.

Conventions: "rel-L2" = relative L2 error of a field over the sampled points
of one case, geometric or arithmetic mean over cases as the artifact defines
(state which). "MT2/GT" = MT2 error divided by GeoTransolver error (< 1 means
MT2 more accurate). Seeds are 42/43 unless stated. Protocol P0 = the frozen
recipe: 10,000 sampled surface points per case, 500 epochs, bf16, no
augmentation, learning rate 1e-3 for MT2 and GeoTransolver (each one's best
on the full split), 3e-3 for Transolver where stated.

## Architectures and resources (chapter 2–4)

- **MeshTransformer2 (MT2)**: SE(3)-equivariant soft-slice transformer.
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
  Interior (volume) queries in the boundary→interior task are always decoded
  passively. Every HiLift and DrivAerML surface checkpoint in this book is
  the reference (interacting) configuration. Heads: scalars and vectors
  (vectors in an equivariant basis). Parameters 8.9M (8,866,484 at hidden
  192, 12 layers, 256 slices; 11.0M with query_independent). Peak training memory
  **3.6 GB** at 10,000 tokens (the relational pooling is computed as
  pool-then-project, an exact identity; forward+backward 283 ms/step on an
  RTX 4090 in bf16). Contracts verified by fp64 tests: exact rotation and
  translation covariance; query-set independence (in the query_independent
  configuration); geometric-scale equivariance when the similarity gauge is
  on. Source
  `physicsnemo/experimental/nn/mt2/model.py`, tests
  `test/experimental/nn/test_mt2.py` (33 tests).
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
  evaluated once. Parameters: MT2 8.9M vs GeoTransolver 9.0M. Memory: MT2
  3.6 GB < GeoTransolver 4.6 GB at 10k tokens;
  the trained MT2 checkpoints were produced by a less memory-efficient but
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

| train cases | GeoTransolver | Transolver (best lr) | MT2 | GT/MT2 | T/MT2 |
|---|---|---|---|---|---|
| 35 | 0.392 (0.375–0.409) | 0.402 (0.398–0.406; lr 3e-3) | 0.138 (0.135–0.141) | 2.9 | 2.9 |
| 210 | 0.159 (0.157–0.160) | 0.086 (0.083–0.089; lr 3e-3) | 0.064 (0.062–0.067) | 2.5 | 1.3 |
| 1,260 | 0.042 | — | 0.041 | 1.02 | — |

- Paired per-case: at 35 cases MT2 has lower error on 360/360 paired
  validation cases vs each baseline (median ratio 3.2, worst decile 1.8);
  at 210, 354/360 vs GeoTransolver (median 3.1), 330/360 vs Transolver
  (median 1.5); at 1,260, 258/360 vs GeoTransolver (median 1.06).
- Force coefficients (MAE relative to mean |coefficient|, 180 val cases):
  35 cases — MT2 CD 3.3% / CL 3.0%; GeoTransolver 17.1% / 11.0%;
  Transolver 11.9% / 8.1%. 210 — 1.9/1.5%, 3.2/2.5%, 2.0/1.5%.
  1,260 — MT2 1.4/0.9%, GeoTransolver 1.2/0.8%.
- Steel-man controls for GeoTransolver (all four in, none within 1.2x of
  MT2): lr 3e-3 at 35 → 0.392/0.480; 1,000 epochs at 35 → 0.373/0.429;
  lr 3e-3 at 210 → 0.220/0.290; 1,000 epochs at 210 → 0.152/0.128.
- Stratified at 35 cases: GT/MT2 between 2.2 and 4.3 at every angle of
  attack, 2.5 at the one angle (18°) absent from training, 3.0 on the 146
  validation geometries never seen in training; the baselines' error is
  flat across strata (0.33–0.47) while MT2's tracks flow difficulty
  (0.09–0.10 attached, 0.17–0.19 post-stall).
- Reference predictors (gauge pressure, same scale as: MT2 0.25, GT 0.63,
  Transolver 0.68): constant freestream pressure 1.00; dataset mean 0.94;
  per-case mean 0.93. Median per-case correlation of predicted vs true
  pressure: MT2 0.97, GeoTransolver 0.76, Transolver 0.69.
- Two-seed disagreement vs per-case error (Spearman): MT2 +0.67 at 35
  cases (baselines −0.20, −0.47); all architectures ≈ +0.6 at 210; 0.15–0.19
  for both at 1,260. Two-seed ensembles gain only 1–5%.
- Scope: a data-efficiency claim for 35–1,260 cases; not a scaling-law
  claim. Segment slopes (log error vs log cases): GeoTransolver 0.50 then
  0.74; MT2 0.43 then 0.25. Shared-floor evidence at 1,260: 70% shared
  residual between architectures (DrivAerML/HiLift noise probe), collapse
  of seed-disagreement correlation, accuracy still rising with tokens per
  case (6,500 → 10,000 tokens worth 8% for MT2).
- Mechanism (single-factor ablations at 35 cases, reference MT2 0.138):
  + raw coordinates (breaks SE(3)) 0.143 (1.04x); measure weights off
  0.122 (0.88x); similarity gauge on 0.149 (1.08x); invariant seeds
  replaced by raw [r, n, d] 0.137 (0.99x); relational routing off 0.160
  (1.16x). No single ingredient carries it. Every MT2 variant retains an
  invariant path; the baselines never had drive-relative invariants. The
  transplant test (baselines + [n·d, r̂·d, |r|/L] as per-cell inputs at 35
  cases, 2 seeds, each at its better lr) is DONE: Transolver 0.294/0.301
  (from 0.402; 1.35x better, still 2.15x behind MT2), GeoTransolver
  0.400/0.388 (from 0.392; no change). Preregistered bars: carrier if both
  ≤ 0.235; falsifier ≥ 0.30. Verdict: the invariant inputs are not the
  carrier; the advantage lies in MT2's routing/decoder structure as a whole,
  and no single ingredient has been isolated.
- Third seed (44) at 35 cases: MT2 0.1376, GeoTransolver 0.359, Transolver
  0.409 — inside the two-seed spread ×1.5 for all three (three-seed means
  0.138 / 0.381 / 0.404). Pending (state as running, with preregistered
  bars, no numbers): third seed at 210; 510-case rung; 1,260 cases at 20,000
  tokens; single-angle split vs matched mixed-angle control; unseen-geometry
  ladder.

## Accuracy at full data and robustness (chapter 6)

- HiLift full split: MT2 0.041 vs GeoTransolver 0.042 (0.98), P0.
- DrivAerML surface (435 train / 48 val cars, 10k tokens): best-vs-best
  MT2 0.0577 (lr 1e-3) vs GeoTransolver 0.0532 (lr 3e-3): MT2 1.085x
  behind. GeoTransolver at lr 1e-3: 0.0548 (0.0546/0.0550); MT2 at lr 3e-3:
  0.0633. Transolver (surface recipe) has 6.0M parameters.
- Learning-rate robustness (DrivAerML, lr ∈ {1e-3, 3e-3, 1e-2}, 2 seeds):
  GeoTransolver degrades 4–14x at 1e-2; MT2 1.55x. On HiLift full,
  GeoTransolver at 3e-3 destabilizes (0.148/0.397) where MT2 spans
  0.040–0.049.
- Sampling-density robustness (10:1 biased density, DrivAerML): MT2 with
  similarity gauge 1.20x degradation; MT2 constant gauge 13–14.5x;
  GeoTransolver 3.6x; exact-kernel MT1 about 1.8x (1.6–2.2x over three seeds). In-family accuracy with
  the gauge unchanged (0.0638/0.0629 vs 0.066 bar). Zero-shot cross-family
  transfer unchanged by the gauge.
- Pose: MT2 exactly rotation/translation covariant (fp64). GeoTransolver
  trained in the recipe's frame degrades 26–32x under random SO(3) poses;
  trained in a canonical frame it is exactly pose-invariant at 1.01x cost.
  Therefore equivariance is a contract (no frame estimation, no freestream
  needed at inference), not an in-distribution accuracy advantage on
  vehicle data.
- Query independence: in the reference configuration MT2's predictions at a
  point depend weakly on the other sampled points (~7% companion-set
  sensitivity vs 2% for GeoTransolver at a shared 5,000-cell prefix). The
  query_independent configuration (passive read blocks) makes them exactly
  independent (fp64 test) at 2.7–3.9x accuracy cost: passive decode
  0.203–0.211 vs 0.0577 in-family pressure on DrivAerML; the anchor-
  conditioned variant with a fixed interacting core 0.209/0.221 vs
  0.0697/0.0671 at matched memory (6.35 vs 6.56 GB).
- Memory/compute: MT2 3.6 GB and 283 ms/step vs GeoTransolver 4.6 GB at
  10,000 tokens (MT2 memory measured on RTX 4090 fwd+bwd bf16; the
  GeoTransolver figure is from the training run on the same protocol).

## Regime extrapolation (chapter 7)

HiLift `aoa` split (train 788 cases at AoA ≤ 12°, val 112 in-regime, test
900 at AoA ≥ 14°) and `stall` split (train 942 pre-stall, val 135, test 723
post-stall), P0, sealed test evaluated once.
Artifacts `results/hilift_sealed_test_2026-09-04.json`.

| split | GeoTransolver val → test | MT2 val → test | d = test/val (GT, MT2) |
|---|---|---|---|
| aoa | 0.0301 → 0.315 | 0.0207 → 0.244 | 10.5, 11.8 |
| stall | 0.0296 → 0.276 | 0.0207 → 0.260 | 9.3, 12.5 |

- Neither architecture generalizes across a flow-regime boundary: both
  lose ~10x; the OOD errors (0.24–0.33) are worse than MT2 reaches from 35
  in-regime cases (0.138). MT2 keeps the lower absolute OOD error on both
  axes; GeoTransolver has the smaller ratio because its in-regime floor is
  higher. Coverage of the regime, not architecture, governs this axis.
- Mechanism test DONE: MT2 with a raw-coordinate channel (breaking SE(3))
  on the stall split, 2 seeds, one sealed-test evaluation: in-regime val
  0.0206/0.0201, OOD test ≈0.27 (final numbers in the notebook), d ≈ 13.
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
| GeoTransolver-volume (queries as interacting tokens) | 0.062 | 1.0 |
| MT2 passive interior decode, SDF-gradient as query normal | 0.160 | 2.6 |
| MT2 passive interior decode, soft-assigned proxy normal | 0.193 | 3.1 |
| MT2 + 769 equivariant latent volume tokens | 0.228 | 3.7 |
| exact-kernel MT1 (double-layer decoder) | 0.496 | 8.0 |

- Token-level interaction at the query is worth 2.6x on the interior; MT2's
  deficit grows with distance from the wall (2.5x near → 3.3–5.4x far),
  i.e. missing volumetric context. Off-surface latent tokens along anchor
  normals make it worse (1.43x). The exact double-layer kernel is not a
  usable prior for separated flow (potential-flow oracle rel-L2 4.8 vs 0.77
  for predicting the case mean), and its "good" far-field pressure is a
  zero-output artefact (far-field velocity 16.7x GT).
- Product statement: GeoTransolver for interior accuracy; MT2 for exact
  contracts and query independence at a 2.6x interior price. Open designs
  (not measured): wake-aligned probe tokens; a request-independent canonical
  probe set that interacts GeoTransolver-style.

## Controlled suites (Part III) — the exact-kernel line

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

## Do not state (delete wherever found)

- That MT2 is similarity- or scale-equivariant by default (only with the
  similarity gauge on; default is a constant reference length).
- That MT2 is "measure-complete" as a general property (true only with the
  gauge on; state the measured density robustness instead).
- The "drive-degree" contract for MT2 (vacuous on unit-direction inputs).
- A 1.6x accuracy dividend from equivariance (canonicalized GeoTransolver
  is pose-invariant at 1.01x).
- That the DrivAerML MT2–GeoTransolver gap is explained by label noise.
- "Equal-or-lower cost" *as a retired claim* — it is now simply true
  (3.6 vs 4.6 GB); state the numbers.
- That no architecture generalizes across boundary-condition frequency
  (the exact-kernel decoder does).
- Locality as the carrier of cross-family transfer (refuted).
- Any "provisional" tag on the data-efficiency result (controls are in).
- Any statement that the interior gap is closable by latent volume tokens
  along anchor normals (falsified).
- Any project code names, dates, "retired", "reframed", "critic".
