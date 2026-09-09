# Boundary-response track: lab notebook

## 2026-09-09 — Setting, question, preregistration

**Setting.** Agenda direction 2 of the cross-dataset review (topology- and
conditioning-aware boundary response learning) rests on a factorization: the
boundary response of a homogeneous linear PDE is an exact, geometry-independent
principal part plus a smooth, compressible, geometry-dependent remainder, and a
learner should target the remainder. The review's own caution (Han et al.:
reusable kernels, geometry-dependent resolvents) says the remainder may not be
shared across geometries. Nothing in the repository measures either half.

**Question.** On exactly computable 2D Laplace Dirichlet-to-Neumann (DtN)
operators: (a) how much rank does the full response need, (b) how much after the
exact principal part `2π|k|/L` is removed, (c) how much after the per-component
constants are also treated exactly, and (d) does a basis fitted on source
geometries (single bodies, wide-gap pairs) transfer to held-out topology (annuli),
narrow gaps and rounded corners at equal rank? Bars are in `PREREG.md`, written
and committed (7934fb972) before any number was computed.

**Instrument.** Nyström discretization of the direct formulation
`V ψ = (½I + K) g`, Kress quadrature for the log kernel, trapezoid for the
double layer, arclength resampling so the discrete Fourier basis is the arclength
Fourier basis. Qualification: unit-disk DtN exact to `1.1e-13` on every mode up
to the Nyquist mode (N = 32 and 64); annulus `u = log r` fluxes `1.0000` and
`−2.0000` (exact `1`, `−2`); two bodies in a grounded circle with `u = x`
flux error `2.2e-13`; every case recomputed at `2N` with the common-mode
agreement recorded. One instrument repair before the study: the plain 2D log
kernel's single layer is singular for bodies of unit logarithmic capacity (the
unit disk itself), which produced garbage on the disk while the ellipse was
converged to `1e-10`; the kernel was rescaled to `−(1/2π) log(|x−y|/R₀)` with
`R₀ = 20`, exact for the interior problem because the total flux vanishes.

## 2026-09-09 — Verdict: the residual is compressible and resolution-stable; its basis does not transfer

Run: `run_study.py` (65 min CPU, `results.json`, `run.log`). 23 geometries; 22
qualified (`N` vs `2N` common-mode agreement ≤ `1e-6`); `star_a0.3_m5` is
unconverged at `2.2e-6` and is excluded from the rank tables below (its 1e-3
rank changes 111 → 58 from `N` to `2N`, exactly the symptom the qualification
exists for).

**Compression (bar part 1: passes everywhere, one borderline case).**
Ranks at 1% self-relative Frobenius error, L2 pairing, at the working
resolution; the `2N` value is identical for every qualified case (the residual
ranks are not resolution-bound; the full operator's rank is the node count and
doubles with `N`):

| Geometry | nodes | full Λ | Λ − principal | + constants removed | ‖R‖_F/‖Λ‖_F (L2) | ‖R‖_F/‖Λ‖_F (energy) |
|---|---:|---:|---:|---:|---:|---:|
| disk | 192 | 184 | exact 0 | exact 0 | 0.0000 | 0.000 |
| ellipse 1.5 / 2 / 3 | 192 | 184 | 8 / 12 / 20 | 8 / 12 / 20 | 0.001–0.003 | 0.03–0.12 |
| stars a=0.1–0.3, m=3,5 | 512 | 489 | 12–32 | 12–32 | 0.0002–0.001 | 0.02–0.07 |
| pair g/L 0.5 / 0.3 (source) | 768 | 733 | 16 / 19 | 16 / 18 | 0.001 | 0.07 / 0.09 |
| pair g/L 0.2 / 0.1 | 1024 | 977 | 22 / 29 | 21 / 28 | 0.001 / 0.002 | 0.11 / 0.18 |
| pair g/L 0.05 / 0.02 | 1536 | 1467 | 38 / 58 | 37 / 56 | 0.003 / 0.008 | 0.24 / 0.44 |
| annulus ρ 0.3 / 0.6 / 0.85 | 384 / 384 / 768 | 361 / 369 / 740 | 20 / 50 / 160 | 21 / 50 / 160 | 0.002 / 0.007 / 0.014 | 0.14 / 0.31 / 0.62 |
| square ρ/L 0.14 / 0.065 / 0.032 / 0.016 | 256 / 256 / 512 / 1024 | 245 / 245 / 489 / 977 | 16 / 24 / 32 / 39 | same | 0.001–0.002 | 0.03–0.05 |

At `1e-3` the residual ranks are 1.4–1.5x the 1% values (e.g. pair 0.02:
84; annulus 0.85: 225; square p32: 54). In the energy pairing the residual
ranks are 25–45% lower (pair 0.02: 34; annulus 0.85: 87; square p32: 31).
Narrow gaps: 2.5–4% of the full rank (bar: not within 20%). Corners: 3–7%.
The thin annulus is the hard case: 160 of 740 (22%) in L2, 87 of 762 (11%)
in the energy pairing; it was not named in the bar but is the honest worst
case, and the curvature-coupled inner/outer response of a thin shell is not
smooth in the arclength-Fourier sense. Removing the per-component constants
changes ranks by at most 2 (they are already in or near the kernel).

**Gap scaling.** Residual rank at 1% grows roughly as `(g/L)^{-0.4}`: 16 →
19 → 22 → 29 → 38 → 58 for g/L = 0.5 → 0.02 (L2). The residual norm relative
to the full operator grows faster (0.07 → 0.44 in the energy pairing), i.e.
the near-field coupling becomes a large fraction of the response and stays
low-rank. Corners: 16 → 39 for ρ/L 0.14 → 0.016, slower than the gap.

**Spectral point (mechanisms chapter, sec-spectrum).** 98.3–98.4% of
`‖Λ‖_F²` sits in input modes with `|k| > N/4` in the L2 pairing, for every
geometry; 42–75% in the energy pairing. Any value-weighted budget on the
full operator is spent on the principal part, which is known exactly. This is
why the L2 "rank relative to the full norm" of every residual is 0 at 1%: the
residual is 0.02–1.4% of the operator in that norm. The energy pairing is
the one in which the residual is a physically sized object (3–62%).

**Basis transfer (bar part 2: fails on every held-out family).** Frozen
output/input bases from the stacked source residuals (single bodies + wide
pairs, `|k| ≤ 48` arclength-Fourier coordinates), applied at each held-out
block's own 1% oracle rank:

| Pairing / object | held-out family | oracle error (median) | source-basis error (median) | ratio median / worst |
|---|---|---:|---:|---:|
| energy / residual, constants removed | gaps (16 blocks) | 0.0070 | 0.31 | 46 / 70 |
| energy / residual, constants removed | annuli (12) | 0.0086 | 0.063 | 7.0 / 33 |
| energy / residual, constants removed | corners (4) | 0.0078 | 0.25 | 34 / 61 |
| L2 / residual, constants removed | gaps / annuli / corners | 0.009 | 0.74 / 0.23 / 0.56 | 83 / 25 / 64 (worst 113 / 297 / 98) |
| L2 / full operator (no removal) | gaps / annuli / corners | 0.008 | 0.52 / 0.40 / 0.013 | 55 / 42 / 1.8 |

The bar was 2x. Random-probe errors (64 Gaussian data vectors per block)
match the Frobenius numbers to within 0.01, so this is not a basis artifact.
The only transfer that works is the full operator on corners at rank 93 of 97
modes, i.e. the near-complete Fourier basis, which is not a compression. The
compressible residual exists for every geometry, but its singular vectors are
the geometry's own: a rank-30 object whose 30 directions are different for
each body. A post-hoc readout (`posthoc_transfer_rank.py`, not preregistered)
records how much rank the frozen source basis needs to reach 1% on each
held-out block (`results_posthoc.json`, `posthoc.log`). Energy pairing,
constants-removed residual, medians over blocks: annuli need rank 30 against
an oracle 13 (max 64); corners 78 against 22 (max 96); gaps 50 against 9
(max 94), out of 97 modes. At 2x the oracle rank the source basis still has
errors 0.02–0.14, at 4x 0.002–0.03. The shared basis exists only at a rank
approaching the mode count of the coordinate system, which is no
compression at all.

**Verdict against PREREG.md.** Part 1 (compressibility, resolution
stability, gaps and corners far below 20% of the full rank) passes. Part 2
(source-fitted basis within 2x of the oracle) fails by one to two orders of
magnitude on every held-out family. By the preregistered rule the outcome is
"partial", and the informative half is the failure: a learner for this
direction cannot be a fixed basis plus geometry-conditioned coefficients (the
agenda's phrasing "geometry-conditioned coefficients that act linearly on
boundary data"); it has to predict the basis itself, block by block, from the
geometry. That is the geometry-dependent inverse the mechanisms chapter
warned about, now measured: the resolvent's low-rank part is not shared.

**Why it matters.** (1) The factorization is real and cheap to exploit
exactly: the principal part carries 98% of the operator's L2 energy and all
of its resolution dependence, so any learned boundary-response model should
have it built in, not learned, and should be trained in the energy pairing
where the residual is a sized object. (2) What is left to learn is low-rank
(10–60 directions for smooth bodies, near gaps and rounded corners, up to
160 for a thin shell) but geometry-specific, so the learning problem is
"predict a rank-r operator from the geometry", not "predict r coefficients
in a known basis". (3) The thin-annulus and narrow-gap cases are where the
residual becomes both large and higher-rank; a 3D learner's failure modes
will be the multi-element gaps, which is the HiLift slat/flap configuration.

**What 2D does not say.** No arclength-Fourier basis exists on a 3D surface;
the principal part there is the surface half-Laplacian with metric and
curvature terms and would have to be assembled from the mesh. The 2D
residual is smoother than its 3D counterpart (corner singularities are
lines, not points). Nothing here concerns Navier–Stokes or turbulence; the
test asks only whether the agenda's linear factorization has the claimed
structure, and answers: half of it.

**Next (for the coordinator, not this track).** If the direction continues,
the learner design is fixed by this result: exact principal part + a
geometry-to-operator map producing a rank-r residual in component-local
coordinates with explicit coupling blocks, graded against the per-geometry
oracle at equal rank, in the energy pairing, with thin gaps as the held-out
family. The cheap next nonlearned test is whether a *geometry-local* basis
rule (e.g. curvature-indexed rather than Fourier-indexed) transfers better
than the Fourier-coordinate basis did; that is a coordinate question, not a
learning question, and should precede any training.
