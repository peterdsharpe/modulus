# Preregistration: boundary-response rank and basis-transfer gate (2D Laplace)

Written 2026-09-09 before any result was computed. Agenda direction 2 of
`research/cross_dataset_generalization/03-agenda.qmd` (topology- and
conditioning-aware boundary response learning) proposes to learn the
geometry-dependent boundary response with its known singular structure
removed. Before any learner, this study asks the nonlearned question: on
qualified 2D Laplace Dirichlet-to-Neumann (DtN) operators, how compressible
is the response, how much does exact principal-part and null-mode removal
help, and does a basis fitted on source geometries transfer to held-out
topology, narrow gaps and corners?

## Instrument

- Interior Laplace on bounded 2D domains whose boundary is a union of
  smooth closed curves, discretized by a Nyström method: Kress quadrature
  for the logarithmic single-layer kernel, trapezoid rule for the smooth
  double-layer kernel, `2N` equispaced nodes in an arclength parameterization
  of each component (the curve is resampled at equal arclength so that the
  discrete Fourier basis in the parameter is the arclength Fourier basis).
  Direct formulation: `V ψ = (½ I + K) g`, so the discrete DtN is
  `Λ = V⁻¹ (½ I + K)` mapping nodal Dirichlet data to nodal outward flux.
- Qualification: (a) unit disk, mode `cos(mθ)` must map to `m cos(mθ)` with
  relative error below `1e-8` for `m ≤ N/2` at `N = 64`; (b) every reported
  operator is recomputed at `2N` and the two must agree to `1e-6` in relative
  Frobenius norm on the common Fourier modes (`|k| ≤ N/2`), otherwise the
  case is marked unconverged and excluded from the rank tables.
- Two-body configurations are posed as the interior of a bounding circle of
  radius 4 containing two unit-scale bodies; the reported response is the
  sub-block of `Λ` mapping data on the two bodies to flux on the two bodies
  with the bounding circle grounded (that block, not the whole operator, is
  the physically relevant object; the whole operator is also saved).
- Families. Source development set: single smooth bodies (ellipses of aspect
  1, 1.5, 2, 3; star shapes `r = 1 + a cos(mθ)` with `a ∈ {0.1, 0.2, 0.3}`,
  `m ∈ {3, 5}`) and wide-gap pairs (two unit circles, gap `g/L ≥ 0.3` where
  `L` is the body diameter). Held-out development families: narrow-gap pairs
  (`g/L = 0.2, 0.1, 0.05, 0.02`), annuli (inner/outer radius `0.3, 0.6,
  0.85`), rounded squares (corner radius `ρ/L = 0.2, 0.1, 0.05, 0.025`).
  Sharp corners are excluded because the Nyström scheme is not accurate on
  non-smooth curves; the rounded family is read as the trend `ρ → 0`.
- Principal part: per component, the operator with arclength-Fourier symbol
  `2π|k|/L_c` (`L_c` the component length), i.e. the half-Laplacian on the
  boundary, applied exactly in the discrete Fourier basis. Residual
  `R = Λ − P`. Null/component modes: the per-component constant directions
  (input) and their responses (the `c × c` capacitance block) are removed
  exactly; the residual is then measured on the complement. For a single
  body the constant is exactly the DtN null space.
- Pairings: primary readout in the L2 (arclength-uniform node) pairing;
  secondary in the energy pairing `D^{-1/2} Λ D^{-1/2}` with
  `D = 2π max(|k|, 1)/L_c`, under which the principal part is the identity
  on nonconstant modes and the residual is expected to be compact.
- Rank readout: for an operator `A`, `rank_τ(A)` is the smallest `r` with
  `‖A − A_r‖_F ≤ τ ‖A‖_F` (`A_r` the rank-`r` truncated SVD). Ranks are
  reported at `τ = 1e-2` and `1e-3`, self-relative for the full operator, the
  principal-part residual and the residual after null-mode removal; the
  absolute size of each residual relative to `‖Λ‖_F` is reported alongside so
  that self-relative ranks cannot be read as absolute accuracy.
- Basis transfer: stack the source residuals (single-body self blocks, in
  arclength-Fourier coordinates of the component) and take the left and
  right singular vectors as the frozen output/input basis; apply the two-sided
  projection at rank `r` to each held-out self block and coupling block;
  compare the relative Frobenius projection error with the per-geometry
  oracle basis (the block's own SVD) at the same `r`, and with the same
  procedure applied to the full operator (no principal-part removal).
  Component-local coordinates are used because there is no common pullback
  across component counts; coupling blocks are projected on both sides with
  the same frozen bases. Excitation is the full Fourier basis up to the
  discretization plus 64 Gaussian random probes per geometry (projection
  error on random data is reported as a check that the Frobenius numbers are
  not basis artifacts).
- Resolution scaling: the rank tables are recomputed at `N` and `2N`; a rank
  that grows by more than 50% between the two at fixed tolerance is called
  resolution-bound.
- Spectral point: the fraction of `‖Λ‖_F²` carried by Fourier modes with
  `|k| > N/4`, per geometry, in both pairings.

## Bars, in readout units

- **Worth a learner** if, on every held-out family, the residual after
  principal part and null-mode removal reaches `1e-2` self-relative
  Frobenius error at a rank at most one third of `rank_{1e-2}(Λ)`, and the
  source-fitted basis achieves a projection error within `2x` of the oracle
  basis at that rank.
- **Not worth it** if `rank_{1e-2}` of that residual grows by more than 50%
  from `N` to `2N` (resolution-bound), or if the narrow-gap (`g/L ≤ 0.05`) or
  corner (`ρ/L ≤ 0.05`) families need a residual rank within 20% of the full
  operator's.
- Otherwise: partial, reported per family.

## What the outcomes change

Passes: a learner should target the null-mode-removed residual in
component-local arclength-Fourier coordinates, with explicit coupling
blocks, and the basis it learns should be graded against the oracle basis
exactly as here. Fails: the compressible object does not exist across the
intended shifts in 2D, and the 3D proposal (which can only be harder: metric
and curvature terms, no arclength Fourier basis) is stopped without a
training campaign.

## Scope

2D interior Laplace with smooth boundaries. Nothing here measures 3D
surfaces, Navier–Stokes, or turbulence; the study tests whether the
factorization the agenda relies on has the claimed structure in the simplest
setting where it can be computed exactly.
