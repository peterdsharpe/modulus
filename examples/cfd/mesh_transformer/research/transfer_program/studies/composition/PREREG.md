# Preregistration: composition gate (nonlearned) — 2026-09-09

Written before any result was computed. Agenda direction 3 ("learn local flux
responses and solve their global compatibility", audit companion
`research/cross_dataset_generalization/03-agenda.qmd`, `sec-direction-composition`).

## Question

If local component responses (port-to-port Dirichlet-to-Neumann maps) were
supplied by a learner with a known relative error, would the assembled global
solution be accurate at useful total cost on **new global arrangements** and
**new local shapes**? This study answers the nonlearned half: exact responses,
reduced responses, and responses with injected errors, composed by an
interface solve. No learned predictor is trained.

## Hypotheses

- H1 (stability): with every local response perturbed by a fixed relative
  Frobenius error ε, the global energy-norm error of the assembled solution
  grows no faster than √N with the number of components N (errors are
  incoherent across components), and its ratio to ε stays O(1).
- H2 (reduction): transfer-optimal port modes (from the transfer operator of a
  reference two-component pair) reach 1% global energy error at a port rank
  well below the full port dimension, and do so on assemblies of components
  whose shapes were not used to construct the modes.
- H3 (coarse space): an interface Krylov solve without a coarse (one constant
  per component) correction needs an iteration count that grows with N; with it
  the count is nearly N-independent.
- Rival R1: local errors amplify superlinearly through the interface solve
  (S_II is ill conditioned and amplifies coherent errors), so a learned response
  would need unattainable local accuracy.
- Rival R2: reduction ranks that work on the library shapes fail on new shapes
  (port modes are shape specific), so the route offers no transfer.

## Instrument

- PDE: −∇·(κ∇u) = 0 on unit-square components, linear finite elements on a
  structured triangulation with n = 16 subdivisions per side (289 nodes, 64
  perimeter nodes). Optional screened variant −∇·(κ∇u) + c u = 0 (c = 1) as a
  null-space-free control.
- Component library (port geometry identical: the four unit edges, 17 nodes each):
  `plain` (κ = 1); `contrast` (4×4 blocks, log-uniform κ in [0.1, 10]);
  `hole` (central 0.4×0.4 square removed, natural boundary); `channel_wide`
  and `channel_narrow` (a vertical κ = 10⁻³ barrier of width 0.1 with an
  opening of height 0.25 or 0.0625 = one element).
  New shapes (held out from mode construction): `hole_offset`, `diag_barrier`,
  `checker8` (8×8 checkerboard contrast 0.1/10).
- Exact response: S = A_BB − A_BI A_II⁻¹ A_IB onto the perimeter nodes.
  Checks: symmetry to 1e-12 relative; eigenvalues ≥ −1e-10·‖S‖; S·1 = 0 to
  1e-10 (pure diffusion); the two-component composition law
  S_EE − S_EI S_II⁻¹ S_IE equals the merged-mesh Schur complement to 1e-10.
- Assemblies: chains 1×N and grids (2×2, 2×4, 4×4), N ∈ {2, 4, 8, 16}, random
  component choice and κ draws, 5 draws per size; outer boundary Dirichlet data
  a random smooth field (three random Fourier modes plus a linear term).
  Reference: monolithic FEM on the merged mesh.
- Readouts: relative energy-norm error ‖e‖_A/‖u‖_A and relative L2 field error
  of the recovered nodal field; interface unknown count, cond(S_II), LU
  fill (nnz of the monolithic factors) as cost proxies; PCG iterations on
  S_II with block-Jacobi ± coarse correction.
- Reduced responses (per shared edge, its 15 interior nodes reduced to r modes,
  vertices kept exact): (a) transfer-optimal modes = left singular vectors of the
  reference pair's transfer operator (outer data → interface trace) restricted
  to edge-interior nodes; (b) POD modes from interface traces of 40 random
  library assemblies (monolithic solutions); (c) rank-r truncation of each
  component's exact S (SVD), assembled and solved in full interface coordinates.
- Error injection: S ← S + ε ‖S‖_F E/‖E‖_F with E symmetric; `random`
  (Gaussian symmetric), `high` (supported on the top third of S's eigenvectors),
  `low` (bottom third excluding the null vector); ε ∈ {1e-3, 1e-2, 1e-1};
  5 draws per (size, ε, kind). Amplification = global energy error / ε.

## Bars (readout units)

- WORTH a learner if, at ε = 1e-2, the median global energy error is below 5%
  at every N ∈ {2, 4, 8, 16} and the error at N = 16 is at most 2·√(16/2) =
  5.7 times the error at N = 2 (no faster than √N); AND transfer-optimal modes
  at rank r ≤ 4 of 15 (one quarter of the edge-interior dimension, rounded up)
  reach ≤ 1% median global energy error on assemblies of NEW shapes.
- NOT WORTH IT if the ε = 1e-2 error at N = 16 exceeds 16/2 = 8 times the
  N = 2 error (superlinear amplification), or if the interface solve's dense
  cost proxy (n_I³/3 flops) exceeds the monolithic sparse factor proxy
  (nnz(L)+nnz(U) times a fill factor of 10) at every tested size, or if rank ≤ 4
  never reaches 1% on new shapes while it does on library shapes (R2).
- Between: reported as partial, per readout.
- Coarse-space ablation is reported, not a bar: iteration growth with N without
  the coarse space versus with it.
- Near-field: `channel_narrow` reported separately (rank needed and amplification)
  against `channel_wide`.

## Scope

2D scalar diffusion only. Convection breaks symmetry and positivity (Petrov–
Galerkin port spaces, no energy-norm certificate); elasticity adds six rigid
modes per component to the null space; exterior aerodynamics has no identical
connector ports (the multipole alternative of the agenda applies there).
