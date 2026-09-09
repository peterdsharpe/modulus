# Preregistration: SOLVER-SPACES, does a frozen source-fitted correction space transfer to new linear systems better than an equally cheap classical one?

Written 2026-09-09 before any result was computed. Agenda direction 4
(research/cross_dataset_generalization/03-agenda.qmd, sec-direction-solver):
"transfer a correction space or preconditioner, then finish the target
solve". This is the nonlearned gate. The "learned" component is a
source-fitted basis (proper orthogonal decomposition of smoother-resistant
error modes on source geometries), not a network: if no frozen
source-derived space beats a classical coarse space of the same dimension on
held-out systems, a network trained to produce such a space has nothing to
win.

## Hypotheses

H1 (transfer). The error modes a point smoother leaves behind are smooth,
geometry-conditioned functions; a basis fitted to them on source geometries
spans the slow error of NEW geometries of the same family well enough to
beat a classical coarse space of equal dimension on slow-mode probes.

H2 (topology). H1 degrades but survives a topology shift (one hole to two
holes; a hole touching the outer wall, i.e. a narrow gap).

H3 (residual vs error). Coarse corrections that contract the residual do
not necessarily contract the error; the ratio of residual contraction to
error contraction identifies the residual-not-error failure of the
mechanisms chapter (sec-residual).

## Instrument

PDEs, linear P1 finite elements on a uniform right-triangle mesh of the
unit square with n = 48 intervals per side (2,401 grid nodes before
removal):

- Diffusion: −∇·(κ∇u) = f, κ = 1 outside and κ_c inside a circular
  inclusion; κ_c = 10 for source and within-family sets, κ_c = 100 for the
  coefficient-shift set. Dirichlet u = g on the outer square boundary with
  g(x, y) = x + 0.5 y (lifted and eliminated, so A is symmetric positive
  definite); circular holes are cut from the mesh (triangles whose
  centroids fall inside a hole are removed) and carry the natural Neumann
  condition; f = 1 for the solve targets.
- Convection–diffusion: −ε∆u + b·∇u = f with ε = 0.02, b = (1, 0.3) (mesh
  Péclet ≈ 0.5, plain Galerkin stable), same geometry and boundary
  conditions, nonsymmetric A, solved with GMRES.

Geometry sets (random centres and radii, fixed seeds):

- Source: 20 geometries, one hole (radius 0.08–0.18) and one inclusion
  disc (radius 0.10–0.20), both away from the outer wall by at least 0.12.
- Held-out (i) within family: 8 new geometries drawn the same way.
- Held-out (ii) topology shift: 8 geometries with two holes, or one hole
  whose edge lies 2 to 3 cells from the outer wall (narrow gap).
- Held-out (iii) coefficient shift: 8 within-family geometries with
  κ_c = 100.

Registration. All geometries live on the same background grid; a field on
a geometry is embedded into the grid with zeros on removed nodes, and a
grid basis is restricted to a target's active nodes and re-orthonormalized.
The registration loss of a basis on a target is reported as the fraction
of its squared Frobenius norm that falls on removed nodes.

Correction spaces, all of dimension r ∈ {16, 32} (Galerkin coarse
correction δu = −Y (YᵀAY)⁻¹ Yᵀ r):

- (a) Y_src, frozen: POD (leading left singular vectors) of the stacked
  source error snapshots e = u − u_k, where u is the exact solution for a
  random smooth right-hand side and u_k the iterate after k = 3 damped
  Jacobi sweeps (ω = 2/3) from zero; 10 right-hand sides per source
  geometry, embedded into the grid.
- (b) Y_oracle: the same construction on the TARGET's own system (not
  deployable; upper bound).
- (c) classical: piecewise-constant aggregation on a √r × √r spatial
  partition of the active nodes (Y_agg), its smoothed variant
  (I − ω D⁻¹A) Y_agg (Y_sa), and bilinear tent functions of a geometric
  coarse grid with about r interior nodes (Y_geo).
- (d) learned initial guess: the source mean solution embedded and
  restricted, used as x₀ in the complete solves (control; not a space).
- (e) no correction.

Probes on each held-out system (error directions e with r = A e):

- random: 20 Gaussian vectors;
- slow classical modes: eigenvectors 4–20 of the generalized problem
  A v = λ D v (the modes damped Jacobi leaves);
- near-null: eigenvectors 1–3 of the same problem.

Readout per probe class: the energy-norm contraction
‖e − δu‖_A / ‖e‖_A (diffusion) or the Euclidean contraction (convection),
averaged over probes and geometries, per unit work. Work is counted in
matvec equivalents: one residual matvec plus the coarse apply
(2 r N + r²)/nnz(A); it is identical for all spaces of equal r, so the
per-work comparison between spaces reduces to the contraction ratio, and
the work is reported for the no-correction baseline comparison.

Complete solves: preconditioned conjugate gradients (diffusion) or GMRES
(convection) with the additive two-level preconditioner
M⁻¹ = ω D⁻¹ + Y (YᵀAY)⁻¹ Yᵀ, stopping at relative residual 1e-8 with a cap
of 500 iterations; the A-norm (or Euclidean) error against the direct
solve is recorded at every iteration. Per-target setup is counted as r
matvecs plus 2 r² N flops for the Galerkin matrix (expressed in matvec
equivalents); the source basis construction is offline and reported
separately. Break-even right-hand-side count = (setup difference) ÷ (per-solve
work saving), reported when the saving is positive.

## Bars, in readout units

ADVANCES if all three hold:

1. On the within-family held-out set, Y_src's slow-mode error contraction
   per unit work is at least 2x better than the best classical space of
   equal dimension (contraction ratio classical ÷ Y_src ≥ 2, i.e. Y_src
   leaves at most half the slow-mode energy-norm error the classical space
   leaves).
2. On the topology-shift set the same ratio is at least 1.5.
3. Complete solves with Y_src reach 1e-8 in at least 20% fewer total
   matvec equivalents than with the best classical space, including
   per-target setup amortized over a declared 10 right-hand sides; the
   break-even count is reported.

STOPS if any holds:

- the best classical space matches or beats Y_src (ratio ≤ 1.0) on either
  held-out set;
- on any probe class the residual contraction outpaces the error
  contraction by more than 10x (the residual-not-error failure);
- the frozen space stagnates (fails to reach 1e-8 within 500 iterations)
  on any topology-shift system while the classical space converges.

Between: reported per set and per probe class as partial.

Convection–diffusion is reported separately; the energy norm is not
available for a nonsymmetric operator, contraction is measured in the
Euclidean norm, and the bars are applied as stated with that substitution
and flagged as such.

## What each outcome changes

Advances: the object a learner should produce is a geometry-conditioned
basis for smoother-resistant error modes in the background-grid
registration, judged in the A-norm; the next node is a network that maps
geometry (and κ field) to that basis, compared against the frozen POD
basis it replaces. Stops: the direction closes for this class of systems;
classical coarse spaces already capture the transferable slow modes, and
the remaining question (if any) moves to problems where no classical
coarse space is available.
