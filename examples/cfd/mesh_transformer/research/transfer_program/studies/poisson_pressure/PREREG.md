# Preregistration: H1, pressure by constraint solve from a predicted velocity (written 2026-09-09, before any readout on predictions)

## Question

Hypothesis 1 of the classical-solver chapter: a solver enforces the pressure
constraint globally at every step, while a surrogate predicts pressure
pointwise. On the DrivAerML volume task, can the pressure be recovered from a
predicted velocity field by solving the constraint it must satisfy, and is
that pressure more accurate than the directly predicted one? ISLA query
tokens + SDF leads GeoTransolver-volume on velocity (0.0853 against 0.1334)
by more than on pressure (0.0567 against 0.0620), so if the mechanism works
the pressure gap should widen in ISLA's favour.

## The equation solved, and what is dropped

DrivAerML labels are time averages of an incompressible hybrid RANS–LES
simulation (ρ = 1 kg/m³ in the saved metadata, gauge pressure with
p_∞ = 0). The divergence of the mean momentum equation with ∇·⟨u⟩ = 0 gives

    ∆p = −ρ ∂_i ∂_j ( ⟨u_i⟩⟨u_j⟩ + τ_ij ),

where τ_ij is the total turbulent (modelled plus resolved) stress. Two
right-hand sides are solved for:

- **RHS-A (mean convection only):** S_A = −ρ ∂_i∂_j(u_i u_j), evaluated from
  the saved mean velocity; τ_ij dropped entirely.
- **RHS-B (with the eddy-viscosity closure):** S_B = S_A + ρ ∂_i∂_j
  [ν_t (∂_i u_j + ∂_j u_i)], with the saved ν_t; the resolved part of the
  stress and any deviatoric residual of the closure are dropped, and the
  isotropic part is absorbed in p by definition.

Neither is the exact equation the labels satisfy (the resolved stress is
not saved), which is why the qualification gate below is decided with the
TRUE fields first: if the true velocity does not reproduce the true pressure
through this operator on this sample, the hypothesis is not testable here,
whatever the surrogates do.

## Discretization

The sample is 10,000 scattered interior points per car (median wall
distance 0.001 L_ref, L_ref = 5 m; nearest-neighbour spacing from about
0.6 mm to 0.1 m in physical units), i.e. about 6 × 10⁻⁵ of the CFD mesh.
Gradient, Hessian and Laplacian are built meshfree by RBF-FD: polyharmonic
spline φ(r) = r³ augmented with polynomials to degree 2, on k = 40 nearest
neighbours per point (unisolvent for the 10 quadratic monomials), weights
from the saddle-point system per stencil. Boundary treatment on the point
cloud: far-field rows (the 2% of points with the largest SDF) impose
p = p_∞ = 0; near-wall rows (points with SDF below the 5% quantile, whose
stencils straddle the body's hole in the cloud) impose one of three
conditions, all recorded:

- **BC-D (qualification only):** p = true pressure at the near-wall points
  (Dirichlet from labels; not a model-free condition).
- **BC-N0:** ∂p/∂n = 0 along the saved SDF gradient (homogeneous Neumann,
  model-free).
- **BC-Nm:** ∂p/∂n = −ρ [(u·∇)u]·n from the field in use (wall-normal
  momentum without the viscous term).

Interior rows: L p = S. The sparse nonsymmetric system is solved directly
(SuperLU). Cost per car is recorded.

Amendments made while writing the code, still before any prediction was
read (2026-09-09):

- **Discrete form of RHS-A.** ∂_i∂_j(u_i u_j) = (∂_i u_j)(∂_j u_i) +
  u·∇(∇·u); the code uses S_A = −ρ (∂_i u_j)(∂_j u_i), which needs first
  derivatives only and is identical for a divergence-free field. The
  divergence of each velocity field in use is recorded so the dropped term
  is visible. RHS-B's closure term ∂_i∂_j[ν_t(∂_i u_j + ∂_j u_i)] is formed
  by two successive first-derivative applications.
- **A third formulation, M (momentum least squares).** Instead of the
  divergence form, fit the gradient of p to the momentum equation directly:
  minimise Σ_i |∇_h p(x_i) − f(x_i)|² with f_A = −ρ (u·∇)u (and f_B = f_A +
  ∇·[ν_t(∇u + ∇uᵀ)]), with p = 0 enforced at the far-field points by
  weighted rows. This is the discrete weak form of the same constraint
  (the pressure-from-velocimetry route), needs no wall condition, and is
  the more robust instrument when the point cloud is irregular. It counts as
  a model-free variant for G1.
- **Wall set.** Primary: SDF below the 5% quantile. Sensitivity: points
  whose SDF is below half their stencil radius (the stencil ball intersects
  the body). Both recorded for every variant.
- **G0 fields.** Quadratics are reproduced exactly by the degree-2
  augmentation and test nothing; the analytic fields are
  f_λ = sin(2πx/λ) sin(2πy/λ) sin(2πz/λ) at λ = 1 m and λ = 0.25 m (car
  length about 4.6 m), with ∆f_λ = −3(2π/λ)² f_λ; the 10% bars apply to
  λ = 1 m on the band SDF < 0.05 L_ref (where about 90% of the points
  lie), and all bands and both wavelengths are reported.

## Gates and bars, in readout units (all relative L2 over the 10,000 points, mean over the 48 validation cars)

- **G0, operator on analytic fields (before any prediction is read):** on
  each car's point cloud, f₁ = x² + y² − 2z² (harmonic) and f₂ = x² + y² + z²
  (∆f₂ = 6). Report the relative L2 error of ∇f and the error of ∆f₁ against
  zero (relative to the mean |∆f₂|) and of ∆f₂ against 6. **Operator
  qualifies** if the gradient error is ≤ 10% and the Laplacian errors are
  ≤ 10%; otherwise the instrument fails at this sampling and the study
  stops after reporting which stencil sizes k ∈ {25, 40, 60} were tried.
- **G1, recovery from true fields:** solve with the TRUE velocity (and TRUE
  ν_t for RHS-B), each BC variant. **Instrument qualifies for the
  hypothesis** if the recovered pressure's relative L2 against the true
  pressure is ≤ 0.10 for at least one model-free BC (N0 or Nm) with either
  RHS. If only BC-D reaches 0.10, the constraint solve can only reproduce
  pressure when given wall pressure, which is a negative for the hypothesis
  as a *prediction* mechanism but is reported. If no variant reaches 0.10,
  **STOP**: the hypothesis is untestable on the saved sample density;
  report the recovered-from-true errors, the operator tests, and the
  spacing statistics that would have to change.
- **G2, recovery from predicted fields (only if G1 passes):** for each arm
  (ISLA query tokens + SDF seed 42; GeoTransolver-volume seed 42; single
  seeds, stated), recover pressure from the PREDICTED velocity (and
  predicted ν_t) with the qualifying model-free BC and RHS, and compare
  against the true pressure. **Supported** if the recovered pressure's
  relative L2 is ≤ 0.9x the directly predicted pressure's on the same cars
  (seed-42 direct errors are read from the same artifacts; the two-seed
  references are 0.0567 and 0.0620). **Falsified** if ≥ 1.1x. Between:
  neutral. Also reported: the near-wall band (SDF < 1% of L_ref) separately,
  and the recovered pressure's error decomposed into the part explained by
  the velocity error (recover from true velocity minus recover from
  predicted velocity) versus the operator's own error (G1 residual).

## What each outcome changes

G0/G1 fail: the constraint-solve mechanism cannot be evaluated on the
saved 10,000-point samples; testing it needs either the full mesh (150M
points) or a re-sampled evaluation at a density that passes G0, which is a
data-plumbing node, not a modelling one. G2 supported: pressure should be
obtained by a solve from the surrogate's velocity rather than predicted
directly, and the interior pressure comparison in the program book gains a
solver-assisted column. G2 falsified: the velocity error, once
differentiated twice, is too large for the constraint to help at this
accuracy, which bounds the value of "global constraint by solve" for
surrogates of this quality.
