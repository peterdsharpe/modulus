# SOLVER-SPACES study notebook

## 2026-09-09 — Preregistered and launched

Agenda direction 4 of the cross-dataset generalization review proposes
learning a correction space or preconditioner that transfers across
geometries and letting a correct target operator finish the solve. Before any
network is trained, this study asks whether *any* frozen source-derived
correction space beats an equally cheap classical one on held-out linear
systems. If not, a network trained to produce such a space starts from
behind a free classical construction.

Instrument (PREREG.md, written before the run): P1 finite elements on the
unit square (48 intervals per side, about 2,000–2,200 unknowns after
cutting circular Neumann holes and eliminating the lifted Dirichlet
boundary), diffusion with a κ = 10 inclusion (SPD, conjugate gradients,
energy norm) and convection–diffusion at mesh Péclet 0.5 (nonsymmetric,
GMRES, Euclidean norm). Source set 20 geometries; held-out sets of 8 for
within-family shapes, topology shift (two holes, or a hole 2–3 cells from
the wall) and a 10x larger κ contrast. Correction spaces of dimension r ∈
{16, 32}: a frozen POD of smoother-resistant source errors (u minus three
damped-Jacobi sweeps, 10 smooth random right-hand sides per source
geometry, embedded into the shared background grid and restricted to each
target), a per-target oracle POD, and three classical spaces (aggregation,
smoothed aggregation, bilinear geometric coarse grid). Probes: random,
slow (generalized eigenvectors 4–20 of A v = λ D v) and near-null
(eigenvectors 1–3) error directions; readout the energy-norm error left
after one Galerkin coarse correction, plus complete two-level PCG/GMRES
solves to relative residual 1e-8 with setup amortized over ten right-hand
sides. Bars: advances only if the frozen source space leaves at most half
the slow-mode error of the best classical space within family (≥ 2x) and
at most two thirds under the topology shift (≥ 1.5x), and saves ≥ 20% of
total solve work; stops if a classical space matches or beats it on either
held-out set.

## 2026-09-09 — Result: the geometric coarse grid beats the frozen source space everywhere; the direction stops for this class

Run: `solver_spaces_probe.py` (47 s on one CPU), reduced by
`reduce_and_plot.py`; artifacts `solver_spaces_results.json`,
`solver_spaces_summary.json`, figures `fig_contraction_*.png`,
`fig_solve_*.png`, `fig_pod_*.png`.

**Diffusion, one-step error contraction on slow-mode probes** (energy-norm
error left after one coarse correction; mean over 8 held-out geometries;
lower is better):

| rank | held-out set | frozen source POD | best classical (space) | ratio classical ÷ source | per-target oracle POD (rank 10, see below) |
|---|---|---|---|---|---|
| 16 | within family | 0.963 | 0.701 (geometric, 16 dims) | **0.73** | 0.785 |
| 16 | topology shift | 0.964 | 0.706 (geometric) | **0.73** | 0.784 |
| 16 | κ shift | 0.961 | 0.732 (geometric) | **0.76** | 0.774 |
| 32 | within family | 0.748 | 0.465 (geometric, 36 dims) | **0.62** | 0.780 |
| 32 | topology shift | 0.754 | 0.470 (geometric) | **0.62** | 0.776 |
| 32 | κ shift | 0.801 | 0.527 (geometric) | **0.66** | 0.782 |

Near-null probes (eigenvectors 1–3), rank 32 within family: frozen source
0.53, oracle 0.42, aggregation 0.92, smoothed aggregation 0.86, geometric
**0.37**. Random probes: every space leaves ≥ 0.99 of the error, as a
low-dimensional coarse correction must. Residual-versus-error: for the SPD
Galerkin correction the energy-norm error left and the residual left
coincide (ratio 1.0 on every probe class), so no residual-not-error
failure occurs here; the check is retained for the nonsymmetric case.

**Diffusion, complete solves** (additive two-level PCG, unit source,
relative residual 1e-8; work in matvec equivalents including per-target
setup amortized over 10 right-hand sides):

| rank | held-out set | frozen source: iterations / work | best classical: iterations / work | no coarse space: iterations | work saving of frozen source |
|---|---|---|---|---|---|
| 16 | within family | 143 / 835 | 82 / 487 (geometric) | 179 | −72% |
| 16 | topology shift | 141 / 825 | 84 / 495 (geometric) | 179 | −67% |
| 16 | κ shift | 162 / 942 | 99 / 581 (smoothed aggregation) | 196 | −62% |
| 32 | within family | 84 / 928 | 61 / 774 (geometric) | 179 | −20% |
| 32 | topology shift | 84 / 932 | 61 / 782 (geometric) | 179 | −19% |
| 32 | κ shift | 104 / 1140 | 72 / 906 (smoothed aggregation) | 196 | −26% |

Every solve converged; no break-even right-hand-side count exists because
the frozen source space never saves work. Registration loss of the source
basis on the held-out geometries (fraction of its energy on removed nodes)
is 7–15% on average and up to 30% on two-hole geometries.

**Verdict: STOPS**, on the first stop condition. The best classical coarse
space of equal (rank 16) or near-equal (rank 32: 36 versus 32 dimensions)
size beats the frozen source-fitted space on every held-out set, on slow
and near-null probes, and in complete solves; the ratios are 0.62–0.76
against a support bar of 2.0 within family and 1.5 under topology shift.
The second bar (≥ 20% work saving) is missed by 20–72 points.

**Why, mechanically.** The source POD spectrum decays fast (normalized
singular values 0.10, 0.06, 0.04, 0.02, 0.001 at modes 2, 5, 9, 17, 32),
so the smoother-resistant source errors *are* low-dimensional; the space
they span, however, is the span of the smooth right-hand sides used to make
them (four random Fourier modes each), embedded around 20 particular hole
positions, not the operator's slow subspace on a new geometry. The
geometric coarse grid spans all smooth functions generically, which is
exactly the slow space of a discrete elliptic operator under a point
smoother, without seeing any source problem. The per-target oracle POD
confirms it: even a basis fitted on the target's own smoothed errors (rank
10, the number of right-hand sides used) leaves 0.78 of the slow-mode
error against the geometric grid's 0.47. Whatever an elliptic solver's slow
space needs, source solution snapshots are the wrong object to fit it
from; a learner trained on them inherits the same mismatch.

**Convection–diffusion, reported separately and uninformative at this
setting.** At mesh Péclet 0.5 the Jacobi-preconditioned GMRES converges in
2–3 iterations with any coarse space and 5 without, so no space can save
work, and the one-step Galerkin correction is not an orthogonal projection
for a nonsymmetric operator: it *increases* the Euclidean error on slow
probes for the frozen source (0.95–1.13) and oracle (1.2–1.4) spaces, while
aggregation stays at 0.91 and the geometric grid at 0.49–0.54 at rank 32.
The energy-norm argument does not apply; a Petrov–Galerkin coarse
correction (with Aᵀ-weighted restriction) and a higher Péclet number would
be needed for this case to say anything. Recorded as not measured.

## 2026-09-09 — What this does and does not say

Says: for discrete elliptic problems on a fixed background discretization,
the transferable slow correction space is generic smoothness, already
supplied by a classical geometric or aggregation coarse space at zero
learning cost; a frozen space fitted to source solution errors transfers
worse than that, on shapes, topology and coefficient shifts alike. Any
learned correction-space proposal for this class must beat the geometric
coarse grid, not the absence of one, and must be trained on an
operator-aware target (near-null vectors of the target operator, which
adaptive algebraic multigrid also computes without learning) rather than on
solution snapshots.

Does not say: anything about problems where no classical coarse space
exists or works (strongly anisotropic or indefinite operators, high-Péclet
convection, coupled nonlinear systems whose Jacobian changes with the
state), where the agenda's nonlinear extension lives; nor about learned
smoothers or learned prolongation *weights* on a classical coarse
hierarchy (learned multigrid), which are a different object from a frozen
correction space. Those remain the only places a learned solver component
could still earn its cost; the prerequisite is a target operator whose
classical coarse spaces demonstrably fail.
