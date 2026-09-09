# Composition track notebook

## 2026-09-09 — Nonlearned composition gate: verdict

**Setting.** Agenda direction 3 (learn local flux responses, solve their global
compatibility). Preregistration in `PREREG.md`; code `composition_study.py`
(run time 1064 s on one CPU, `results.json`), figures from `make_figures.py`,
tables from `summarize.py`.

**Question.** Would a learned port response, supplied with a known relative
error, give an accurate assembled solution on new arrangements and new
component shapes, and does port reduction compress the interface?

**Instrument.** 2D diffusion −∇·(κ∇u) = 0, P1 finite elements, unit-square
components with 16 subdivisions (64 perimeter nodes, 15 interior nodes per
shared edge), five library shapes (`plain`, `contrast`, `hole`, `channel_wide`,
`channel_narrow`) and three held-out shapes (`hole_offset`, `diag_barrier`,
`checker8`). Chains 1×N and grids (2×2, 4×2, 4×4), N ∈ {2, 4, 8, 16}, five
random draws (components, κ, Dirichlet data) per condition; monolithic FEM
reference. Screened variant (+u) as a null-space-free control on chains.

**Checks (all pass).** Every exact response is symmetric to roundoff, has
minimum eigenvalue ≥ −3e-16 of its norm, and annihilates the constant vector
to 6e-16. The composition law S_EE − S_EI S_II⁻¹ S_IE matches the merged-mesh
Schur complement to 2.4e-16; the exact interface solve reproduces the
monolithic solution to 5e-15 in energy norm.

**What we saw.**

1. Error propagation (every response perturbed by relative Frobenius error ε;
   medians over 5 draws; global relative energy error):

   | layout | N | n_I | cond(S_II) | ε=1e-3 random | ε=1e-2 random | ε=1e-2 low modes | ε=1e-2 high modes | ε=1e-1 random |
   |---|---|---|---|---|---|---|---|---|
   | chain | 2 | 15 | 15 | 0.0021 | 0.017 | 0.039 | 0.0001 | 0.15 |
   | chain | 4 | 45 | 16 | 0.0041 | 0.040 | 0.092 | 0.0002 | 0.50 |
   | chain | 8 | 105 | 88 | 0.0079 | 0.071 | 0.072 | 0.0005 | 0.85 |
   | chain | 16 | 225 | 76 | 0.0070 | 0.065 | 0.091 | 0.0002 | 1.36 |
   | grid | 4 | 61 | 220 | 0.0062 | 0.055 | 0.083 | 0.0002 | 0.97 |
   | grid | 8 | 153 | 330 | 0.0152 | 0.333 | 0.267 | 0.0006 | 1.87 |
   | grid | 16 | 369 | 6900 | 0.0163 | 0.340 | 0.207 | 0.0004 | 3.33 |

   The screened variant gives the same picture on chains (ε = 1e-2 random:
   0.017 → 0.056 from N = 2 to 16). Errors on the high (rough) port modes are
   attenuated 20–100x; errors on the low (smooth) modes are amplified 4–30x
   and set the global error. Chains: N = 16 error is 3.7x the N = 2 error,
   between √N (2.8x) and linear (8x). Grids: N = 16 error is 6.2x the N = 4
   error, faster than linear (4x); the 4×4 grid at ε = 1e-2 has 34% global
   error, and even ε = 1e-3 leaves 1.6%.

2. Port reduction (smallest r of 15 edge-interior modes reaching the median
   error target; vertices kept exact):

   | shapes | layout, N | transfer-optimal 1% / 5% | POD 1% / 5% | error at r = 4 (transfer / POD) |
   |---|---|---|---|---|
   | library | chain 8 | 10 / 7 | 7 / 3 | 0.22 / 0.037 |
   | library | grid 8 | 14 / 9 | 10 / 6 | 0.24 / 0.092 |
   | library | grid 16 | 15 / 12 | 11 / 6 | 0.21 / 0.10 |
   | new shapes | chain 8 | 13 / 11 | 15 / 10 | 0.17 / 0.15 |
   | new shapes | grid 8 | 15 / 11 | 15 / 13 | 0.16 / 0.15 |
   | new shapes | grid 16 | 15 / 11 | 15 / 15 | 0.27 / 0.17 |

   No basis reaches 1% at r ≤ 4 anywhere. POD modes fitted on library
   assemblies are shape specific (7–11 modes on library shapes, all 15 on new
   shapes); transfer-optimal modes from a plain reference pair behave alike on
   library and new shapes but need 10–15 of 15 for 1%. The transfer operator's
   singular values decay fast (0.69 → 6.6e-9) but the assemblies' interface
   traces, driven by κ contrasts, holes and neighbours, do not live in its
   leading modes. Rank-r truncation of the component response itself is
   catastrophic (errors 1e2–1e7 below r = 63 of 64): the small-eigenvalue,
   smooth part of S is the part that matters.

3. Cost. Dense interface solve proxy n_I³/3 against the monolithic sparse
   factor nnz(L+U) times 10: chain N = 16: 3.8e6 vs 1.0e6; grid N = 16:
   1.7e7 vs 2.0e6. The interface solve is cheaper only for N ≤ 4. cond(S_II)
   grows from 15 to 6.9e3.

4. Coarse space. PCG on S_II with exact edge blocks: 1 → 5 iterations on
   chains, 10 → 34 on grids as N grows; adding a partition-of-unity constant
   per component does not reduce the count (2 → 8, 12 → 30). This coarse
   space does not help here; H3 is not supported by this instrument.

5. Near field. `channel_narrow` (one-element opening) is indistinguishable
   from `channel_wide` and `plain` in rank and amplification (r = 8: 0.0013
   in all three; amplification 4.6–4.7). The barrier is internal to the
   component, so the port response is smooth; this instrument does not
   exercise a near-port singularity.

**Verdict against the preregistered bars.** NOT WORTH a learner in the form
proposed (learned port responses at ~1e-2 local error, reduced interface).
Both halves of the WORTH bar fail: at ε = 1e-2 the global error exceeds 5% for
chains with N ≥ 8 and for every grid, and grid growth from N = 4 to 16 is
superlinear (6.2x > 4x), which is the NOT-WORTH clause; transfer-optimal
modes at r ≤ 4 reach 16–27%, not 1%, on new shapes and on library shapes
alike. The cost clause is met only partly (the interface solve is cheaper at
N ≤ 4). Rival R1 (interface amplification) is confirmed for grids; rival R2
(shape-specific modes) is confirmed for POD and refuted for transfer-optimal
modes, which transfer but do not compress.

**Why it matters.** A learned local response would need relative error near
1e-3 on the smooth port modes to deliver ~2% global energy error on a 4×4
assembly, and no cheap port reduction is available at 1%. The one favourable
structure is the mode asymmetry: errors on rough port modes are attenuated
20–100x. A learner that keeps the smooth response exact (a coarse local
solve, or analytic low-mode response) and learns only the rough remainder
would have its errors damped rather than amplified. That is the only version
of this direction the gate leaves open, and it is a hybrid solver, not a
field regressor.

**Scope.** 2D scalar diffusion with identical straight ports. Convection
removes symmetry and the energy norm (no Galerkin certificate for the
interface solve); elasticity adds rigid-body null modes per component that
must be kept exact; exterior aerodynamics has no shared connector ports, so
the multipole coupling variant of the agenda is the relevant analogue, with
the same low-mode/high-mode question to ask first.

**Next.** Not a training arm. If the direction is kept, the next nonlearned
test is the split response: exact low-mode block (r ≤ 4 modes per port from a
coarse local solve) plus perturbed high-mode remainder, to measure the global
error budget a rough-mode learner would enjoy; and a near-port singularity
family (barrier opening adjacent to a port) to test the near-field claim
properly.
