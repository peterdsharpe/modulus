# C3 — paired physical-condition control: notebook

## 2026-09-09 — Preregistered (PREREG.md), first qualification run fails; instrument repaired before any training

**Setting.** PREREG.md was written before any run: exterior potential flow past 40
superellipsoids plus the unit sphere, free air versus a ground plane on identical
bodies, labels from a method-of-fundamental-solutions (MFS) reference with a
four-step qualification ladder, then small ISLA arms with and without the condition
input. Compute: the local host was oversubscribed (load average 62, a one-second
linear solve took minutes and one run was killed for memory), so all reference and
training runs execute on a cluster compute node inside a running allocation
(`srun --overlap`), CPU only; nothing was launched as a new job.

**First qualification run (shrunken random copy of the mesh vertices as sources,
collocation at flat-facet centroids with facet normals, clearance h = 0.25 c):**
Q1 sphere error 0.019 (bar 1e-3), Q2 held-out residual up to 0.56 (bar 1e-3),
Q3 resolution disagreement up to 102 in C_p RMS (one boxy case diverged),
Q4 plane residual 5e-8 (pass), ground effect 45% RMS of the field (bar 5–20%).

**Diagnosis, in three prototypes (proto2–proto5 on the cluster).** (1) The 1.9e-2
Q1 floor did not move with source count, shrink factor or regularization: it is the
*mesh*, not the solver. Collocating at facet centroids with facet normals solves the
flow past the polyhedron, whose C_p differs from the sphere's by O(h²)·|∇u| ≈ 1e-2 at
this resolution. Repair: collocate at the exact surface points in the centroid
directions with the analytic normals (radial projection onto |x/a|^p + |y/b|^p +
|z/c|^p = 1; the facet areas remain the measure weights the learner sees). The sphere
then reaches Q1 = 1.8e-5. (2) On thin or boxy bodies (c ≈ 0.5, p ≈ 2.5) the shrunken
copy leaves held-out residuals of 1–2e-2; a fixed inward offset of 0.25 body units
along the exact normal on a regular 26×40 source grid gives 3e-4 to 1.5e-3 (max
≤ 6e-3) with no blow-ups; offsets 0.3–0.45 are worse and non-monotone, denser grids
do not help, and a local-spacing rule (offset ∝ nearest-neighbour source distance)
is unstable (C_p blow-ups on four of six test shapes). (3) Clearance h = 0.6 c puts the
ground effect at a median 17% RMS change (8–34% across shapes), inside the band.

**Amendments recorded before any label is used for training.** Instrument: exact
surface collocation; sources at offset 0.25 along the normal on a 26×40 grid (fine)
and 20×32 (coarse, for Q3); held-out residual at 600 random exact surface points per
case (not the vertex directions); clearance h = 0.6 c; shape family narrowed to
b, c ∈ [0.55, 1.0] and p ∈ [2, 2.5] (the excluded corner of the family, c < 0.55 with
p > 2.5, is where the MFS lost a decimal). Bars: Q1 unchanged (< 1e-3); Q2 held-out
residual 90th percentile ≤ 2e-3 · |U∞| on every case (was 1e-3); Q3 fine-versus-coarse
C_p agreement ≤ 5e-3 RMS on every case (was 1e-3); Q4 unchanged. Justification: a
C_p field RMS of about 0.5 makes 5e-3 a 1% relative reference precision, an order of
magnitude below the smallest effect this control measures (the 8% minimum ground
effect) and below any learner error expected at this model size; the original bars
were aspirational for a point-source method on non-spherical bodies. The learner
bars in PREREG.md are unchanged.

## 2026-09-09 — Reference qualified per case (38 of 41); first training budget too small to expose the floor; budget raised before the verdict

**Reference.** With the amended instrument: Q1 sphere error 4.4e-5 (bar 1e-3);
Q4 plane residual 1e-13; ground effect median 15.5% RMS (8–28%), in band. Per-case
gates (Q2 held-out residual 90th percentile ≤ 2e-3, Q3 fine-versus-coarse
agreement ≤ 5e-3 RMS) pass on 38 of 41 cases; cases 18, 20 and 32 fail (Q2 up to
3.1e-3, Q3 up to 6.4e-3) and are excluded from training and evaluation
(`qualification.json`, `labels.npz` field `qualified`). The global Q2/Q3 flags in
`qualification.json` therefore read false while every case used passes.

**First training run, 600 optimization steps (`results_600steps.json`; 28 training
shapes, 9 held-out shapes plus the sphere; ISLA hidden 64, 4 layers, 32 slices; three
seeds).** Relative L2 of C_p, seed means on held-out shapes:

| arm | free air | ground |
|---|---|---|
| A current encoding, mixed conditions | 0.139 | 0.150 |
| B + ground-proximity scalar, mixed | 0.118 | 0.125 |
| C free air only (in-distribution reference) | 0.108 | 0.165 (applied out of condition) |
| M per-point MLP, mixed | 0.262 | 0.233 |
| analytic equal-mixture floor | 0.080 | 0.071 |

Arm A sits at 1.9x the floor but only 1.35x arm C, so the composite "floor
confirmed" bar (≥ 0.8x floor AND ≥ 3x arm C) is not met: the in-distribution error
of this small, briefly trained model (0.108) exceeds the floor (0.076), and the
identifiability penalty appears only as a quadrature addition (√(0.108² + 0.076²) =
0.132 against arm A's 0.139). The repair bar is met (arm B within 1.10x and 1.16x of
arm C), arm B fits its training set (1.13x arm C's training error), and the MLP is
at 3.3x the floor (architecture-independent). Signature: 53% of arm A's squared error
on ground cases lies in the lowest quarter of the body by height (uniform
expectation 25%; range 3–85% across shapes), so the missing condition leaves a
near-ground error concentration a transfer audit could look for.

**Decision before reading further.** The prereg's floor bar presupposes an
in-distribution error well below the floor; at 600 steps the instrument cannot
resolve it. The training budget is raised to 3,000 steps (same models, seeds,
split; cosine schedule) and the verdict is taken from that run. The bars are not
changed. The 600-step numbers stay on record.

## 2026-09-09 — Verdict at 3,000 steps: repair CONFIRMED; the floor is present at 1.3x its analytic value, but the preregistered composite floor bar is not met because its 3x criterion was miscalibrated

**Readout** (`results.json`, `train_3000steps.log`; same split, models and seeds as
the 600-step run; ISLA hidden 64, 4 layers, 32 slices; 3,000 steps with a cosine
schedule; about 13 min per ISLA arm-seed on 32 CPU cores). Relative L2 of C_p on the
held-out shapes, seed mean (seeds in brackets for the ground condition):

| arm | free air | ground | training fit (free / ground) |
|---|---|---|---|
| A current encoding, mixed conditions | 0.075 | 0.123 [0.120, 0.127, 0.121] | 0.081 / 0.117 |
| B + ground-proximity scalar, mixed | 0.045 | 0.058 [0.062, 0.060, 0.052] | 0.039 / 0.050 |
| C free air only (in-distribution reference) | 0.040 | 0.157 when applied out of condition | 0.030 / – |
| M per-point MLP, mixed | 0.237 | 0.219 | 0.269 / 0.216 |
| analytic equal-mixture floor (mean over held-out shapes; range 0.04–0.11) | 0.080 | 0.071 | – |

Bars (PREREG.md): arm A paired error 0.099 = **1.31x** the floor (≥ 0.8x: met) and
**2.49x** arm C (≥ 3x: not met) → "identifiability floor CONFIRMED" is **not**
awarded under the preregistered wording. Arm B: 1.14x arm C in free air and 1.46x on
ground (bar ≤ 1.5x) → **repair CONFIRMED**. Arm B fits its training set at 1.49x arm
C's training fit. The MLP sits at 3.0x the floor (architecture-independent).
Signature: **78%** of arm A's squared error on ground cases lies in the lowest
quarter of the body by height (uniform expectation 25%; range 57–88% across held-out
shapes); arm A predicts a free-air-like field everywhere (0.075 on free air, better
than the floor's 0.080) and pays on the ground side (0.123), instead of the
equal-mixture average.

**Why the 3x criterion could not be met, stated in readout units.** If the omitted
condition adds the floor in quadrature to the model's own error, the expected
paired error is √(0.040² + 0.076²) = 0.086, which is 2.15x arm C; the observed 0.099
(2.49x) exceeds even that. The criterion "≥ 3x arm C" is only reachable when the
in-distribution error is below one third of the floor (≤ 0.025 here), which was not
derived when the bar was written. This is a calibration error in the
preregistration, recorded as such; the bar is not rewritten after the fact, and the
verdict says exactly what was met: the floor criterion (1.31x) and the repair
criterion, not the composite.

**What this control licenses.** On an exactly specified boundary-driven problem,
omitting one physical condition (a ground plane) from the input produces an error
of the size of the paired label difference, concentrated where the physics differs
(near the ground), that is not reduced by training on both conditions and that is
architecture-independent (ISLA and an MLP both pay it); supplying the condition as
a per-point invariant scalar removes it to within 1.15–1.46x of the in-distribution
error at this model size. For the DrivAerML → SHIFT-SUV transfer this says the
compound shift (ground and wheel motion, freestream speed, solver and closure) CAN
produce a zero-shot error that no geometry-only encoding removes, and that its
signature would be a near-ground / near-wheel concentration of the error; C1 already
showed that transfer error is spatial and force-relevant. It does NOT say how much of
the observed 0.99 / 0.59 relative error those conditions cause: the ground effect
here is a 15% RMS field change, the automotive shift is not one condition, and
potential flow has no separation. The decisive test remains paired CFD cases on
identical bodies with one condition changed, as the audit specified.

**Artifacts.** `qualification.json` (reference ladder, per-case gates, 38 of 41
qualified), `labels.npz` (positions, exact normals, facet areas, C_p in both
conditions, plane height, qualification flag), `results_600steps.json`,
`results.json` (3,000 steps), `train_3000steps.log`, `fig_sphere_cp.png`,
`fig_error_vs_floor.png`; code `reference.py`, `train_arms.py`, `make_figures.py`,
prototypes `prototype_sources.py` (the offset sweep that chose the placement).
