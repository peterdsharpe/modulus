# C3 — Paired physical-condition control (preregistration, written before any run)

Date: 2026-09-09. Track: transfer program, controls (C3). Owner: forked agent.

## Question

When the geometry is identical and exactly one physical condition changes, does the
current ISLA encoding (surface points, normals, cell areas, freestream direction)
fail in the way the identifiability argument predicts (an irreducible floor set by the
mixture of the two conditions), and does a complete encoding (the condition supplied
as an invariant per-point scalar) repair it? This is the exactly specified analogue of
the DrivAerML → SHIFT-SUV compound shift (ground/wheel motion, speed, solver), on a
problem where the labels are exact and the changed condition is one thing.

## Instrument

Physics: exterior incompressible potential flow past a smooth closed body, unit
freestream along +x, pressure coefficient C_p = 1 − |u|². Reference solver: method of
fundamental solutions (MFS) — point sources on a shrunken copy of the body surface
(scale 0.7), strengths from a least-squares impermeability fit u·n = 0 at the sampled
surface cell centroids (more collocation points than sources). Ground effect: a plane
z = −h below the body, enforced by the image method (mirror sources, same strength; the
plane's normal velocity vanishes by symmetry); the freestream is parallel to the plane.

Qualification ladder (must pass before labels are used): (Q1) the sphere's analytic
C_p = 1 − (9/4) sin²θ (θ from the +x axis) matched with maximum absolute error < 1e-3 at
the sampled centroids; (Q2) impermeability residual |u·n| at HELD-OUT surface points
(triangle vertices, not used in the fit) below 1e-3 · |U∞| at the 90th percentile;
(Q3) two resolutions (sources 600 and 1,200) agree in C_p to < 1e-3 RMS on every case;
(Q4) the image-method ground plane satisfies u·e_z = 0 at sampled plane points to
< 1e-3.

Geometry family: 40 superellipsoids |x/a|^p + |y/b|^p + |z/c|^p = 1 with a ∈ [0.7, 1.4],
b ∈ [0.5, 1.0], c ∈ [0.45, 1.0], p ∈ [2, 2.8] (p = 2 is the ellipsoid), sampled by a
seeded Latin hypercube; the unit sphere added as case 0 for Q1. Surface discretization:
a (θ, φ) grid triangulated to about 2,000 triangles; per cell the centroid, outward
normal and area (the recipe's surface inputs). Body length L = 2a.

Paired condition: each body is solved (a) in free air and (b) at ground clearance
h = 0.25 c above the plane (a quarter of the body half-height; the freestream is
parallel to the plane). The ground effect must change C_p by 5–20% RMS relative to the
free-air C_p field's RMS on the family median, or h is adjusted once, before any
training, and the adjustment recorded.

Learners: ISLA (physicsnemo.experimental.nn.isla.ISLA), hidden 64, 4 layers, 32 slices,
float32, CPU, out_scalars 1 (the vector head is unused); inputs points, normals, drive
direction, cell areas as measure weights. Adam, learning rate 1e-3, 600 optimization
steps of one case each (random case order), Huber loss on C_p; three seeds (0, 1, 2).
Split: 30 training geometries, 10 held-out geometries (by shape), in both conditions.

- Arm A, current encoding: trained on the mixed set (30 shapes × 2 conditions), no
  condition input.
- Arm B, complete encoding: as A plus one boundary scalar per point, the ground
  proximity s = 1 / (1 + z_i / L) where z_i is the point's height above the plane in
  body lengths L, and s = 0 in free air (an invariant per-point scalar; it goes to
  zero as the plane recedes, so the free-air limit is continuous).
- Arm C, in-distribution reference: trained on free-air cases only, evaluated on
  free-air held-out shapes.
- Arm M, architecture-independent check: a per-point MLP (3 × 128) on body-frame
  coordinates, normals and drive, trained as arm A (mixed, no condition input).

Readouts: per case and condition the relative L2 error of C_p, ||ŷ − y|| / ||y||
(uniform over sampled cells), seed mean; the analytic equal-mixture floor per geometry,
||(y_free − y_ground)/2|| / ||y_c|| for each condition c (the best single prediction for
both conditions is their average); the ratio of arm A's error to that floor; and the
spatial distribution of arm A's error versus height above the plane (fraction of the
squared error in the lowest quarter of the body by height), to show whether the missing
condition leaves a physically interpretable signature.

## Bars, in readout units

- **Identifiability floor CONFIRMED** if arm A's seed-mean relative L2 on the paired
  set (both conditions, held-out shapes) is ≥ 0.8 × the analytic equal-mixture floor
  AND ≥ 3 × arm C's in-distribution error. (Arm A cannot do better than the floor
  except by chance; an error far below it would mean the instrument leaks the
  condition.)
- **Repair CONFIRMED** if arm B's seed-mean error on held-out shapes, in each condition
  separately, is ≤ 1.5 × arm C's in-distribution error.
- **Repair NOT confirmed** otherwise; then check that arm B fits its training set to
  ≤ 1.5 × arm C's training error (capacity/optimization) and report which.
- Architecture independence: arm M's mixed error must also be ≥ 0.8 × the floor.
- Signature: report the fraction of arm A's squared error in the lowest height quarter;
  no bar (descriptive).

## What each outcome licenses

Floor confirmed and repair confirmed: an omitted physical condition produces an
irreducible, architecture-independent error whose size is the paired label difference,
and supplying the condition as an input removes it at this scale. That licenses the
statement that the DrivAerML → SHIFT-SUV gap COULD be dominated by omitted conditions
and that the decisive test is paired simulations on identical bodies; it does NOT
establish how much of the observed 0.99 / 0.59 error those conditions cause (that needs
the paired CFD cases the audit asked for, not this control). Floor confirmed but repair
not confirmed: the condition input alone is insufficient at this model size; report,
do not generalize. Floor NOT confirmed: the instrument is wrong (a leak of the
condition through the inputs, e.g. via the weights or coordinates), and the control is
repaired before any conclusion.
