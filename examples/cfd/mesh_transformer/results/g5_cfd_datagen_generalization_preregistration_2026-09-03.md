# Preregistration draft: G5 — single-factor generalization on the cfd-datagen corpus

*Draft written 2026-09-03 by the mesh-transformer program fork, for import into `examples/cfd/mesh_transformer/results/` (as JSON) and the lab notebook once the cfd-datagen launch plan is fixed. Status: PRE-DATA. Every threshold below is stated in the units of the readout it must explain, per the N1/L1 lesson (feedback memory `prereg-threshold-calibration`).*

## Question

Does any architectural structure buy generalization on 3D RANS labels along a *single controlled factor*, when the factor is either (a) a symmetry the architecture carries exactly (pose, scale) or (b) a coverage axis it does not (Reynolds number, geometry family, boundary-condition layout and values)? The 2D conformal-Laplace result (G3, kernel rerun) says structure pays exactly on (a)-type axes and not at all on (b)-type axes; the high-lift and P1'' results say that on production datasets nothing separates the architectures in-family and canonicalization matches equivariance. G5 asks the same question on a 3D corpus built to isolate factors.

## Dependencies on the dataset (must exist for the arm to be valid)

| G5 arm | requires cfd-datagen recommendation |
|---|---|
| A1 pose twins, A2 scale twins | rec. 1 twins on `geometry_pose` and scale |
| A3 Re twins | rec. 1 twins on `physics_parameters` |
| A4 mesh twins (noise floor) | rec. 1 or rec. 3 refinement twins |
| B1 bank-held-out geometry | rec. 2 val/test geometries generated |
| B2 profile-held-out BCs | none (profiles are in the base corpus) |
| all surface metrics | rec. 4 `body_id` on `no_slip` |
| uncertainty-weighted metrics | rec. 5 block-difference fields |

If rec. 1 is not adopted, A1–A3 degrade to post-hoc marginal splits and their predictions below are void; only B1/B2 and the in-family control remain.

## Architectures and protocol

- GeoTransolver (lr 3e-3 and 1e-3, best per val), GeoTransolver + canonicalization where a single freestream exists (`open_opposed`, `ground_effect`, `confined_opposed`; NOT `dual_inlet`/`crossflow`), MeshTransformer2 with `similarity_gauge=true` (lr 1e-3), and the kernel-decoder MeshTransformer flagship config as the GLOBE-lineage arm. Optional: AB-UPT if the baseline exists by then.
- Surface task first (pressure and wall shear on body faces, measure-weighted rel-L2), volume task second (interior velocity/pressure on cell centroids with `_target_quadrature_measure`), since boundary→interior is the program's stated objective and MT2 has never run on it.
- Training set: the 3,360-case train corpus minus any held-out bank (B1). Two seeds minimum per architecture; report seed spread with every number. Sampling resolution 10k surface / 40k interior points per case; frozen otherwise.
- Every OOD number is reported next to the per-stratum label floor from A4 (medium→fine discretization difference) and the block-difference field (rec. 5). A difference between architectures smaller than the floor is reported as "not resolvable", not as a result.

## Arms, predictions, falsifiers (rel-L2 on body pressure unless stated)

**A1 — pose twins (exact SE(3) axis).** Evaluate each model on the pose twin of a training case (same geometry, physics, BCs, mesh stream; different rotation and translation).
- Prediction: MT2 twin-vs-original prediction difference equals its own mesh-realization floor (A4) to within 10% of that floor; raw GeoTransolver's difference exceeds the floor by ≥ 5× on `dual_inlet`/`crossflow` cases (no canonical frame available); canonicalized GeoTransolver on single-freestream profiles matches MT2 within 1.05× (P1'' replicated on real labels).
- Falsifier for the equivariance contract's practical value: canonicalized GT within 1.05× on *all* profiles including the two without a freestream (some other frame estimate suffices) → equivariance is a convenience here too.

**A2 — scale twins (exact similarity axis, first real test).** Same case at a different `reference_length_m` with viscosity adjusted so Re is fixed.
- Prediction: MT2 (gauge) twin difference ≤ 1.1× its A4 floor across the full 0.02–20 m range; GeoTransolver degrades monotonically with |log scale ratio|, exceeding 2× in-family error at a 10× scale change (it has no scale gauge and cannot be canonicalized for scale without knowing the reference length is arbitrary).
- Falsifier: GeoTransolver within 1.2× at 10× scale change → the network learns scale from context and the gauge buys nothing measurable.

**A3 — Reynolds twins (coverage axis).** Same case at a different Re, one decade away, within the training Re range.
- Prediction (from G3 and G2): no architecture generalizes across a decade of Re better than another; all degrade by > 1.5× versus in-family, with the ordering by in-family accuracy preserved. Structure does not buy Re transfer.
- Falsifier: any architecture degrades < 1.2× while another degrades > 2× on the same twins → a real Re-transfer mechanism exists and becomes the program's next object of study.

**A4 — mesh twins (the noise floor, not an architecture test).** Same case, different `mesh_resolution` draw (and, where rec. 3 exists, the refined mesh).
- Readout: per-stratum (profile × Re decade × bank) label difference in the recipe's metric. This defines "resolvable" for every other arm. Prediction: the floor exceeds 5% rel-L2 at Re ≥ 1e7 for most strata and is below 3% at Re ≤ 1e5.

**B1 — bank-held-out geometry (coverage axis, strong version of G2).** Train on four banks, test on the fifth's val/test geometries; rotate the held-out bank.
- Prediction: degradation ratio ordering follows shape-space MMD (Chapter 1b): holding out `zero-to-cad` costs the most, `streamlined` the least. Between architectures, prediction from G2 is that GeoTransolver's transfer improves more with training size than MT2's; run at two training sizes (1,000 and full) to test the slope. Falsifier for the "pointwise coordinate features carry transfer" hypothesis: MT2 (gauge) slope ≥ GeoTransolver's on the held-out bank.

**B2 — profile-held-out boundary conditions.** Train without `crossflow` (or `dual_inlet`), test on it.
- Prediction: architectures with explicit BC operators/values as inputs (kernel-decoder MT, MT2 with boundary scalars) degrade less than coordinate-feature architectures; threshold: ≥ 1.3× separation in degradation ratio. Falsifier: all within 1.1× → BC ingestion structure does not help on realized 3D flows either.

## Cost

Training: 4 architectures × 2 seeds × (1 in-family + up to 5 held-out configurations) on 3k cases of 1.5–5M cells — dominated by the volume task; budget ~2–4k GPU-hours on GB300, in chain-ahead links. Evaluation: twins and held-out sets are eval-only once trained. Preregistration and dataset decisions must be frozen together, which is why this draft exists before generation.

## What would change the program

- A1/A2 confirmed and A3/B negative: the thesis becomes "exact symmetries buy exact generalization on their axes, coverage axes are bought by data" — publishable with the 2D result as mechanism and 3D as confirmation.
- Any B-arm showing a ≥ 1.3× architecture separation with the A4 floor respected: the first measured architecture-driven generalization in 3D, and MT3's design target.
- Everything within the A4 floor: the honest conclusion that at 1.5–5M cells with transition SST the labels do not resolve architecture differences on OOD axes, which itself bounds what any surrogate benchmark on this operator can claim.

## Amendments 2026-09-08

Written after the independent audit of 2026-09-08 (book notebook entry `sec-nb-audit-2026-09-08`). Each amendment corrects a predicted outcome above that does not follow from the current input pipeline, or a scheduling statement superseded by the program's current rulings. The original text above is left as written; where an amendment and the original conflict, the amendment governs.

1. **A2 scale twins: the predicted GeoTransolver failure is not an architecture prediction as written.** The recipe's nondimensionalizer (`unified_external_aero_recipe/src/nondim.py`) divides coordinates by the per-case reference length, so under x → a·x with L_ref → a·L_ref every model, GeoTransolver included, receives tensor-identical coordinates; a factor-of-two GeoTransolver degradation at a 10× scale change cannot occur unless the input convention deliberately changes. The arm is split into three, each with a tensor-identity test run *before* training: (a) *physical rescaling with L_ref carried* (the twin as the corpus generates it): prediction for every architecture is agreement with the original to within the A4 floor, and a difference between architectures here is a pipeline bug, not a result; (b) *reference-length convention change* (L_ref held at the original while the geometry is rescaled, so the nondimensional coordinates change by a): this is the axis on which ISLA's similarity gauge is a contract and GeoTransolver has none, and the original A2 prediction (GeoTransolver > 2× in-family error at 10×, ISLA within 1.1× of the A4 floor) applies to (b) only; (c) *mesh change at fixed geometry*, which is A4 and not a scale test. The falsifier for (b) is unchanged.

2. **A3 Reynolds twins need a conditioning channel before they are a Reynolds experiment.** The surface recipes ingest no viscosity and no Reynolds number; only the freestream vector (physical for GeoTransolver and Transolver, unit direction for ISLA) and the geometry. Varying viscosity at fixed geometry and freestream therefore produces different targets from identical admitted inputs for every model, and the arm as written measures the label spread of an ill-posed map, not transfer. Amendment: specify one Reynolds conditioning channel per model (a log-Re scalar as a global feature for GeoTransolver and Transolver; the same scalar through ISLA's dimensionless-scalar path, `scale_conditioning` or a boundary scalar, whichever the composition tests of node QMASS certify), verify by an input-sufficiency test that two twins with different Re have different admitted inputs, and only then run A3. Two estimands are separated: a decade shift *inside* the training Re range (interpolation; the original prediction of >1.5× degradation for all architectures with preserved ordering applies here) and a decade *beyond* it (extrapolation), which receives its own prediction: every architecture degrades by more than 3× and no ordering claim is made.

3. **B2 boundary-condition profiles need an input-sufficiency test.** A model that admits one global drive does not by construction represent crossflow, dual inlets, moving walls or ground effect. Before B2 runs, each architecture's admitted inputs for each profile are listed, and a transformed-input test confirms that boundary types, local boundary values and every symmetry-breaking direction transform together under the pose twin (A1). Profiles for which a model's admitted inputs do not distinguish the held-out condition are reported as "input-insufficient", separately from a generalization failure.

4. **A4 reference qualification.** Medium-versus-fine disagreement is a discretization *diagnostic*, not a calibrated bound on continuum error. The per-stratum floor is kept as the resolvability threshold for ranking architectures against fixed CFD labels, but no statement of "physically meaningful superiority below reference uncertainty" is made from it; where a third resolution or a justified convergence model is available, the floor is restated as a continuum-error estimate, and temporal repeatability is reported where the solver is unsteady.

5. **Superseded scheduling statements.** The architectures paragraph schedules "the kernel-decoder MeshTransformer flagship config as the GLOBE-lineage arm": the exact-kernel MeshTransformer is retired as a product candidate for 3D aerodynamics (potential-flow oracle gate failed, interior 8× behind; book chapter on the interior and CLAIMS scope decision 2026-09-07) and does not run in G5; the arms are GeoTransolver, GeoTransolver + canonicalization where a single freestream exists, ISLA with `similarity_gauge=true`, Transolver as the same-family control, and AB-UPT if node ABUPT delivers it. The same paragraph says "MT2 has never run on" the volume task: ISLA has, and in its query-token configuration with the signed distance leads GeoTransolver-volume on interior pressure (0.91×) and velocity (0.64×) and trails on eddy viscosity (1.29×) at 435 DrivAerML cars (results/a35_v0_reduction_2026-09-08.json); the volume task in G5 uses that configuration, subject to the query-mass repair of node QMASS, and the passive-decode configuration as the query-independent arm.

6. **Symmetry statement.** The A1/A2 predictions rest on the symmetry of the actual problem (incompressible RANS at fixed Re, free-space far field, one distinguished drive direction, coefficients and boundary data transformed with the geometry), not on homogeneity of the equations, which by itself buys translation invariance only; every profile that adds a ground plane, a second inlet or a size-dependent Reynolds number removes the corresponding symmetry and is excluded from the A1/A2 exact-axis predictions.

7. **Baseline input scale (carried from node UDRV).** GeoTransolver's and Transolver's global freestream input is the physical velocity, unnormalized; before any G5 arm trains, the drive is delivered to every architecture as a unit direction plus separately normalized physical parameters, so that the conditioning difference verified on HiLiftAeroML and DrivAerML (results/geotransolver_drive_conditioning_2026-09-08.json) cannot enter a G5 comparison.

Contract tests of transformed inputs (exact, in float64) stay separate from solver-twin tests (statistical, against the A4 floor) throughout; neither is reported as the other.
