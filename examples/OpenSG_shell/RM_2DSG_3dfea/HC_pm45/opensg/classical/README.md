# classical/ -- the classical-plate recovery + the path-1 RM debug

Two jobs live here.  FIRST the deliverable: the CLASSICAL plate model
(refined: 0, ABD 6x6) dehomogenized at the same station and added as the
third curve to both path comparisons.  SECOND the debug: WHY the RM
recovery under-predicts the path-1 in-plane oscillation -- settled with
a chain decomposition, a free-edge scan, a mesh-refinement study and a
mechanism test.  `bash run.sh` reruns the deliverable half (server,
opensg_2_0 env); the debug scripts run standalone the same way.

## The classical run (deliverable)

    45pm_singleextrude_bd_classical.yaml   copied from ../sg_2d (refined: 0)
    45pm_singleextrude_bd_classical.ff     FF ONLY -- the classical route
                                           consumes no strain derivatives
                                           and no face pressure (those need
                                           the refined: 1 ladder); same
                                           station resultants as
                                           ../dehomo/45pm_singleextrude_bd.ff
    ..._classical.out / _classical_dehom.* `opensg <yaml> D`, MATERIAL frame
    sm_to_dat_classical.py                 .SM -> path_1_classical.dat /
                                           path_2_classical.dat (same gates
                                           as ../path_1/sm_to_dat.py)
    make_path_plots_3way.py                the 12 path_N_S*.png (Abaqus 3-D
                                           solid / MSG-RM / MSG classical)
                                           + rms_3way.txt

Result: classical is far off everywhere the refined chains matter --
path 1 in-plane 30-34 % RMS (vs RM 9-10 %) because it has NO pressure
load-ladder, transverse 53-73 % (vs RM 3-16 %) because it has NO
gradient (Eq. 63) recovery; path 2 similar story.  The 3-way plots are
the honest picture of what the RM refinement buys.

## The path-1 RM debug (why RM is not the 3-D either)

Ran in this order; each writes its .log next to it.

1.  debug_decompose_path1.py -- the dehom is linear, so full = base
    (V0 eps) + grad (Eq. 63/64-66) + q (pressure ladder), split exactly
    (additivity 0.0; the stored ../dehomo .SM is reproduced to 7e-8).
    FINDING: the path-1 oscillation is ~92 % q-chain (std 1.51 vs 0.13
    for the strain chains, S11); the residual (3-D - RM) correlates
    0.985 with the q-chain shape; alpha_q = 1.28 uniformly on
    S11/S22/S12 while S13/S23/S33 sit at alpha ~ 1.  So the ONLY
    deficient piece is the in-plane content of the pressure column.
2.  abaqus_cell_scan.py -- the same top-row profile from every
    width-offset window copy (k = -3..+3).  FINDING: interior windows
    match the station to 1.3-3.8 % (in-plane), so the 3-D IS
    cell-periodic at the centre and the gap is NOT a free-edge effect.
    (S23 varies 15-23 % cell-to-cell -- its 15.7 % path-1 RMS is real
    non-periodicity of a near-zero component, not a dehom defect.)
3.  refine_sg_test.py -- 2x2 / 3x3 bilinear SG refinement.  FINDING:
    the q-chain amplitude converges 0.78 -> 0.80 of the 3-D, so the
    deficit is NOT Q4 discretization/locking.
4.  test_qcolumn_redistribution.py -- THE MECHANISM.  The pressure
    column is a pure-Neumann solve whose net face force is absorbed by
    the <w> = 0 KKT rows = a UNIFORM body force over the material area;
    physically the balancing spanwise shear flows through the CORE
    WALLS, so the uniform reaction under-loads the top-face bay
    bending.  Re-reacting the net force on the walls (net-zero column,
    KKT silent) lifts the in-plane amplitude 0.78 -> 0.91 and cuts the
    path-1 in-plane RMS 9-10 % -> 6.3-6.6 %.
5.  final_composition_test.py -- the matrix.  tau-weighted reaction
    (the cell's own sigma_xz under unit Q1, integral gate 0.9999) ==
    walls to within noise.  Feeding the 3-D's own integrated resultants
    (check_3d_moment.py: M11 matches the plate to 0.3 %, but the 3-D
    carries N11 = -0.26, N22 = -0.17 N/mm the linear plate cannot) makes
    the top-row mean WORSE -- so the remaining ~0.3 MPa mean offset is a
    through-thickness redistribution (St-Venant content of the clamped
    end faces, ~4 panel thicknesses away), not a driver error.

BOTTOM LINE: the RM dehom implementation is correct (every rebuild gate
digit-tight).  The path-1 gap decomposes into (a) the uniform-reaction
gauge of the pressure load column -- a formulation choice that is exact
for in-plane-homogeneous laminates (where it was gated) but damps the
bay bending of an in-plane-heterogeneous sandwich by ~20 %; fixable by
reacting the load column with the cell's shear path (walls / tau), at a
small cost in sigma33/sigma13 interior accuracy -- and (b) a ~10 % mean
offset from non-asymptotic 3-D content no periodic SG model carries.

## SwiftComp cross-check inputs

`opensg yaml_to_sc` (new subcommand, io/yaml_to_sc.py -> io/sg_input.py)
emitted ../sg_2d/45pm_singleextrude_bd.sc and ..._classical.sc from the
yamls themselves (n_model/refined from the header, SG dim measured,
angles folded into pre-rotated type-2 blocks).  Run as
`SwiftComp 45pm_singleextrude_bd.sc 2D H`; the ABDG should match
../sg_2d/45pm_singleextrude_bd.out (round-trip gate through sc_to_yaml
already matches to 9e-16).
