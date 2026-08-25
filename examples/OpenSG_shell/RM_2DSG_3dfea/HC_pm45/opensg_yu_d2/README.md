# opensg_yu_d2 -- HC_pm45 under the PURE Yu (V2-constitutive) composition

Experiment folder (2026-08-25), mirroring the CMC study's AR5_yu_d2
pattern.  The study's VALIDATED results (opensg/dehomo + compare/four_way)
were produced in the tau mode: `q_reaction: tau`, whose re-reacted
pressure column subsumes the moment-gradient transverse content, so the
d2eps chains are dropped/inert.  This folder reruns the SAME station with
`q_reaction: uniform`, i.e. Yu's 2003 composition: pressure carried
constitutively by the V1L/V2L load columns with the <w>=0 uniform KKT
reaction, second-order content from the Eq. 64-66 d2eps two-chain
recovery.

The .ff here = opensg/dehomo/45pm_singleextrude_bd.ff (the symmetrized
midspan drivers behind the validated tau .SM) with ONLY the q_reaction
line changed tau -> uniform, so the A/B isolates the composition.
(The HC_pm45 root .ff is the older pre-symmetrization station .ff --
NOT used, its nonzero deps_dx* are inconsistent with the midspan 3-D
reference plane.)

NOTHING in this folder overwrites the study's tau-mode deliverables.

Contents / rerun order (server, opensg_2_0 env, PINNED snapshot
PYTHONPATH=$HOME/opensg_pin/src):

    45pm_singleextrude_bd.yaml     copy of the study SG (hash-identical
                                   to the root and opensg/sg_2d copies)
    45pm_singleextrude_bd.ff       the Yu-composition load (see above)
    python -s -m opensg 45pm_singleextrude_bd.yaml D
                                   -> _dehom.SM/.U here (console must
                                   print the q-recovery + V2 second-order
                                   lines and NO tau lines)
    path_1/, path_2/               .coords copied from opensg/path_N;
                                   sm_to_dat.py adapted (reads ../ SM and
                                   ../ yaml) -> path_N_opensg.dat
    path_plots/make_path_plots_yu.py
                                   four_way plot conventions, RM curve =
                                   THIS folder's recovery; 3-D C3D20R
                                   reference and classical curves are the
                                   study's originals, untouched
                                   -> path_N_S*.png + rms_yu_d2.txt
                                   (extra column = the study's tau-mode
                                   RM for the A/B)
    d2_driver_quality.txt          the plate-equilibrium identity
                                   (C_eff@d2eps_11)[M11] + 2(C_eff@d2eps_12)[M12]
                                   + (C_eff@d2eps_22)[M22] vs +q -- how
                                   much of the pressure the d2 drivers
                                   actually carry

Regenerable exports (*.vtk, *_dehom.txt, *.EM) are deleted after the run
(home quota); .SM and .U are kept.
