"""opensg_shell -- the MSG SHELL engine (Reissner-Mindlin shell SGs ->
beam / equivalent-solid macro models), split file-per-concern:

    xsec_5v6_master.py / oml_ring.py  1-D ring yaml loaders (load_ring /
                                      load_ring_ref, reference conventions)
    segment_element.py,
    segment_element_general.py,
    segment_indep.py                  shell element operators + assembly
                                      (production 6-DOF drilling element)
    run_ring_indep.py                 ring homogenization -> Timoshenko 6x6
                                      (+ V0/V1 warping fields)
    dehom_rm.py                       build_rm_bundle driver + RM two-step
                                      dehomogenization
    solid_props.py                    equivalent 3-D solid from the
                                      cross-section SG (+ junction census)
    junction_micro.py                 junction micro/microcell corrections
    shell_sg3d.py                     3-D shell SG -> equivalent solid
                                      (TPMS-class; boundary periodic|aperiodic)
    segment_taper.py,
    boundary_from_yaml.py             aperiodic/tapered segment pipeline
                                      (boundary V0/V1 Dirichlet)
    emit_abd.py                       per-station wall-law ABDG emitter
    periodic_multiscale.py            periodic assembly map (6 DOF/node)
    orient_check.py                   orientation-frame reports / PNGs
    fe_jax/                           msg_* kernels: materials/ABD, RM
                                      operators, Timoshenko finalization
                                      (msg_solver), KL bundle, recovery

Every model writes a timed SwiftComp-layout .out; examples own all paths
(yaml in -> .out back).

The x64 flag below is LOAD-BEARING: every solver in this package
assembles/factorizes in float64; without it JAX silently degrades to
float32 and the 6x6 digits are wrong.
"""
import jax

jax.config.update("jax_enable_x64", True)

from .xsec_5v6_master import load_ring, ring_6dof, ring_5dof, LBL          # noqa: E402
from .oml_ring import load_ring_ref                                        # noqa: E402
from .run_ring_indep import ring_indep                                     # noqa: E402
from .dehom_rm import (build_rm_bundle, stress_at_points, disp_at_points,  # noqa: E402
                       _macro_fields, _rm_shell_strain)
from .emit_abd import emit_station_abd, load_station_abd                   # noqa: E402
from .fe_jax.msg_rm_timo import timoshenko_rm                              # noqa: E402
from .fe_jax.orient_plot import plot_orient, auto_emit                     # noqa: E402
from .solid_props import (build_solid_bundle, ring_solid,                  # noqa: E402
                          elastic_constants, GBAR_ORDER)
