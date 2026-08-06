"""opensg_shell -- OpenSG thin-walled Reissner-Mindlin SHELL cross-section -> Timoshenko beam.

The validated OpenSG-TW stack, transplanted as-is (Phase 3):

  HOMOGENIZATION (1-D shell ring SG -> Timoshenko 6x6, VABS order [EA GA2 GA3 GJ EI2 EI3])
    * build_rm_bundle(shell_yaml)      -- the production route: 6-DOF independent-omega3 RM
      ring element with element-wise drilling Lagrange multiplier, gamma_23 tied by MITC
      (run_ring_indep.ring_indep, shear="mitc4_g23"), MSG (Yu-2002 LS) wall G by default
      (g_source="msg", from opensg_solid.rm_plate_1D.msg_rm_plate).  Returns the bundle the
      two-step dehomogenization consumes (Timo 6x6 + V0/V1 warping + strip geometry + layup dbs).
    * load_ring / ring_6dof            -- the bare ring driver (xsec_5v6_master), Whitney G.
    * rm_timoshenko_6x6 / timoshenko_rm -- the 5-DOF drilling-eliminated RM element
      (fe_jax.strip_RM / fe_jax.msg_rm_timo); shear="mitc_both" is the validated default.

  TWO-STEP DEHOMOGENIZATION (beam FF -> shell section strains -> through-thickness recovery)
    * _macro_fields(B, beam_force_vabs=FF)  -- step 1a: st = C6^-1 FF + RM warping combos
    * _rm_shell_strain(B, e, xi, ...)       -- step 1b: the 8 RM shell strains per element
    * msgrm_strain_at_depth (import from opensg_solid.rm_plate_1D.msg_rm_plate)
                                            -- step 2: MSG-RM plate through-thickness recovery
    * stress_at_points / disp_at_points     -- point-query wrappers (KL plate-SG step 2)

The x64 flag below is LOAD-BEARING: every solver in this package assembles/factorizes in
float64; without it JAX silently degrades to float32 and the 6x6 digits are wrong.
"""
import jax

jax.config.update("jax_enable_x64", True)

from .xsec_5v6_master import load_ring, ring_6dof, ring_5dof, LBL          # noqa: E402
from .oml_ring import load_ring_ref                                        # noqa: E402
from .run_ring_indep import ring_indep                                     # noqa: E402
from .dehom_rm import (build_rm_bundle, stress_at_points, disp_at_points,  # noqa: E402
                       _macro_fields, _rm_shell_strain)
from .emit_abd import emit_station_abd, load_station_abd                   # noqa: E402
from .fe_jax.strip_RM import rm_timoshenko_6x6                             # noqa: E402
from .fe_jax.msg_rm_timo import timoshenko_rm                              # noqa: E402
from .fe_jax.orient_plot import plot_orient, auto_emit                     # noqa: E402
from .solid_props import (build_solid_bundle, ring_solid,                  # noqa: E402
                          elastic_constants, GBAR_ORDER)
