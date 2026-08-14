"""opensg_shell -- the MSG SHELL engine (Reissner-Mindlin shell SGs ->
beam / equivalent-solid macro models), split file-per-concern like
opensg_solid:

    sg_mesh.py         SG input loaders (1-D ring / 3-D segment yaml,
                       reference conventions), topological boundary
                       extraction, orientation-frame reports and PNGs
    sg_materials.py    per-section ABD 6x6 + transverse-shear G 2x2 laws,
                       per-station wall-law ABDG emitter
    sg_assembly.py     shell element operators + assembly: the production
                       6-DOF drilling element, 5-DOF general element,
                       MITC tying, k22/kg corrections, Dirichlet solvers
    sg_periodicity.py  periodic assembly map (6 DOF/node)
    sg_homo.py         homogenization drivers: ring -> Timoshenko 6x6,
                       RM bundle, cross-section -> equivalent 3-D solid,
                       3-D shell SG -> C3D, aperiodic/tapered segment
    sg_dehom.py        RM two-step dehomogenization / recovery
    sg_junction.py     junction micro/microcell corrections
    fe_jax/            msg_* kernels: materials/ABD, RM operators,
                       Timoshenko finalization (msg_solver), KL bundle,
                       recovery, orientation plot
    helper/            format converters

Every model writes a timed SwiftComp-layout .out; examples own all paths
(yaml in -> .out back).

The x64 flag below is LOAD-BEARING: every solver in this package
assembles/factorizes in float64; without it JAX silently degrades to
float32 and the 6x6 digits are wrong.
"""
import os as _os

import jax
import numpy as _np

_np.set_printoptions(precision=5)   # the examples' shared print default

jax.config.update("jax_enable_x64", True)

# persistent XLA compilation cache (same setup as opensg_solid.__init__): cold runs
# pay each kernel compile once per machine; warm runs load the cached binaries --
# this is what makes the msg ABDG / ring / segment kernels start fast.
_CACHE_DIR = _os.environ.get("JAX_COMPILATION_CACHE_DIR",
                             _os.path.expanduser("~/.cache/opensg_jax"))
try:
    _os.makedirs(_CACHE_DIR, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except Exception:
    pass    # the cache is an optimization; the engine must run without it

# point pypardiso straight at the conda MKL runtime: without PYPARDISO_MKL_RT its
# wrapper falls back to a recursive sys.prefix glob (~3 s of scandir on NFS homes)
# on EVERY first solve of a process
if "PYPARDISO_MKL_RT" not in _os.environ:
    import glob as _glob
    import sys as _sys
    _mkl = sorted(_glob.glob(_os.path.join(_sys.prefix, "lib", "libmkl_rt.so*")))
    if _mkl:
        _os.environ["PYPARDISO_MKL_RT"] = _mkl[-1]

from .sg_mesh import (load_ring, ring_6dof, ring_5dof, LBL,                # noqa: E402
                      load_ring_ref)
from .sg_homo import (ring_indep, build_rm_bundle, build_solid_bundle,     # noqa: E402
                      ring_solid, elastic_constants, GBAR_ORDER,
                      shell_sg3d, segment_timo_from_3dyaml)
from .sg_dehom import (stress_at_points, disp_at_points,                   # noqa: E402
                       ring_wall_strains, _macro_fields, _rm_shell_strain)
from .sg_dehom_junction import (flag_recovery_points,                      # noqa: E402
                                junction_families)
from .sg_materials import emit_station_abd, load_station_abd               # noqa: E402
from .fe_jax.msg_rm_timo import timoshenko_rm                              # noqa: E402
from .fe_jax.orient_plot import plot_orient, auto_emit                     # noqa: E402

from .sg_blade import Blade, opensg_blade                                                # noqa: E402,F401
