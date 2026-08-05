"""opensg_solid -- the GENERAL structure-gene engine (beam / plate /
3-D-elastic macro models from a 1-D/2-D/3-D SG, SwiftComp .sc or yaml
input), split file-per-concern:
    sg_materials.py  6x6 stiffness tables (per material / per element)
    sg_mesh.py       load_sg_input, _cell_basis, plot_sg_mesh
    sg_assembly.py   element kernels, periodic assembly, preconditioners,
                     sparse/KKT solvers
    sg_homo.py       homogenization drivers + plate_homo_2d
    sg_dehom.py      recovery kernels + plate_dehom_2d, export_gauss
Examples import opensg_solid.sg_homo / sg_dehom directly.  The beam
(n_model=1) Timoshenko/KKT machinery was lifted from the OneDrive
Beam_solid.py (Claude_code/); provenance lives in the sg_* docstrings.

This package __init__ carries the process-level setup every submodule
relies on:
- the legacy fe_jax core on sys.path ($FE_JAX_CORE, default
  ~/OpenSG_2.0) -- import the engine THROUGH the package;
- the x64 switch, LOAD-BEARING (see rm_plate/__init__.py);
- the persistent JAX compilation cache (the original SSDM script's
  setup): cold runs pay each XLA compile once, warm runs load the
  binaries from $JAX_COMPILATION_CACHE_DIR (default
  ~/.cache/opensg_jax)."""
import os
import sys

_FE_CORE = os.environ.get("FE_JAX_CORE", os.path.expanduser("~/OpenSG_2.0"))
if _FE_CORE not in sys.path:
    sys.path.insert(0, _FE_CORE)

import jax

jax.config.update("jax_enable_x64", True)

_CACHE_DIR = os.environ.get("JAX_COMPILATION_CACHE_DIR",
                            os.path.expanduser("~/.cache/opensg_jax"))
try:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
except Exception:
    pass    # the cache is an optimization; the engine must run without it
