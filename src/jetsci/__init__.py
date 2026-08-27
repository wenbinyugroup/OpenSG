import jax

jax.config.update("jax_enable_x64", True)

import pathlib
jax.config.update("jax_compilation_cache_dir", str(pathlib.Path(__file__).parent.resolve() / "__jax_cache__"))

# the nonlinear solve stack rides on the cupy/CUDA-bound PETSc
# primitives; a CPU-only install still gets the pure option plumbing
# (petsc_ksp.options), which opensg_solid.sg_gamg consumes
try:
    from .solve import *
except ImportError:
    pass
