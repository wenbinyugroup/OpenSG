import jax

jax.config.update("jax_enable_x64", True)

# Legacy FEniCSx-dependent modules (require flax, petsc4py, dolfinx)
try:
    from .np_types import *
    from .basis_quadrature import *
    from .fea import *
    from .linear_elasticity import *
    from .hyperelasticity import *
    from .profiling import *
    from .setup import *
    from .utils import *
    from .sc_to_msh import *
    from .sparse_matrix import *
    from .sparse_linear_solve import *
    from .constraints import *
    from .constraint_system import *
    from .boundary_conditions import *
    #from .periodic_dofmap import *
    #from .multiscale import *
    #from .fiber_mechanics import *
except ImportError:
    pass  # FEniCSx / flax not installed — MSG shell modules still available

# MSG Shell Timoshenko beam homogenization (quadratic Lagrange elements)
from .msg_materials import (
    build_stiffness_6x6,
    rotation_6x6,
    rotated_stiffness_6x6,
    compute_ABD_matrix,
    compute_ABD_CLT,
    plate_dehom_strain,
    plate_stress_at_depth,
    shift_abd_reference,
)
from .msg_transverse_shear import (
    transverse_shear_stiffness,
    plate_8x8,
)
from .msg_mesh import (
    load_yaml,
    read_mesh,
    order_mesh,
    compute_curvature,
    mesh_curvature,
    offset_oml_to_iml,
    element_e3_from_yaml,
)
# Shared FEM infrastructure (quadrature, element geometry, KKT solver,
# Timoshenko assembly) — used by the Hermite C1 TW pipeline.
from .msg_solver import (
    gauss_legendre_01,
    compute_element_geometry,
    solve_fluctuation_field,
    prepare_v1_rhs,
    finalize_v1_and_compute_deff,
)
# Hermite C1 cubic — the MSG thin-walled (TW) Timoshenko method.
from .msg_hermite import (
    hermite_shape_functions,
    hermite_strain_operators,
    make_hermite_mesh,
    build_hermite_dof_map,
    compress_hermite_dofs,
    assemble_system_matrices_hermite,
    build_constraints_hermite,
    timoshenko_from_yaml,
    solve_tw_from_yaml,
)
# Two-step dehomogenization (shell strain recovery + plate dehom)
from .msg_dehom import (
    recover_shell_strains,
    dehomogenize,
    stress_at_points,
)
# Canonical e1/e2/e3 orientation plot -- COMPULSORY output on every homogenization run.
from .orient_plot import (
    plot_orient,
    auto_emit,
)

