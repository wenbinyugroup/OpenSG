"""opensg_shell.fe_jax -- the msg_* kernel subpackage (OpenSG-authored):

    msg_materials.py        ABD / 6x6 stiffness tables, reference shifts
    msg_transverse_shear.py wall transverse-shear G, 8x8 plate law
    msg_mesh.py             1-D shell yaml mesh utilities
    msg_rm.py, msg_rm_timo.py  RM operators + 5-DOF Timoshenko assembly
    msg_solver.py           KKT solve, V1 RHS, Timoshenko finalization
    msg_hermite.py          Hermite C1 KL (thin-walled) pipeline
    msg_dehom.py            shell strain recovery + plate dehom
    orient_plot.py          canonical e1/e2/e3 orientation figure
"""
import jax

jax.config.update("jax_enable_x64", True)

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

