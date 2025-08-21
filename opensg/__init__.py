"""OpenSG: Open Source Structural Analysis for Wind Turbine Blades.

A Python package for structural analysis of wind turbine blades using
Mixed-Space-Galerkin (MSG) formulations and finite element methods.

Main modules:
- core: Core computation functions (ABD matrices, beam models)
- mesh: Mesh handling and data structures
- io: Input/output operations
- utils: Utility functions and helpers
- config: Configuration management
"""

# Core computations
import opensg.core as core
# from opensg.core.shell import compute_ABD_matrix, compute_timo_boun, compute_stiffness
# from opensg.core.solid import

# I/O operations
import opensg.io as io
# from opensg.io import load_yaml, write_yaml

# Mesh handling
import opensg.mesh as mesh
# from opensg.mesh.blade import (
    # ShellBladeMesh,
    # SolidBladeMesh,
    # ShellBladeSegmentMesh,
    # SolidBladeSegmentMesh,
# )


# Utility functions
import opensg.utils as utils
# from opensg.utils.shared import solve_ksp, compute_nullspace, local_frame_1D, deri_constraint
# from opensg.utils.shell import (
#     solve_eb_boundary, local_boun, local_frame, local_grad, deri, ddot,
#     gamma_d, gamma_e, gamma_h, gamma_l, initialize_array, dof_mapping_quad,
#     generate_boundary_markers, facet_vector_approximation, tangential_projection
# )

__version__ = "0.1.0"

