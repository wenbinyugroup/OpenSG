"""Utility functions for OpenSG.

This module contains helper functions for:
- Mathematical operations
- Mesh utilities
- Solver utilities
- Local frame computations
"""

from .shell import (
    solve_eb_boundary,
    local_boun, local_frame, local_grad,
    deri, ddot, gamma_d, gamma_e, gamma_h, gamma_l,
    initialize_array, dof_mapping_quad, generate_boundary_markers,
    facet_vector_approximation, tangential_projection
)

from .solid import (
    local_boun, gamma_e, gamma_h, gamma_l,
    initialize_array, dof_mapping_quad, generate_boundary_markers,
)

from .shared import (
    deri_constraint, local_frame_1D, solve_ksp, compute_nullspace
)

__all__ = [
    'solve_ksp', 'solve_eb_boundary', 'compute_nullspace',
    'local_boun', 'local_frame_1D', 'local_frame', 'local_grad',
    'deri', 'ddot', 'gamma_d', 'gamma_e', 'gamma_h', 'gamma_l',
    'initialize_array', 'dof_mapping_quad', 'generate_boundary_markers',
    'deri_constraint', 'facet_vector_approximation', 'tangential_projection'
] 