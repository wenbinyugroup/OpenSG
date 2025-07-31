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
from opensg.core.shell import compute_ABD_matrix, compute_timo_boun, compute_stiffness
# from opensg.core.solid import 

# I/O operations
import opensg.io as io
from opensg.io import load_yaml, write_yaml

# Mesh handling
from opensg.mesh.blade import BladeMesh
from opensg.mesh.solidblade import SolidBladeMesh
from opensg.mesh.segment import SegmentMesh
from opensg.mesh.solidsegment import SolidSegmentMesh


# Utility functions
import opensg.utils as utils

import opensg.core as core

__version__ = "0.1.0"

__all__ = [
    # Core functions
    'compute_ABD_matrix', 'compute_timo_boun', 'compute_stiffness',
    
    # I/O functions
    'load_yaml', 'write_yaml',
    
    # Classes
    'BladeMesh', 'ConfigManager',
    
    # Utility functions
    'solve_ksp', 'solve_eb_boundary', 'compute_nullspace',
    'local_boun', 'local_frame_1D', 'local_frame', 'local_grad',
    'deri', 'ddot', 'gamma_d', 'gamma_e', 'gamma_h', 'gamma_l',
    'initialize_array', 'dof_mapping_quad', 'generate_boundary_markers',
    'deri_constraint', 'facet_vector_approximation', 'tangential_projection'
]