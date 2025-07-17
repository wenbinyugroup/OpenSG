"""Core computation modules for OpenSG.

This module contains the main computational functions for:
- ABD matrix computations
- Timoshenko beam model
- General solvers
"""

from opensg.core import shell, solid

__all__ = ['compute_ABD_matrix', 'compute_timo_boun', 'compute_stiffness'] 