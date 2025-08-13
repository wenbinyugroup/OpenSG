"""Input/output modules for OpenSG.

This module contains functions for:
- Reading and writing mesh data
- File format handling
- Data validation
"""

from .io import load_yaml, write_yaml, validate_mesh_data
from .util import generate_segment_shell_mesh_files
