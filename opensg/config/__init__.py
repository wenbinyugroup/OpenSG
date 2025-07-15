"""Configuration management for OpenSG.

This module contains configuration classes and functions for:
- Default parameters
- Material properties
- Solver settings
- Mesh generation parameters
"""

from .defaults import (
    SOLVER_SETTINGS,
    MESH_SETTINGS,
    DEFAULT_MATERIALS,
    ANALYSIS_SETTINGS,
    OUTPUT_SETTINGS
)
from .manager import ConfigManager

__all__ = [
    'SOLVER_SETTINGS',
    'MESH_SETTINGS', 
    'DEFAULT_MATERIALS',
    'ANALYSIS_SETTINGS',
    'OUTPUT_SETTINGS',
    'ConfigManager'
] 