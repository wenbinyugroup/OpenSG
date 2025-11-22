"""Mesh handling modules for OpenSG.

This module contains classes and functions for:
- Blade mesh management
- Segment mesh operations
- Mesh data structures
"""

from opensg.mesh.segment import (
    ShellSegmentMesh,
    SolidSegmentMesh,
    ShellBounMesh,
    SolidBounMesh,
)


__all__ = [
    "ShellSegmentMesh",
    "SolidSegmentMesh",
    "ShellBounMesh",
    "SolidBounMesh",
]
