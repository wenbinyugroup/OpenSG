"""Test that all package imports work correctly."""

import pytest


def test_import_opensg():
    """Test that the main opensg package can be imported."""
    import opensg
    assert opensg.__version__ == "0.1.0"


def test_import_core():
    """Test that core modules can be imported."""
    from opensg.core import compute_ABD_matrix, compute_timo_boun, compute_stiffness
    assert callable(compute_ABD_matrix)
    assert callable(compute_timo_boun)
    assert callable(compute_stiffness)


def test_import_mesh():
    """Test that mesh modules can be imported."""
    from opensg.mesh import BladeMesh, SegmentMesh
    assert BladeMesh is not None
    assert SegmentMesh is not None


def test_import_io():
    """Test that I/O modules can be imported."""
    from opensg.io import load_yaml, write_yaml
    assert callable(load_yaml)
    assert callable(write_yaml)


def test_import_utils():
    """Test that utility modules can be imported."""
    from opensg.utils import solve_ksp, compute_nullspace, local_frame
    assert callable(solve_ksp)
    assert callable(compute_nullspace)
    assert callable(local_frame)


def test_package_structure():
    """Test that the package structure is correct."""
    import opensg
    
    # Check that main functions are available
    assert hasattr(opensg, 'BladeMesh')
    assert hasattr(opensg, 'load_yaml')
    assert hasattr(opensg, 'compute_ABD_matrix')
    assert hasattr(opensg, 'solve_ksp')
    
    # Check that __all__ is defined
    assert hasattr(opensg, '__all__')
    assert isinstance(opensg.__all__, list)
    assert len(opensg.__all__) > 0 