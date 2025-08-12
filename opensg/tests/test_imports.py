"""Test that all package imports work correctly."""

import unittest


class TestImports(unittest.TestCase):
    """Test case for package imports."""
    
    def test_import_opensg(self):
        """Test that the main opensg package can be imported."""
        import opensg
        self.assertEqual(opensg.__version__, "0.1.0")

    def test_import_core(self):
        """Test that core modules can be imported."""
        from opensg.core.shell import compute_ABD_matrix, compute_timo_boun, compute_stiffness
        self.assertTrue(callable(compute_ABD_matrix))
        self.assertTrue(callable(compute_timo_boun))
        self.assertTrue(callable(compute_stiffness))

    def test_import_mesh(self):
        """Test that mesh modules can be imported."""
        from opensg.mesh import BladeMesh, SegmentMesh
        self.assertIsNotNone(BladeMesh)
        self.assertIsNotNone(SegmentMesh)

    def test_import_io(self):
        """Test that I/O modules can be imported."""
        from opensg.io import load_yaml, write_yaml
        self.assertTrue(callable(load_yaml))
        self.assertTrue(callable(write_yaml))

    def test_import_utils(self):
        """Test that utility modules can be imported."""
        from opensg.utils.shared import solve_ksp, compute_nullspace, local_frame_1D
        self.assertTrue(callable(solve_ksp))
        self.assertTrue(callable(compute_nullspace))
        self.assertTrue(callable(local_frame_1D))

    def test_package_structure(self):
        """Test that the package structure is correct."""
        import opensg
        
        # Check that main functions are available
        self.assertTrue(hasattr(opensg, 'BladeMesh'))
        self.assertTrue(hasattr(opensg, 'load_yaml'))
        self.assertTrue(hasattr(opensg, 'compute_ABD_matrix'))
        # Note: solve_ksp is not directly exposed at the top level
        self.assertTrue(hasattr(opensg.utils.shared, 'solve_ksp'))
        
        # Check that __all__ is defined
        self.assertTrue(hasattr(opensg, '__all__'))
        self.assertIsInstance(opensg.__all__, list)
        self.assertGreater(len(opensg.__all__), 0)


if __name__ == "__main__":
    unittest.main()
