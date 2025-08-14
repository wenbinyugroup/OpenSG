#!/usr/bin/env python3
"""
Test script to compare segment meshes created using BladeMesh vs StandaloneSegmentMesh.

This test ensures that the new standalone segment workflow produces identical results
to the original BladeMesh-based workflow.
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from os.path import abspath, dirname, join

import opensg
from opensg.mesh.blade import ShellBladeMesh
from opensg.mesh.segment import ShellSegmentMesh


testdir = dirname(abspath(str(__file__)))
test_data_dir = Path(join(testdir, "testing_data"))


class TestSegmentComparison(unittest.TestCase):
    """Test cases for comparing BladeMesh and StandaloneSegmentMesh segment creation."""

    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        # Test with a known blade mesh file
        cls.blade_mesh_yaml = test_data_dir / "bar_urc_shell_mesh.yaml"

        segment_list = [6]

        if not Path(cls.blade_mesh_yaml).exists():
            raise FileNotFoundError(f"Blade mesh file not found: {cls.blade_mesh_yaml}")

        # Create temporary directory for test files
        cls.test_dir = Path(tempfile.mkdtemp(prefix="test_comparison_"))

        # Generate segment files
        print(f"Generating segment files from: {cls.blade_mesh_yaml}")
        cls.segment_files = opensg.io.generate_segment_shell_mesh_files(
            cls.blade_mesh_yaml, segment_list, str(cls.test_dir)
        )

        generated_files = [f for f in cls.test_dir.glob("*.yaml")]
        if not generated_files:
            raise RuntimeError("No segment files generated")

        cls.segment_files = generated_files

        cls.test_file = cls.segment_files[0]
        print(f"Using test file: {cls.test_file}")

        # Extract segment index from filename for comparison
        # Expected format: bar_urc_shell_mesh_segment_X.yaml
        cls.segment_index = int(str(cls.test_file).split("_")[-1].replace(".yaml", ""))
        print(f"Testing segment index: {cls.segment_index}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        if hasattr(cls, "test_dir") and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def test_segment_creation_methods(self):
        """Test that both segment creation methods work."""
        # Method 1: Create segment using BladeMesh
        mesh_data = opensg.load_yaml(self.blade_mesh_yaml)
        blade_mesh = ShellBladeMesh(mesh_data)
        blade_segment = blade_mesh.generate_segment_mesh(self.segment_index)

        # Method 2: Create segment using StandaloneSegmentMesh
        standalone_segment = ShellSegmentMesh(self.test_file)

        # Verify both segments were created successfully
        self.assertIsNotNone(blade_segment)
        self.assertIsNotNone(standalone_segment)

        return blade_segment, standalone_segment

    def test_node_coordinates(self):
        """Test that node coordinates are identical between methods."""
        blade_segment, standalone_segment = self.test_segment_creation_methods()

        # Compare node coordinates
        blade_nodes = blade_segment.mesh.geometry.x
        standalone_nodes = standalone_segment.mesh.geometry.x

        # Check shapes
        self.assertEqual(
            blade_nodes.shape,
            standalone_nodes.shape,
            "Node coordinate arrays should have the same shape",
        )

        # Check values
        nodes_match = np.allclose(blade_nodes, standalone_nodes, rtol=1e-10, atol=1e-10)
        self.assertTrue(nodes_match, "Node coordinates should be identical")

        print(f"Node coordinates match: {blade_nodes.shape[0]} nodes")

    def test_element_connectivity(self):
        """Test that element connectivity is identical between methods."""
        blade_segment, standalone_segment = self.test_segment_creation_methods()

        # Get element connectivity
        blade_elements = blade_segment.mesh.topology.connectivity(2, 0).array
        standalone_elements = standalone_segment.mesh.topology.connectivity(2, 0).array

        # Check shapes
        self.assertEqual(
            blade_elements.shape,
            standalone_elements.shape,
            "Element connectivity arrays should have the same shape",
        )

        # Check values
        elements_match = np.array_equal(blade_elements, standalone_elements)
        self.assertTrue(elements_match, "Element connectivity should be identical")

        print(f"Element connectivity match: {blade_elements.shape[0]} elements")

    def test_material_properties(self):
        """Test that material properties are identical between methods."""
        blade_segment, standalone_segment = self.test_segment_creation_methods()

        # Compare material databases
        blade_materials = blade_segment.blade_mesh.material_database
        standalone_materials = standalone_segment.material_database

        # Check that they have the same materials
        self.assertEqual(
            set(blade_materials.keys()),
            set(standalone_materials.keys()),
            "Material databases should contain the same materials",
        )

        # Check each material's properties
        for material_name in blade_materials.keys():
            blade_mat = blade_materials[material_name]
            standalone_mat = standalone_materials[material_name]

            # Check required properties
            for prop in ["id", "E", "G", "nu"]:
                self.assertIn(
                    prop, blade_mat, f"Blade material {material_name} missing {prop}"
                )
                self.assertIn(
                    prop,
                    standalone_mat,
                    f"Standalone material {material_name} missing {prop}",
                )

                # Compare values
                blade_val = blade_mat[prop]
                standalone_val = standalone_mat[prop]

                if isinstance(blade_val, (list, np.ndarray)):
                    self.assertTrue(
                        np.allclose(blade_val, standalone_val, rtol=1e-10, atol=1e-10),
                        f"Material {material_name} {prop} values should match",
                    )
                else:
                    self.assertEqual(
                        blade_val,
                        standalone_val,
                        f"Material {material_name} {prop} values should match",
                    )

        print(f"Material properties match: {len(blade_materials)} materials")

    def test_layup_properties(self):
        """Test that layup properties are identical between methods."""
        blade_segment, standalone_segment = self.test_segment_creation_methods()

        # Compare layup databases
        blade_layups = blade_segment.layup_database
        standalone_layups = standalone_segment.layup_database

        # Check required keys
        required_keys = ["mat_names", "thick", "angle", "nlay"]
        for key in required_keys:
            self.assertIn(key, blade_layups, f"Blade layup database missing {key}")
            self.assertIn(
                key, standalone_layups, f"Standalone layup database missing {key}"
            )

        # Check that all arrays have the same length
        blade_lengths = [len(blade_layups[key]) for key in required_keys]
        standalone_lengths = [len(standalone_layups[key]) for key in required_keys]

        self.assertEqual(
            blade_lengths,
            standalone_lengths,
            "Layup database arrays should have the same lengths",
        )

        # Check each property array
        for key in required_keys:
            blade_array = blade_layups[key]
            standalone_array = standalone_layups[key]

            if key == "mat_names":
                # For material names (strings), compare directly
                for i, (blade_item, standalone_item) in enumerate(
                    zip(blade_array, standalone_array)
                ):
                    self.assertEqual(
                        blade_item, standalone_item, f"Layup {key}[{i}] should match"
                    )
            elif isinstance(blade_array[0], (list, np.ndarray)):
                # For nested arrays (like mat_names)
                for i, (blade_item, standalone_item) in enumerate(
                    zip(blade_array, standalone_array)
                ):
                    self.assertTrue(
                        np.allclose(
                            blade_item, standalone_item, rtol=1e-10, atol=1e-10
                        ),
                        f"Layup {key}[{i}] should match",
                    )
            else:
                # For simple arrays (like thick, angle, nlay)
                self.assertTrue(
                    np.allclose(blade_array, standalone_array, rtol=1e-10, atol=1e-10),
                    f"Layup {key} should match",
                )

        print(f"Layup properties match: {len(blade_layups['mat_names'])} layups")

    def test_element_orientations(self):
        """Test that element orientations are identical between methods."""
        blade_segment, standalone_segment = self.test_segment_creation_methods()

        # Compare the raw orientation data from the YAML files
        # For blade_segment, we need to extract orientations from the parent blade mesh
        blade_orientations = []
        for i, eo in enumerate(blade_segment.blade_mesh.element_orientations):
            if blade_segment.segment_element_labels[i] > -1:
                o = []
                for k in range(9):
                    o.append(eo[k])
                blade_orientations.append(o)

        standalone_orientations = standalone_segment.element_orientations

        # Check lengths
        self.assertEqual(
            len(blade_orientations),
            len(standalone_orientations),
            "Element orientations should have the same length",
        )

        # Check each orientation
        for i, (blade_orient, standalone_orient) in enumerate(
            zip(blade_orientations, standalone_orientations)
        ):
            self.assertTrue(
                np.allclose(blade_orient, standalone_orient, rtol=1e-10, atol=1e-10),
                f"Element orientation {i} should match",
            )

        print(f"Element orientations match: {len(blade_orientations)} orientations")

    def test_abd_matrices(self):
        """Test that ABD matrices are identical between methods."""
        blade_segment, standalone_segment = self.test_segment_creation_methods()

        # Compute ABD matrices
        blade_abd = blade_segment.compute_ABD()
        standalone_abd = standalone_segment.compute_ABD()

        # Check lengths
        self.assertEqual(
            len(blade_abd),
            len(standalone_abd),
            "ABD matrices should have the same length",
        )

        # Check each ABD matrix
        for i, (blade_matrix, standalone_matrix) in enumerate(
            zip(blade_abd, standalone_abd)
        ):
            self.assertTrue(
                np.allclose(blade_matrix, standalone_matrix, rtol=1e-10, atol=1e-10),
                f"ABD matrix {i} should match",
            )

        print(f"ABD matrices match: {len(blade_abd)} matrices")

    def test_stiffness_matrices(self):
        """Test that stiffness matrices are identical between methods."""
        blade_segment, standalone_segment = self.test_segment_creation_methods()

        # Compute ABD matrices first
        blade_abd = blade_segment.compute_ABD()
        standalone_abd = standalone_segment.compute_ABD()

        # Compute stiffness matrices
        blade_stiffness = blade_segment.compute_stiffness(blade_abd)
        standalone_stiffness = standalone_segment.compute_stiffness(standalone_abd)

        # Check that we get the expected number of results
        self.assertEqual(
            len(blade_stiffness),
            len(standalone_stiffness),
            "Should return the same number of stiffness matrices",
        )

        # Compare each stiffness matrix
        stiffness_names = [
            "Timoshenko",
            "Euler-Bernoulli",
            "Left boundary",
            "Right boundary",
        ]
        for i, (name, blade_matrix, standalone_matrix) in enumerate(
            zip(stiffness_names, blade_stiffness, standalone_stiffness)
        ):
            self.assertTrue(
                np.allclose(blade_matrix, standalone_matrix, rtol=1e-2, atol=1e-2),
                f"{name} stiffness matrix should match",
            )

        print(f"Stiffness matrices match: {len(blade_stiffness)} matrices")

    def test_boundary_submeshes(self):
        """Test that boundary submeshes are identical between methods."""
        blade_segment, standalone_segment = self.test_segment_creation_methods()

        # Compare left submeshes
        blade_left = blade_segment.left_submesh
        standalone_left = standalone_segment.left_submesh

        # Check that both have the required keys
        required_keys = ["mesh", "subdomains", "frame", "facets"]
        for key in required_keys:
            self.assertIn(key, blade_left, f"Blade left submesh missing {key}")
            self.assertIn(
                key, standalone_left, f"Standalone left submesh missing {key}"
            )

        # Compare mesh geometries
        blade_left_nodes = blade_left["mesh"].geometry.x
        standalone_left_nodes = standalone_left["mesh"].geometry.x
        self.assertTrue(
            np.allclose(
                blade_left_nodes, standalone_left_nodes, rtol=1e-10, atol=1e-10
            ),
            "Left submesh node coordinates should match",
        )

        # Compare boundary frames
        for i, (blade_frame_func, standalone_frame_func) in enumerate(
            zip(blade_left["frame"], standalone_left["frame"])
        ):
            blade_frame_values = blade_frame_func.x.array
            standalone_frame_values = standalone_frame_func.x.array
            self.assertTrue(
                np.allclose(
                    blade_frame_values, standalone_frame_values, rtol=1e-10, atol=1e-10
                ),
                f"Left boundary frame function {i} should match",
            )

        # Compare right submeshes
        blade_right = blade_segment.right_submesh
        standalone_right = standalone_segment.right_submesh

        for key in required_keys:
            self.assertIn(key, blade_right, f"Blade right submesh missing {key}")
            self.assertIn(
                key, standalone_right, f"Standalone right submesh missing {key}"
            )

        blade_right_nodes = blade_right["mesh"].geometry.x
        standalone_right_nodes = standalone_right["mesh"].geometry.x
        self.assertTrue(
            np.allclose(
                blade_right_nodes, standalone_right_nodes, rtol=1e-10, atol=1e-10
            ),
            "Right submesh node coordinates should match",
        )

        # Compare boundary frames
        for i, (blade_frame_func, standalone_frame_func) in enumerate(
            zip(blade_right["frame"], standalone_right["frame"])
        ):
            blade_frame_values = blade_frame_func.x.array
            standalone_frame_values = standalone_frame_func.x.array
            self.assertTrue(
                np.allclose(
                    blade_frame_values, standalone_frame_values, rtol=1e-10, atol=1e-10
                ),
                f"Right boundary frame function {i} should match",
            )

        print("Boundary submeshes match")

    def test_orientation_vectors(self):
        """Test that orientation vectors are identical between methods."""
        blade_segment, standalone_segment = self.test_segment_creation_methods()

        # Compare orientation vectors
        vectors_to_check = ["EE1", "EE2", "N"]

        for vector_name in vectors_to_check:
            blade_vector = getattr(blade_segment, vector_name)
            standalone_vector = getattr(standalone_segment, vector_name)

            # Extract numerical values from DOLFINx Functions
            blade_values = blade_vector.x.array
            standalone_values = standalone_vector.x.array

            self.assertTrue(
                np.allclose(blade_values, standalone_values, rtol=1e-10, atol=1e-10),
                f"Orientation vector {vector_name} should match",
            )

        # Compare frame (extract values from each function in the frame)
        for i, (blade_frame_func, standalone_frame_func) in enumerate(
            zip(blade_segment.frame, standalone_segment.frame)
        ):
            blade_frame_values = blade_frame_func.x.array
            standalone_frame_values = standalone_frame_func.x.array
            self.assertTrue(
                np.allclose(
                    blade_frame_values, standalone_frame_values, rtol=1e-10, atol=1e-10
                ),
                f"Frame function {i} should match",
            )

        print("Orientation vectors match")


def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSegmentComparison)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
