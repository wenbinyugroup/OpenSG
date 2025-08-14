#!/usr/bin/env python3
"""
Unit tests for comparing solid segment meshes created using SolidBladeMesh vs StandaloneSolidSegmentMesh.

This test suite ensures that the new standalone solid segment workflow produces identical results
to the original SolidBladeMesh-based workflow.

Note: In the solid workflow, SolidBladeMesh takes a segment YAML file directly,
not a full blade mesh like in the shell workflow.
"""

from os.path import abspath, dirname, join
from pathlib import Path
import unittest
import opensg
import numpy as np
from pathlib import Path
import tempfile
import shutil
import yaml

from opensg.mesh.blade import SolidBladeMesh
from opensg.mesh.segment import SolidSegmentMesh


testdir = dirname(abspath(str(__file__)))
test_data_dir = Path(join(testdir, "testing_data"))

class TestSolidSegmentComparison(unittest.TestCase):
    """Test cases for comparing SolidBladeMesh and StandaloneSolidSegmentMesh segment creation."""

    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        # Test with a known solid segment mesh file
        cls.segment_mesh_file = test_data_dir / "bar_urc_npl_2_ar_10-segment_2.yaml"

        if not Path(cls.segment_mesh_file).exists():
            raise FileNotFoundError(
                f"Solid segment mesh file not found: {cls.segment_mesh_file}"
            )

        # Create temporary directory for test files
        cls.test_dir = Path(tempfile.mkdtemp(prefix="test_solid_comparison_"))

        # Load mesh data from segment YAML file
        mesh_data = opensg.load_yaml(cls.segment_mesh_file)
        cls.blade_mesh = SolidBladeMesh(mesh_data)

        # Generate segment mesh file
        segment_index = 0
        cls.segment_msh_file = cls.test_dir / "solid_segment.msh"
        cls.blade_segment = cls.blade_mesh.generate_segment_mesh(
            segment_index, str(cls.segment_msh_file)
        )

        # Create standalone segment
        cls.standalone_segment = SolidSegmentMesh(str(cls.segment_mesh_file))

        # Debug: Print information about element sets
        print(f"\nDEBUG: Element set comparison")
        print(f"Blade mesh element sets: {len(cls.blade_mesh.sets['element'])} sets")
        for i, es in enumerate(cls.blade_mesh.sets["element"]):
            print(
                f"  Blade Set {i}: name='{es['name']}', labels[0]={es['labels'][0]}, num_labels={len(es['labels'])}"
            )
        print(f"Blade mesh elLayID unique values: {np.unique(cls.blade_mesh.elLayID)}")
        print(f"Blade mesh elLayID shape: {cls.blade_mesh.elLayID.shape}")

        print(
            f"Standalone element sets: {len(cls.standalone_segment.sets['element'])} sets"
        )
        for i, es in enumerate(cls.standalone_segment.sets["element"]):
            print(
                f"  Standalone Set {i}: name='{es['name']}', labels[0]={es['labels'][0]}, num_labels={len(es['labels'])}"
            )
        print(
            f"Standalone elLayID unique values: {np.unique(cls.standalone_segment.elLayID)}"
        )
        print(f"Standalone elLayID shape: {cls.standalone_segment.elLayID.shape}")

        # Check if this is a single-material segment
        if len(np.unique(cls.standalone_segment.elLayID)) == 1:
            print(
                f"NOTE: This appears to be a single-material segment (all layup IDs = {np.unique(cls.standalone_segment.elLayID)[0]})"
            )
            print(
                f"This is expected if the segment YAML file contains only elements from one material"
            )

        # Debug: Print original_cell_index information
        print(
            f"Blade segment original_cell_index shape: {cls.blade_segment.original_cell_index.shape}"
        )
        print(
            f"Blade segment original_cell_index first 10: {cls.blade_segment.original_cell_index[:10]}"
        )
        print(
            f"Standalone segment original_cell_index shape: {cls.standalone_segment.original_cell_index.shape}"
        )
        print(
            f"Standalone segment original_cell_index first 10: {cls.standalone_segment.original_cell_index[:10]}"
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        if hasattr(cls, "test_dir") and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def test_segment_creation_methods(self):
        """Test that both segment creation methods work."""
        # Verify both segments were created successfully
        self.assertIsNotNone(self.blade_segment)
        self.assertIsNotNone(self.standalone_segment)

        # Verify basic mesh properties
        self.assertIsNotNone(self.blade_segment.mesh)
        self.assertIsNotNone(self.standalone_segment.mesh)

        print(f"✓ Both segment creation methods work")
        print(
            f"  Blade segment: {self.blade_segment.mesh.geometry.x.shape[0]} nodes, "
            f"{self.blade_segment.mesh.topology.index_map(3).size_local} elements"
        )
        print(
            f"  Standalone segment: {self.standalone_segment.mesh.geometry.x.shape[0]} nodes, "
            f"{self.standalone_segment.mesh.topology.index_map(3).size_local} elements"
        )

    def test_node_coordinates(self):
        """Test that node coordinates are identical between methods."""
        # Compare node coordinates
        blade_nodes = self.blade_segment.mesh.geometry.x
        standalone_nodes = self.standalone_segment.mesh.geometry.x

        # Check shapes
        self.assertEqual(
            blade_nodes.shape,
            standalone_nodes.shape,
            "Node coordinate arrays should have the same shape",
        )

        # Check values
        nodes_match = np.allclose(blade_nodes, standalone_nodes, rtol=1e-10, atol=1e-10)
        self.assertTrue(nodes_match, "Node coordinates should be identical")

        print(f"✓ Node coordinates match: {blade_nodes.shape[0]} nodes")

    def test_element_connectivity(self):
        """Test that element connectivity is identical between methods."""
        # Get element connectivity
        blade_elements = self.blade_segment.mesh.topology.connectivity(3, 0).array
        standalone_elements = self.standalone_segment.mesh.topology.connectivity(
            3, 0
        ).array

        # Check shapes
        self.assertEqual(
            blade_elements.shape,
            standalone_elements.shape,
            "Element connectivity arrays should have the same shape",
        )

        # Check values
        elements_match = np.array_equal(blade_elements, standalone_elements)
        self.assertTrue(elements_match, "Element connectivity should be identical")

        print(f"✓ Element connectivity match: {blade_elements.shape[0]} elements")

    def test_material_database(self):
        """Test that material database is identical between methods."""
        # Compare material databases
        blade_materials = self.blade_segment.material_database
        standalone_materials = self.standalone_segment.material_database

        # Check structure
        self.assertEqual(
            len(blade_materials),
            len(standalone_materials),
            "Material databases should have the same structure",
        )

        # Check material parameters
        blade_mat_params, blade_density = blade_materials
        standalone_mat_params, standalone_density = standalone_materials

        # Check number of materials
        self.assertEqual(
            len(blade_mat_params),
            len(standalone_mat_params),
            "Should have the same number of materials",
        )

        # Check each material parameter array
        for i, (blade_param, standalone_param) in enumerate(
            zip(blade_mat_params, standalone_mat_params)
        ):
            self.assertTrue(
                np.allclose(blade_param, standalone_param, rtol=1e-10, atol=1e-10),
                f"Material parameters {i} should match",
            )

        # Check density values
        self.assertTrue(
            np.allclose(blade_density, standalone_density, rtol=1e-10, atol=1e-10),
            "Density values should match",
        )

        print(f"✓ Material database match: {len(blade_mat_params)} materials")

    def test_orientation_vectors(self):
        """Test that orientation vectors are identical between methods."""
        # Compare orientation vectors
        vectors_to_check = ["EE1", "EE2", "EE3"]

        for vector_name in vectors_to_check:
            blade_vector = getattr(self.blade_segment, vector_name)
            standalone_vector = getattr(self.standalone_segment, vector_name)

            # Extract numerical values from DOLFINx Functions
            blade_values = blade_vector.x.array
            standalone_values = standalone_vector.x.array

            self.assertTrue(
                np.allclose(blade_values, standalone_values, rtol=1e-10, atol=1e-10),
                f"Orientation vector {vector_name} should match",
            )

        print(f"✓ Orientation vectors match: {len(vectors_to_check)} vectors")

    def test_boundary_submeshes(self):
        """Test that boundary submeshes are identical between methods."""
        # Compare left submeshes
        blade_left = self.blade_segment.left_submesh
        standalone_left = self.standalone_segment.left_submesh

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

        # Compare right submeshes
        blade_right = self.blade_segment.right_submesh
        standalone_right = self.standalone_segment.right_submesh

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

        print("✓ Boundary submeshes match")

    def test_meshdata_structure(self):
        """Test that meshdata structure is identical between methods."""
        # Compare meshdata structure
        blade_meshdata = self.blade_segment.meshdata
        standalone_meshdata = self.standalone_segment.meshdata

        # Check keys
        blade_keys = set(blade_meshdata.keys())
        standalone_keys = set(standalone_meshdata.keys())
        self.assertEqual(
            blade_keys, standalone_keys, "Meshdata should have the same keys"
        )

        print(f"✓ Meshdata structure match: {list(blade_keys)}")

    def test_layup_ids(self):
        """Test that layup IDs are identical between methods."""
        # Compare layup IDs
        blade_layup_ids = self.blade_segment.elLayID
        standalone_layup_ids = self.standalone_segment.elLayID

        # Check that they have the same shape
        self.assertEqual(
            blade_layup_ids.shape,
            standalone_layup_ids.shape,
            "Layup ID arrays should have the same shape",
        )

        # Debug: Print some information about the layup IDs
        print(
            f"Blade layup IDs: dtype={blade_layup_ids.dtype}, unique values={np.unique(blade_layup_ids)}"
        )
        print(
            f"Standalone layup IDs: dtype={standalone_layup_ids.dtype}, unique values={np.unique(standalone_layup_ids)}"
        )
        print(f"Blade layup IDs first 10: {blade_layup_ids[:10]}")
        print(f"Standalone layup IDs first 10: {standalone_layup_ids[:10]}")

        # Check if this is a single-material segment
        if len(np.unique(standalone_layup_ids)) == 1:
            print(
                f"NOTE: This is a single-material segment. Checking that all blade layup IDs match the single value."
            )
            single_value = np.unique(standalone_layup_ids)[0]
            # For single-material segments, all layup IDs should be the same
            layup_ids_match = np.allclose(blade_layup_ids, single_value)
            self.assertTrue(
                layup_ids_match,
                f"All layup IDs should be {single_value} for single-material segment",
            )
        else:
            # For multi-material segments, the arrays should be identical
            layup_ids_match = np.array_equal(blade_layup_ids, standalone_layup_ids)
            self.assertTrue(layup_ids_match, "Layup IDs should be identical")

        print(f"✓ Layup IDs match: {len(blade_layup_ids)} elements")

    def test_solid_workflow_information(self):
        """Test and display solid workflow information."""
        # Get material information
        blade_mat_params, blade_density = self.blade_segment.material_database

        # Verify material names
        self.assertIsNotNone(self.blade_mesh.mat_name, "Material names should exist")
        self.assertGreater(
            len(self.blade_mesh.mat_name), 0, "Should have at least one material"
        )

        # Verify material parameters
        self.assertEqual(
            len(blade_mat_params),
            len(self.blade_mesh.mat_name),
            "Number of material parameters should match number of material names",
        )

        print(f"\nSolid Workflow Information:")
        print(f"  Original segment file: {self.segment_mesh_file}")
        print(f"  Generated MSH file: {self.segment_msh_file}")
        print(f"  Number of materials: {len(blade_mat_params)}")
        print(f"  Material names: {self.blade_mesh.mat_name}")
        print(f"  Blade mesh nodes: {self.blade_mesh.num_nodes}")
        print(f"  Blade mesh elements: {self.blade_mesh.num_elements}")


def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSolidSegmentComparison)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
