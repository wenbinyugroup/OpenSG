import unittest
from os.path import abspath, dirname, join
from pathlib import Path
import numpy as np
import opensg
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness
import filecmp
import shutil
import os

testdir = dirname(abspath(str(__file__)))
test_data_dir = Path(join(testdir, "testing_data"))
validation_data_dir = Path(join(testdir, "validation_data"))

class TestSolid(unittest.TestCase):
    def setUp(self):
        return super().setUp()

    def test_solid_baseline_validation(self):
        """Test against baseline results for a solid segment"""
        segment_file = test_data_dir / "bar_urc_npl_2_ar_10-segment_2.yaml"

        section_mesh = SolidSegmentMesh(segment_file)
        
        section_mesh.generate_mesh_file("test_solid_section.msh")
        baseline_mesh_file = validation_data_dir / "test_solid_section.msh"
        assert filecmp.cmp("test_solid_section.msh", baseline_mesh_file)
        
        # Extract material parameters and mesh data
        material_parameters, density = section_mesh.material_database
        meshdata = section_mesh.meshdata
        l_submesh = section_mesh.left_submesh
        r_submesh = section_mesh.right_submesh
        
        # Compute stiffness matrices
        timo_seg_stiffness, V0, V1s = compute_stiffness(
            material_parameters,
            meshdata,
            l_submesh,
            r_submesh
        )
        
        # Validate results against baseline (if baselines exist)
        test_timo_seg_stiffness = np.loadtxt(validation_data_dir / 'test_solid_timo_stiffness.txt')
        test_V0 = np.loadtxt(validation_data_dir / 'test_solid_v0.txt')
        test_V1s = np.loadtxt(validation_data_dir / 'test_solid_v1s.txt')
        
        # print("TROUBLESHOOT SOLID TEST")
        # print(f"Computed stiffness shape: {timo_seg_stiffness.shape}")
        # print(f"Baseline stiffness shape: {test_timo_seg_stiffness.shape}")
        # print(f"Max difference: {(timo_seg_stiffness - test_timo_seg_stiffness).max()}")
        # print(f"Min difference: {(timo_seg_stiffness - test_timo_seg_stiffness).min()}")
        
        assert np.isclose(timo_seg_stiffness, test_timo_seg_stiffness, rtol=1e-03, atol=1e-04).all()
        assert np.isclose(V0, test_V0, rtol=1e-03, atol=1e-04).all()
        assert np.isclose(V1s, test_V1s, rtol=1e-03, atol=1e-04).all()

        # Remove generated files
        if os.path.exists("test_solid_section.msh"):
            os.remove("test_solid_section.msh")
        
        print("Solid baseline validation passed!")
        return

    def test_solid_mesh_file_generation(self):
        """Test that solid mesh file generation works correctly"""
        segment_file = test_data_dir / "bar_urc_npl_2_ar_10-segment_2.yaml"

        segment_mesh = SolidSegmentMesh(str(segment_file))
        
        # Generate mesh file
        output_file = "test_solid_mesh_output.msh"
        segment_mesh.generate_mesh_file(output_file)
        
        assert os.path.exists(output_file), "Mesh file was not created"
        
        # Basic validation of file content
        with open(output_file, 'r') as f:
            content = f.read()
            assert "$MeshFormat" in content
            assert "$Nodes" in content
            assert "$Elements" in content
            assert "$EndElements" in content
        
        os.remove(output_file)
        
        return


def run_solid_workflow():
    """This function regenerates the solid test results. Use this if updates to the code have
    changed the expected outputs and these new outputs are what should be tested against.
    """

    segment_file = test_data_dir / "bar_urc_npl_2_ar_10-segment_2.yaml"
    
    section_mesh = SolidSegmentMesh(str(segment_file))
    
    section_mesh.generate_mesh_file(validation_data_dir / "test_solid_section.msh")
    
    # Extract material parameters and mesh data
    material_parameters, density = section_mesh.material_database
    meshdata = section_mesh.meshdata
    l_submesh = section_mesh.left_submesh
    r_submesh = section_mesh.right_submesh
    
    # Compute stiffness matrices
    timo_seg_stiffness, V0, V1s = compute_stiffness(
        material_parameters,
        meshdata,
        l_submesh,
        r_submesh
    )
    
    np.savetxt(validation_data_dir / 'test_solid_timo_stiffness.txt', timo_seg_stiffness)
    np.savetxt(validation_data_dir / 'test_solid_v0.txt', V0)
    np.savetxt(validation_data_dir / 'test_solid_v1s.txt', V1s)
    
    return


if __name__ == "__main__":
    # unittest.main()
    run_solid_workflow() 