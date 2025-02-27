import unittest
from os.path import abspath, dirname, join
import numpy as np
import opensg
import filecmp


testdir = dirname(abspath(str(__file__)))
datadir = join(testdir, "data")

class TestExample(unittest.TestCase):
    def setUp(self):
        return super().setUp()
    
    def test_workflow(self):
        # Setup
        mesh_yaml = join(datadir, "bar_urc_shell_mesh.yaml")
        mesh_data = opensg.load_yaml(mesh_yaml)
        blade_mesh = opensg.BladeMesh(mesh_data)
        
        # Mesh
        section_mesh = blade_mesh.generate_segment_mesh(segment_index=1, filename="section.msh")
        assert filecmp.cmp("section.msh", "test_section.msh")
        
        # ABD
        abd = section_mesh.compute_ABD()
        abd_concat = np.concat(abd)
        
        expected_abd = np.loadtxt(join(datadir, "test_abd.txt"))
        assert np.isclose(abd_concat, expected_abd).all()
        
        # timoshenko stiffness
        seg_stiffness, lbound_stiffness = section_mesh.compute_timo_stiffness_segment(abd)
        
        expected_seg_stiffness = np.loadtxt(join(datadir, 'test_seg_stiffness.txt'))
        expected_lbound_stiffness = np.loadtxt(join(datadir, 'test_lbound_stiffness.txt'))
        assert np.isclose(seg_stiffness, expected_seg_stiffness).all()
        assert np.isclose(lbound_stiffness, expected_lbound_stiffness).all()


def run_workflow():
    """This function regenerates the test results. Use this if updates to the code have
    changed the expected outputs and these new outputs are what should be tested against.
    """
    # Setup
    mesh_yaml = join(datadir, "bar_urc_shell_mesh.yaml")
    mesh_data = opensg.load_yaml(mesh_yaml)
    blade_mesh = opensg.BladeMesh(mesh_data)
    
    # Mesh
    section_mesh = blade_mesh.generate_segment_mesh(segment_index=1, filename=join(datadir, "test_section.msh"))
    
    # ABD
    abd = section_mesh.compute_ABD()
    abd_concat = np.concat(abd)
    
    np.savetxt(join(datadir, 'test_abd.txt'), abd_concat) 
    
    # Timoshenko stiffness
    seg_stiffness, lbound_stiffness = section_mesh.compute_timo_stiffness_segment(abd)
    
    np.savetxt(join(datadir, 'test_seg_stiffness.txt'), seg_stiffness)
    np.savetxt(join(datadir, 'test_lbound_stiffness.txt'), lbound_stiffness)
    
    return
    
if __name__ == "__main__":
    # unittest.main()
    run_workflow()