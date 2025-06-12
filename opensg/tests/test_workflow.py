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
        assert filecmp.cmp("section.msh", join(datadir,"test_section.msh"))
        
        # ABD
        abd = section_mesh.compute_ABD()
        abd_concat = np.concat(abd)
        
        expected_abd = np.loadtxt(join(datadir, "test_abd.txt"))
        assert np.isclose(abd_concat, expected_abd).all()
        
        # Timoshenko stiffness
        timo_seg_stiffness, eb_seg_stiffness, l_timo_stiffness, r_timo_stiffness = section_mesh.compute_stiffness(abd)
        
        # Validate
        test_timo_seg_stiffness = np.loadtxt(join(datadir, 'test_timo_seg_stiffness.txt'))
        test_eb_seg_stiffness = np.loadtxt(join(datadir, 'test_eb_seg_stiffness.txt'))
        test_l_timo_stiffness = np.loadtxt(join(datadir, 'test_l_timo_stiffness.txt'))
        test_r_timo_stiffness = np.loadtxt(join(datadir, 'test_r_timo_stiffness.txt'))
        
        print("TROUBLESHOOT TEST")
        print((timo_seg_stiffness - test_timo_seg_stiffness).max())
        print((eb_seg_stiffness - test_eb_seg_stiffness).max())
        print((l_timo_stiffness - test_l_timo_stiffness).max())
        print((r_timo_stiffness - test_r_timo_stiffness).max())
        
        print((timo_seg_stiffness - test_timo_seg_stiffness).min())
        print((eb_seg_stiffness - test_eb_seg_stiffness).min())
        print((l_timo_stiffness - test_l_timo_stiffness).min())
        print((r_timo_stiffness - test_r_timo_stiffness).min())
        
        assert np.isclose(timo_seg_stiffness, test_timo_seg_stiffness, rtol=1e-03, atol=1e-04).all()
        assert np.isclose(eb_seg_stiffness, test_eb_seg_stiffness, rtol=1e-03, atol=1e-04).all()
        assert np.isclose(l_timo_stiffness, test_l_timo_stiffness, rtol=1e-03, atol=1e-04).all()
        assert np.isclose(r_timo_stiffness, test_r_timo_stiffness, rtol=1e-03, atol=1e-04).all()
        
        return


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
    timo_seg_stiffness, eb_seg_stiffness, l_timo_stiffness, r_timo_stiffness = section_mesh.compute_stiffness(abd)
    
    np.savetxt(join(datadir, 'test_timo_seg_stiffness.txt'), timo_seg_stiffness)
    np.savetxt(join(datadir, 'test_eb_seg_stiffness.txt'), eb_seg_stiffness)
    np.savetxt(join(datadir, 'test_l_timo_stiffness.txt'), l_timo_stiffness)
    np.savetxt(join(datadir, 'test_r_timo_stiffness.txt'), r_timo_stiffness)
    
    return
    
if __name__ == "__main__":
    unittest.main()
    # run_workflow()