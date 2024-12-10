import unittest
from os.path import abspath, dirname, join
import numpy as np
import opensg
import filecmp


example_data_path = ""
testdir = dirname(abspath(str(__file__)))
datadir = join(testdir, "data")

class TestExample(unittest.TestCase):
    
    # def setup
    # def close
    def setUp(self):
        mesh_yaml = join(datadir, "bar_urc_shell_mesh.yaml")
        mesh_data = opensg.load_yaml(mesh_yaml)

        self.blade_mesh = opensg.BladeMesh(mesh_data)
        return super().setUp()
    
    def test_segment_creation(self):
        mesh_filename = "section.msh"
        expected_mesh_filename = join(datadir, 'test_section.msh')
        
        _ = self.blade_mesh.generate_segment_mesh(segment_index=1, filename=mesh_filename)
        
        assert filecmp.cmp(mesh_filename, expected_mesh_filename)
    
    def test_ABD_matrix(self):
        section_mesh = self.blade_mesh.generate_segment_mesh(segment_index=1, filename="section.msh")
        abd = section_mesh.compute_ABD()
        
        abd_concat = np.concat(abd)
        # np.savetxt(join(datadir, 'abd_test.txt'), abd_concat) # use to reset ground truth
        
        expected_abd = np.loadtxt(join(datadir, 'abd_test.txt'))
        assert np.isclose(abd_concat, expected_abd).all()
    
    def test_timo_stiffness(self):
        pass
    

    def test_EB_stiffness(self):
        # load expected values
        # expected_ABD = np.loadtxt("")
        # expected_stiffness = np.loadtxt("")
        
        section_mesh = self.blade_mesh.generate_segment_mesh(segment_index=1, filename="section.msh")
        ABD = section_mesh.compute_ABD()
        # assert np.isclsoe(ABD, expected_ABD).all()
        
        computed_stiffness_matrix = section_mesh.compute_stiffness_EB(ABD)
        # np.savetxt(join(datadir, 'stiffness_test.txt'), computed_stiffness_matrix) # use to reset ground truth
        
        assert computed_stiffness_matrix.shape == (4,4)
        
        expected_stiffness_matrix = np.loadtxt(join(datadir, 'stiffness_test.txt'))
        assert np.isclose(computed_stiffness_matrix, expected_stiffness_matrix).all()
        


"""
[[ 5.6138e+09  7.8930e+07 -9.1787e+07 -6.2902e+08]
 [ 7.8930e+07  2.2724e+10 -6.0954e+08  2.0795e+08]
 [-9.1787e+07 -6.0954e+08  1.0064e+10  9.9959e+08]
 [-6.2902e+08  2.0795e+08  9.9959e+08  1.2617e+10]]
"""

if __name__ == "__main__":
    unittest.main()