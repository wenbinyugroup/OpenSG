import unittest

import numpy as np

example_data_path = ""

# class TestExample(unittest.TestCase):
    
#     # def setup
#     # def close
#     def setUp(self):
#         mesh_yaml = join("data", "bar_urc_shell_mesh.yaml")
#         mesh_data = opensg.load_yaml(mesh_yaml)

#         self.blade_mesh = opensg.BladeMesh(mesh_data)
#         return super().setUp()

#     def test_example(self):
#         # load expected values
#         expected_ABD = np.loadtxt("")
#         expected_stiffness = np.loadtxt("")
        
#         section_mesh = self.blade_mesh.generate_segment_mesh(segment_index=1, filename="section.msh")
#         ABD = section_mesh.compute_ABD()
#         assert np.isclsoe(ABD, expected_ABD).all()
        
#         stiffness_matrix = section_mesh.compute_stiffness_EB(ABD)
#         assert np.isclsoe(stiffness_matrix, expected_stiffness).all()


class TestImport(unittest.TestCase):
    def test_import(self):
        import opensg
        
