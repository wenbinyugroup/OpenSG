from os.path import join

import opensg
import numpy as np
import time


mesh_yaml = join("data", "bar_urc_shell_mesh.yaml")
mesh_data = opensg.load_yaml(mesh_yaml)

blade_mesh = opensg.BladeMesh(mesh_data)
stiffness_matrices = []
for i in range(1, 10):
 
    segment_mesh = blade_mesh.generate_segment_mesh(segment_index=i, filename="section.msh")

    ABD = segment_mesh.compute_ABD()

    stiffness_matrix_l, stiffness_matrix_r = segment_mesh.compute_stiffness_EB_boundary(ABD)
    
    stiffness_matrices.append((stiffness_matrix_l, stiffness_matrix_r))
    
pause