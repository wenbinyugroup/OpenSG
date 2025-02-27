from os.path import join

# import pynumad
import opensg
import numpy as np
import time

# 1) Load in mesh data
# mesh_yaml = join("data", "BAR_URC_numEl_52.yaml")
mesh_yaml = join("data", "bar_urc_shell_mesh.yaml")
mesh_data = opensg.load_yaml(mesh_yaml)

# 2) Create BladeMesh Object
blade_mesh = opensg.BladeMesh(mesh_data)

# 3) Compute stiffness matrices for each blade segment
segment_start_index = 1
segment_end_index = 2

segment_stiffness_matrices = []
boundary_stiffness_matrices = []
compute_times = []
for i in range(segment_start_index, segment_end_index):
    segment_mesh = blade_mesh.generate_segment_mesh(segment_index=i, filename="section.msh")
    ABD = segment_mesh.compute_ABD()

    segment_stiffness, left_boundary_stiffness = segment_mesh.compute_timo_stiffness_segment(ABD)

    segment_stiffness_matrices.append(segment_stiffness)
    boundary_stiffness_matrices.append(left_boundary_stiffness)

# combine segment matrices and save
combined_stiffness_matrices = np.concat(segment_stiffness_matrices)

np.savetxt('stiffness_m.txt', combined_stiffness_matrices, fmt='%d')
