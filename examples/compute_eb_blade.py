from os.path import join

# import pynumad
import opensg
import numpy as np
import time

# load blade.yml into pynumad
# blade_path = join("data", "blade.yaml")
# blade = pynumad.Blade(blade_path)

# create mesh

# NOTE: triangular elements are problematic so will need a workaround. 
#   Evan may have a solution.

# create mesh object for use with opensg
#   3 options:
#   - create our own mesh class
#   - use pynumad class
#   - use dolfinx or gmsh structures? not sure if these are flexible enough though

# blade_mesh_info = mesh()
mesh_yaml = join("data", "bar_urc_shell_mesh.yaml")
mesh_data = opensg.load_yaml(mesh_yaml)

blade_mesh = opensg.BladeMesh(mesh_data)
stiffness_matrices = []
compute_times = []
for i in range(1, 2):
    start_time = time.time()
    print(time.time()-start_time)
 
    section_mesh = blade_mesh.generate_segment_mesh(segment_index=i, filename="section.msh")
    print(time.time()-start_time)

    ABD = section_mesh.compute_ABD()
    print(time.time()-start_time)

    stiffness_matrix = section_mesh.compute_stiffness_EB(ABD)
    print(time.time()-start_time)
    
    stiffness_matrices.append(stiffness_matrix)
    end_time = time.time()
    compute_times.append(end_time - start_time)
    
print("Average compute time for a single segment: ", sum(compute_times)/len(compute_times))
print("Maximum compute time for a single segment: ", max(compute_times))

# combine matrices into a global 
combined_stiffness_matrices = np.concat(stiffness_matrices)

np.savetxt('stiffness_m.txt', combined_stiffness_matrices, fmt='%d')

# pause


# ## Extract the mesh for the section
# nodes = mesh_data['nodes']
# numNds = len(nodes)
# elements = mesh_data['elements']
# numEls = len(elements)

# ndNewLabs = -1*np.ones(numNds,dtype=int)
# elNewLabs = -1*np.ones(numEls,dtype=int)
# elLayID = -1*np.ones(numEls,dtype=int)

# # iterate through blade

# segment_matrices = []

# for i in range(len(blade.ispan)):
#     # select ith blade segment
#     blade_segment_mesh = opensg.blade.select_segment(blade_mesh_info)
#     # analysis options
#     #   solid vs shell
#     #   whole segment vs boundary
#     # analyses:
#     # stresses/stiffness/buckling
#     # stresses after beamdyn
#     data = opensg.compute_eb_segment(blade_segment_mesh)
#     # data = opensg.compute_eb_boundaries(blade_segment_mesh)
#     # data = opensg.compute_timo_segment(blade_segment_mesh) # ***** high priority
#     # data = opensg.compute_eb_buckling(blade_segment_mesh)
#     # data = opensg.compute_timo_buckling(blade_segment_mesh) # **** top priority
#     # data = opensg.compute_timo_boundaries(blade_segment_mesh)
    
#     segment_matrices.append(data)
    

# # ideally, we could also have a step to run beamdyn
# opensg.beamdyn.run_analysis(blade_mesh_info, segment_matrices)