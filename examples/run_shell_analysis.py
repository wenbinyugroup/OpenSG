"""Example script demonstrating the computation of blade stiffness matrices.

This script shows how to use the OpenSG package to:
1. Load mesh data from a YAML file
2. Create a BladeMesh object
3. Generate segment meshes
4. Compute ABD and stiffness matrices for each segment
5. Save the results

The script processes a wind turbine blade mesh and computes its structural
properties using both Euler-Bernoulli and Timoshenko beam theories.
"""

from pathlib import Path
import opensg
import numpy as np
import time
from opensg.mesh.segment import ShellSegmentMesh

# Generate all segments from blade data
blade_mesh_file = Path("data", "bar_urc_shell_mesh.yaml")

print("\nGenerating all segments...")
opensg.io.generate_segment_shell_mesh_files(
    blade_mesh_file, 
    segment_folder="data/segments_all/"
)

np.set_printoptions(precision=4)
left_timo, right_timo, taper_timo = [],[],[]
compute_times = []
left_origin, right_origin, taper_origin = [],[],[]
left_mass, right_mass, taper_mass = [],[],[]
blade_length=100

# Process segments (adjust range based on desired segments)
for i in range(3):
    segment_file = Path("segments_all", f"{blade_mesh_file.stem}_segment_{i+1}.yaml")
    
    # Generate mesh for current segment
    segment_mesh = ShellSegmentMesh(segment_file)

    left_origin.append(segment_mesh.left_submesh["origin"]/blade_length)
    right_origin.append(segment_mesh.right_submesh["origin"]/blade_length)
   # taper_origin.append(segment_mesh.meshdata["origin"]/blade_length)
           
    # Compute ABD matrices for the segment
    ABD, mass = segment_mesh.compute_ABD()
    left_mass.append(utils.get_mass_shell(segment_mesh.left_submesh, mass))
    right_mass.append(utils.get_mass_shell(segment_mesh.right_submesh, mass))
  #  taper_mass.append(utils.get_mass_shell(segment_mesh.meshdata, mass, Taper=True))   
    
    print('\n Left mass \n',left_mass[int(segment)])
    print('\n Right mass \n',right_mass[int(segment)])
    
    # Compute stiffness matrices using both beam theories
    taper_stiffness, l_timo_stiffness, r_timo_stiffness = segment_mesh.compute_stiffness(ABD, Taper=False)

    # Store results
    print('\n Left timo \n',l_timo_stiffness)
    print('\n Right timo \n',r_timo_stiffness)
    
    left_timo.append(l_timo_stiffness)
    right_timo.append(r_timo_stiffness) 

    compute_times.append(time.time()-tic) 
    print('Time Taken',str(time.time()-tic))

print('\nleft_origin')
print(left_origin)

print('\nright_origin')
print(right_origin)

print('\ntaper_origin')
print(taper_origin)
#Append tip to left boundary values
left_timo.append(right_timo[-1])
left_origin.append(right_origin[-1])
left_mass.append(right_mass[-1])
print('\n\nleft_origin ###############')
print(left_origin)

#Prepend root to right boundary values
right_timo.insert(0,left_timo[0])
right_origin.insert(0,left_origin[0])
right_mass.insert(0,left_mass[0])
print('\n\nright_origin ###############')
print(right_origin)

left_timo=np.array(left_timo)
right_timo=np.array(right_timo)
#taper_timo=np.array(taper_timo)

left_mass=np.array(left_mass)
right_mass=np.array(right_mass)
#taper_mass=np.array(taper_mass)

print('Total Time for Wind Blade (in sec)', np.sum(compute_times))   

opensg.utils.shared.write_beamdyn_files(right_timo, right_mass, right_origin,'bar_shell_right_52_20251004')
opensg.utils.shared.write_beamdyn_files(left_timo, left_mass, left_origin,'bar_shell_left_52_20251004')
