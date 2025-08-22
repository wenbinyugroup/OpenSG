"""Example script demonstrating the computation of solid blade stiffness matrices
using both Euler-Bernoulli and Timoshenko beam theories.

This script shows how to use the OpenSG package to:
1. Load solid mesh data from mesh YAML files
2. Preprocess segment meshes for solid analysis
3. Compute stiffness matrices for each segment
4. Save the results

"""

from pathlib import Path
import opensg
import numpy as np
import time
from opensg.mesh.segment import SolidSegmentMesh
from opensg.core.solid import compute_stiffness

segments_folder = Path("data", "bar_urc_npl_2_ar_10")

segment_stiffness_matrices = []
boundary_stiffness_matrices = []
compute_times = []

# Process segments (adjust range based on desired segments)
for i in range(3):
    segment_file = Path(segments_folder, f"bar_urc_npl_2_ar_10-segment_{i}.yaml")
    
    print(f"Processing segment {i}...")
    
    # Initialize segment mesh with segment file
    segment_mesh = SolidSegmentMesh(segment_file)
    
    # The solid segment mesh provides the necessary data for stiffness computation
    print(f"  Segment {i}: {segment_mesh.num_cells} elements, {len(segment_mesh.material_database[0])} materials")
    
    print(f"  Computing solid stiffness matrices for segment {i}...")
    
    # Extract material parameters and mesh data
    material_parameters, density = segment_mesh.material_database
    meshdata = segment_mesh.meshdata
    l_submesh = segment_mesh.left_submesh
    r_submesh = segment_mesh.right_submesh
    
    # Compute stiffness matrices using the solid analysis function
    timo_segment_stiffness, V0, V1s = compute_stiffness(
        material_parameters,
        meshdata,
        l_submesh,
        r_submesh
    )
    
    print(f"  Timoshenko stiffness matrix shape: {timo_segment_stiffness.shape}")
    segment_stiffness_matrices.append(timo_segment_stiffness)
    boundary_stiffness_matrices.append(V0)


# Combine segment matrices and save
combined_stiffness_matrices = np.concatenate(segment_stiffness_matrices, axis=0)
np.savetxt('stiffness_solid.txt', combined_stiffness_matrices, fmt='%.6e')
print(f"Saved {len(segment_stiffness_matrices)} segment stiffness matrices to stiffness_solid.txt")
print(f"Matrix shape: {combined_stiffness_matrices.shape}")
