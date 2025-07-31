"""Example script demonstrating the computation of solid blade stiffness matrices.

This script shows how to use the OpenSG package to:
1. Load solid mesh data from a YAML file
2. Generate segment meshes for solid analysis
3. Compute ABD and stiffness matrices for each segment
4. Save the results

The script processes a wind turbine blade solid mesh and computes its structural
properties using both Euler-Bernoulli and Timoshenko beam theories.
"""

from pathlib import Path
import opensg
import numpy as np
import time
from opensg.mesh.solidsegment import StandaloneSolidSegmentMesh
from opensg.core.solid import compute_stiffness

# 1) Load in solid mesh data
segments_folder = Path("data", "Solid OpenSG Beam", "bar_urc_npl_2_ar_10")

# Note: For solid analysis, we work with individual segment YAML files
# The segment YAML files should already exist in the segments_solid/ directory
print("\nUsing existing solid segment YAML files...")

# 3) Compute stiffness matrices for each blade segment
segment_stiffness_matrices = []
boundary_stiffness_matrices = []
compute_times = []

# Process segments (adjust range based on available segments)
for i in range(3):  # Adjust this range based on your solid mesh segments
    segment_file = Path(segments_folder, f"bar_urc_npl_2_ar_10-segment_{i}.yaml")
    
    # Check if segment file exists
    if not segment_file.exists():
        print(f"Segment {i} file not found, skipping...")
        continue
    
    print(f"Processing segment {i}...")
    
    # Generate mesh for current segment using StandaloneSolidSegmentMesh
    segment_mesh = StandaloneSolidSegmentMesh(str(segment_file))
    
    # For solid analysis, we need to compute stiffness matrices directly
    # The solid segment mesh provides the necessary data for stiffness computation
    print(f"  Segment {i}: {segment_mesh.num_cells} elements, {len(segment_mesh.material_database[0])} materials")
    
    # Compute solid stiffness matrices using the core solid analysis functions
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
    
    # For solid analysis, we typically get the Timoshenko stiffness matrix directly
    # The function returns the 6x6 Timoshenko stiffness matrix
    print(f"  Timoshenko stiffness matrix shape: {timo_segment_stiffness.shape}")
    segment_stiffness_matrices.append(timo_segment_stiffness)
    boundary_stiffness_matrices.append(V0)


# Combine segment matrices and save
if segment_stiffness_matrices:
    combined_stiffness_matrices = np.concatenate(segment_stiffness_matrices, axis=0)
    np.savetxt('stiffness_solid.txt', combined_stiffness_matrices, fmt='%.6e')
    print(f"Saved {len(segment_stiffness_matrices)} segment stiffness matrices to stiffness_solid.txt")
    print(f"Matrix shape: {combined_stiffness_matrices.shape}")
else:
    print("No segment stiffness matrices computed.")

print("\nSolid analysis completed successfully!")
print("The script computed Timoshenko stiffness matrices for each segment")
print("using the core solid analysis functions from opensg.core.solid.")
