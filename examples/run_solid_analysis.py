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
from opensg.mesh.segment import StandaloneSegmentMesh

# 1) Load in solid mesh data
blade_mesh_file = Path("data", "Solid OpenSG Beam", "bar_urc_npl_2_ar_10", "bar_urc_npl_2_ar_10.yaml")

# Generate all segments (default behavior)
print("\nGenerating all segments...")
opensg.io.generate_segment_shell_mesh_files(
    blade_mesh_file, 
    segment_folder="segments_solid/"
)

# 3) Compute stiffness matrices for each blade segment
segment_stiffness_matrices = []
boundary_stiffness_matrices = []
compute_times = []

# Process segments (adjust range based on available segments)
for i in range(30):  # Adjust this range based on your solid mesh segments
    segment_file = Path("segments_solid", f"{blade_mesh_file.stem}_segment_{i}.yaml")
    
    # Check if segment file exists
    if not segment_file.exists():
        print(f"Segment {i} file not found, skipping...")
        continue
    
    print(f"Processing segment {i}...")
    
    # Generate mesh for current segment
    segment_mesh = StandaloneSegmentMesh(segment_file)
    
    # Compute ABD matrices for the segment
    ABD = segment_mesh.compute_ABD()

    # Compute stiffness matrices using both beam theories
    timo_segment_stiffness, eb_segment_stiffness, l_timo_stiffness, r_timo_stiffness = segment_mesh.compute_stiffness(ABD)

    # Store results
    segment_stiffness_matrices.append(timo_segment_stiffness)
    boundary_stiffness_matrices.append(l_timo_stiffness)

# Combine segment matrices and save
if segment_stiffness_matrices:
    combined_stiffness_matrices = np.concatenate(segment_stiffness_matrices, axis=0)
    np.savetxt('stiffness_solid.txt', combined_stiffness_matrices, fmt='%.6e')
    print(f"Saved {len(segment_stiffness_matrices)} segment stiffness matrices to stiffness_solid.txt")
else:
    print("No segment stiffness matrices computed.")
