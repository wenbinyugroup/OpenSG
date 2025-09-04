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
blade_mesh_file = Path("..", "data", "shell_blade", "bar_urc_shell_mesh.yaml")
outputs_folder = Path("outputs")
segment_folder = outputs_folder / "segments_all"

print("\nGenerating all segments...")
opensg.io.generate_segment_shell_mesh_files(
    blade_mesh_file, 
    segment_folder=segment_folder
)

segment_stiffness_matrices = []
boundary_stiffness_matrices = []
compute_times = []

# Process segments (adjust range based on desired segments)
for i in range(3):
    segment_file = Path(segment_folder, f"{blade_mesh_file.stem}_segment_{i+1}.yaml")
    
    # Initialize segment mesh with segment file
    segment_mesh = ShellSegmentMesh(segment_file)
    
    # Compute ABD matrices for the segment
    ABD = segment_mesh.compute_ABD()

    # Compute stiffness matrices using both beam theories
    timo_segment_stiffness, eb_segment_stiffness, l_timo_stiffness, r_timo_stiffness = segment_mesh.compute_stiffness(ABD)

    # Store results
    segment_stiffness_matrices.append(timo_segment_stiffness)
    boundary_stiffness_matrices.append(l_timo_stiffness)

# Combine segment matrices and save
combined_stiffness_matrices = np.concat(segment_stiffness_matrices)
np.savetxt(outputs_folder / 'stiffness_shell.txt', combined_stiffness_matrices, fmt='%d')
print(f"Saved {len(segment_stiffness_matrices)} segment stiffness matrices to stiffness_shell.txt")
print(f"Matrix shape: {combined_stiffness_matrices.shape}")
