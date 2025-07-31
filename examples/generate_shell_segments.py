#!/usr/bin/env python3
"""
Complete example demonstrating the segment mesh workflow.

This example shows:
1. How to generate shell mesh YAML files from a blade mesh file
2. How to select specific segment indices when generating
3. How to load segment mesh files into SegmentMesh objects
4. How to perform structural analysis on individual segments
"""

import opensg
import numpy as np
from pathlib import Path
from os.path import join

print("OpenSG Segment Mesh Workflow Example")
print("=" * 50)

# 1. Generate segment mesh YAML files from a blade mesh file
print("\n1. Generating segment mesh YAML files...")

blade_mesh_file = join("data", "bar_urc_shell_mesh.yaml")
print(f"Using blade mesh file: {blade_mesh_file}")

# Generate all segments (default behavior)
print("\nGenerating all segments...")
opensg.io.generate_segment_shell_mesh_files(
    blade_mesh_file, 
    segment_folder="segments_all/"
)

all_segment_files = list(Path("segments_all").glob("*.yaml"))

print(f"Generated {len(all_segment_files)} segment files in 'segments_all/' directory")

# 2. Select specific segment indices when generating
print("\n2. Generating specific segment indices...")

# Example: Generate only segments 6, 10, and 15
specific_indices = [6, 10, 15]
print(f"Generating segments with indices: {specific_indices}")

opensg.io.generate_segment_shell_mesh_files(
    blade_mesh_file, 
    segment_list=specific_indices,
    segment_folder="segments_specific/"
)

specific_segment_files = list(Path("segments_specific").glob("*.yaml"))

print(f"Generated {len(specific_segment_files)} specific segment files:")
for i, file_path in enumerate(specific_segment_files):
    print(f"  {i+1}. {Path(file_path).name}")

# 3. Load segment mesh files into SegmentMesh objects
print("\n3. Loading segment mesh files...")

# Load a specific segment
test_segment_file = specific_segment_files[0]
print(f"\nLoading segment from: {test_segment_file}")

# Use StandaloneSegmentMesh directly
from opensg.mesh.segment import StandaloneSegmentMesh
segment_mesh = StandaloneSegmentMesh(test_segment_file)

print(f"  ✓ Successfully loaded segment")
print(f"  ✓ Mesh has {segment_mesh.mesh.geometry.x.shape[0]} nodes")
print(f"  ✓ Mesh has {segment_mesh.mesh.topology.index_map(2).size_local} elements")

# Show segment information
print(f"\nSegment Information:")
print(f"  Mesh dimensions: {segment_mesh.mesh.geometry.x.shape}")
print(f"  Number of elements: {segment_mesh.mesh.topology.index_map(2).size_local}")
print(f"  Number of layups: {len(segment_mesh.layup_database['mat_names'])}")

# Show layup information
print(f"\nLayup Information:")
for i, (mat_names, thick, angle) in enumerate(zip(
    segment_mesh.layup_database['mat_names'],
    segment_mesh.layup_database['thick'],
    segment_mesh.layup_database['angle']
)):
    print(f"  Layup {i}: {len(mat_names)} layers")
    for j, (mat, t, a) in enumerate(zip(mat_names, thick, angle)):
        print(f"    Layer {j}: {mat}, thickness={t}m, angle={a}°")
