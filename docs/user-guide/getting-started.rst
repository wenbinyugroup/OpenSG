.. _getting-started:

Getting Started
===============

This guide will walk you through your first OpenSG analysis, from installation to running a complete blade segment analysis.

Prerequisites
-------------

Before using OpenSG, you should have:

- Basic knowledge of Python programming
- Understanding of finite element analysis concepts
- Familiarity with composite materials (optional but helpful)

Your First Analysis
-------------------

Let's perform a simple blade segment analysis using the provided example data.

1. **Set up your environment**::

    conda activate opensg_env
    cd /path/to/your/working/directory

2. **Prepare the analysis script**:

   Create a new Python file called ``my_first_analysis.py``:

   .. code-block:: python

      from pathlib import Path
      import opensg
      import numpy as np
      from opensg.mesh.segment import ShellSegmentMesh

      # 1) Load blade mesh data and generate segment files
      blade_mesh_file = Path("data", "bar_urc_shell_mesh.yaml")

      # Generate segment mesh files
      opensg.io.generate_segment_shell_mesh_files(
          blade_mesh_file, 
          segment_folder="segments/"
      )

      # 2) Analyze the first segment
      segment_file = Path("segments", f"{blade_mesh_file.stem}_segment_1.yaml")
      
      # Create segment mesh object
      segment_mesh = ShellSegmentMesh(segment_file)
      
      # Compute ABD matrices for the segment
      ABD = segment_mesh.compute_ABD()
      
      # Compute stiffness matrices using both beam theories
      timo_stiffness, eb_stiffness, l_stiffness, r_stiffness = segment_mesh.compute_stiffness(ABD)
      
      # Display results
      print("Timoshenko stiffness matrix shape:", timo_stiffness.shape)
      print("Euler-Bernoulli stiffness matrix shape:", eb_stiffness.shape)
      
      # Save results
      np.savetxt('segment_1_timoshenko.txt', timo_stiffness, fmt='%.6e')
      print("Results saved to segment_1_timoshenko.txt")

3. **Run the analysis**::

    python my_first_analysis.py

Understanding the Output
------------------------

The analysis produces several types of stiffness matrices:

- **Timoshenko stiffness matrix** (6×6): Includes shear deformation effects
- **Euler-Bernoulli stiffness matrix** (4×4): Classical beam theory without shear
- **Boundary stiffness matrices**: For left and right segment boundaries
