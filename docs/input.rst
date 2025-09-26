.. input:

Mesh Input Format Guide
=======================

This guide describes the yaml input file format supported by OpenSG and how to load mesh data into the 
various mesh classes for wind turbine blade analysis.

Overview
--------

OpenSG currently supports a yaml file format:
   - Primary format for mesh definitions
   - Human-readable and easily editable
   - Contains complete mesh geometry, materials, and layup information

YAML File Structure
-------------------

The YAML format contains several key sections that define the complete mesh:

Basic Structure
~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Mesh geometry
   nodes: 
     - [x1, y1, z1]  # Node coordinates
     - [x2, y2, z2]
     # ... more nodes
   
   elements:
     - [node_1, node_2, node_3, node_4]  # Element connectivity (quad/tri)
     - [node_5, node_6, node_7, node_8]
     # ... more elements
   
   # Material definitions
   materials:
     - name: "carbon_fiber"
       E1: 150000.0      # Young's modulus in fiber direction (MPa)
       E2: 10000.0       # Young's modulus transverse to fiber (MPa)
       G12: 5000.0       # Shear modulus (MPa)
       nu12: 0.3         # Poisson's ratio
     # ... more materials
   
   # Element grouping and layup assignment
   sets:
     element:
       - name: "segment_0_layup_0"
         elements: [element_1, element_1, ..., element_n]  # Element IDs for this layup
       # ... more element sets
   
   # Layup definitions
   sections:
     - elementSet: "segment_0_layup_0"
       layup:
         - ["carbon_fiber", 0.001, 0.0]     # [material, thickness, angle]
         - ["carbon_fiber", 0.001, 1.57]    # 90 degrees in radians
         # ... more layers
   
   # Element orientation matrices
   elementOrientations:
     - [[1.0, 0.0, 0.0],   # 3x3 rotation matrix for element 1
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]
     # ... one matrix per element

Detailed Section Descriptions
-----------------------------

Understanding each section of the YAML file is crucial for creating valid mesh inputs:

Nodes Section
~~~~~~~~~~~~~

The ``nodes`` section defines the 3D coordinates of all mesh vertices:

.. code-block:: yaml

   nodes:
     - [0.0, 0.0, 0.0]      # Node 1: [x, y, z] coordinates
     - [1.0, 0.0, 0.0]      # Node 2
     - [1.0, 1.0, 0.0]      # Node 3
     - [0.0, 1.0, 0.0]      # Node 4

Elements Section
~~~~~~~~~~~~~~~~

The ``elements`` section defines how nodes are connected to form finite elements:

.. code-block:: yaml

   elements:
     - [1, 2, 3, 4]    # Quadrilateral element using nodes 1,2,3,4
     - [1, 2, 3]       # Triangular element using nodes 1,2,3

**Requirements:**
- Node IDs reference the nodes section (1-indexed)
- Supports both triangular (3 nodes) and quadrilateral (4 nodes) elements
- Node ordering should follow standard finite element conventions

Materials Section
~~~~~~~~~~~~~~~~~

The ``materials`` section defines material properties for composite analysis:

.. code-block:: yaml

   materials:
     - name: "carbon_fiber_unidirectional"
       E1: 150000.0    # Longitudinal Young's modulus (MPa)
       E2: 10000.0     # Transverse Young's modulus (MPa)
       G12: 5000.0     # In-plane shear modulus (MPa)
       nu12: 0.3       # Major Poisson's ratio
       
     - name: "glass_fiber_fabric"
       E1: 45000.0
       E2: 45000.0     # Equal for fabric (quasi-isotropic)
       G12: 4500.0
       nu12: 0.25

**Requirements:**
- All moduli in MPa
- E1: fiber direction modulus, E2: transverse modulus
- G12: in-plane shear modulus
- nu12: Poisson's ratio (fiber to transverse direction)

Sets Section
~~~~~~~~~~~~

The ``sets`` section groups elements by their layup configurations:

.. code-block:: yaml

   sets:
     element:
       - name: "segment_0_layup_0"
         elements: [1, 2, 5, 6]      # Shell elements
       - name: "segment_0_layup_1" 
         elements: [3, 4, 7, 8]      # Different layup
     node:
       - name: "root_nodes"
         nodes: [1, 2, 3, 4]         # Boundary nodes

**Requirements:**
- Element IDs must reference valid elements
- Names should be descriptive and unique

Loading Mesh Data into OpenSG
-----------------------------

OpenSG provides several classes for loading and working with mesh data:

Loading Individual Segment Meshes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For segment-based analysis, use the ``ShellSegmentMesh`` or ``SolidSegmentMesh`` classes:

.. code-block:: python

   from opensg.mesh.segment import ShellSegmentMesh, SolidSegmentMesh
   
   # Load segment mesh directly from YAML file
   shell_segment = ShellSegmentMesh("your_shell_segment_mesh.yaml")
   solid_segment = SolidSegmentMesh("your_solid_segment_mesh.yaml")

Generating Segment Files from Blade Mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OpenSG can automatically generate individual segment files from a full blade mesh. 
This is currently only available for shell mesh:

.. code-block:: python

   from pathlib import Path
   import opensg
   
   # Generate all segment files from blade mesh
   blade_mesh_file = Path("your_shell_blade_mesh.yaml")
   
   segment_files = opensg.io.generate_segment_shell_mesh_files(
       blade_mesh_file, 
       segment_folder="segments/",
       segment_indices=None  # Generate all segments (default)
   )
   
   # Generate specific segments only
   specific_segments = opensg_io.generate_segment_shell_mesh_files(
       blade_mesh_file,
       segment_folder="segments/", 
       segment_indices=[0, 1, 2]  # Only segments 0, 1, and 2
   )
