.. _examples:

Examples
========

This page provides examples demonstrating the capabilities of OpenSG for wind turbine blade structural analysis.
All examples are available in the ``examples/`` directory of the OpenSG repository.

Current examples include:

* ``1_get_beam_props_from_solid_cross_section.py`` : Obtain Timoshenko beam stiffness and mass matrix for solid cross-section.

* ``2_get_beam_props_from_shell_cross_section.py`` : Obtain Timoshenko beam stiffness and mass matrix for shell cross-section.

* ``3_get_beam_props_from_3D_solid_segment.py`` : Obtain Timoshenko beam stiffness and mass matrix for solid tapered 3D solid segment from a realistic wind blade.

* ``4_get_3D_stress_from_solid_cross_section.py`` : Perform dehomogenization of solid cross-section using structural responses and return a .vtk file where local stress, strain and displacement can be obtained over the SG domain.

* ``5_get_3D_stress_from_3D_solid_segment.py`` : Perform dehomogenization of solid 3D tapered segment using structural responses and return a .vtk file where local stress, strain and displacement can be obtained over the SG domain. The example obtains the critical buckling values based on the obtained local stress field.

* ``6_write_beamdyn_input_from_solid_cross_sections.py`` : Write beamdyn input file by computing the Timoshenko beam stiffness and mass matrix of all stations separately using solid cross-sections (given), and write in beamdyn input file format. The output would be a .inp file.

* ``7_write_beamdyn_input_from_shell_cross_sections.py`` : Write beamdyn input file by computing the Timoshenko beam stiffness and mass matrix of all stations separately using shell cross-sections (given), and write in beamdyn input file format. The output would be a .inp file.

* ``8_write_beamdyn_input_from_3D_solid_segments.py`` : Write beamdyn input file by computing the Timoshenko beam stiffness and mass matrix of all stations separately using solid 3D tapered segment (given), and write in beamdyn input file format. The origin is taken as midpoint of the 3D SG over the beam reference line. The output would be a .inp file.

The next two examples shows the helper function capability for input data from a pyNumAD realistic full blade mesh. These two examples take 3D blade segemnts mesh, and compute
the shell cross-section mesh (1D yaml) for shell wind blade segment and similarly, solid cross-section mesh (2D yaml) for a solid wind blade segment. It precisely uses the
boundary mapping to obtain the cross-sections at boundary of 3D SG. These examples are:

* ``convert_3D_shell_segment_to_boundary_mesh.py`` 

* ``convert_3D_solid_segment_to_boundary_mesh.py`` 


Running the Examples
--------------------

To run the examples, first ensure OpenSG is properly installed:

.. code-block:: bash

   cd OpenSG/
   conda activate opensg_env

Then navigate to the examples directory and use python to run the example:

.. code-block:: bash

   cd examples/
   python 1_get_beam_props_from_solid_cross_section.py

Example data
-------------

Data for the examples are available in the ``data/`` directory of the OpenSG repository.

* Shell_1DSG and Shell_3D_Taper: This directory contains a shell blade mesh cross-section and 3D taper segment mesh for all stations of realistic wind blade.

* Solid_1DSG and Solid_3DSG: This directory contains a solid blade mesh cross-section and 3D taper segment mesh for all stations of realistic wind blade.
