.. _examples:

Examples
========

This page provides examples demonstrating the capabilities of OpenSG for wind turbine blade structural analysis.
All examples are available in the ``examples/`` directory of the OpenSG repository.

Current examples include:

* ``run_shell_analysis.py`` : This example demonstrates how to use the OpenSG package to compute the stiffness matrix for shell blade segments.

* ``run_solid_analysis.py`` : This example demonstrates how to use the OpenSG package to compute the stiffness matrix for solid blade segments.

* ``generate_shell_segments.py`` : This example demonstrates how to use the OpenSG package 
  to preprocess the shell blade mesh data to individual segment meshes.


Running the Examples
--------------------

To run the examples, first ensure OpenSG is properly installed:

.. code-block:: bash

   cd OpenSG/
   conda activate opensg_env

Then navigate to the examples directory and use python to run the example:

.. code-block:: bash

   cd examples/
   python run_shell_analysis.py

Example data
-------------

Data for the examples are available in the ``data/`` directory of the OpenSG repository.

* shell_blade: This directory contains a shell blade mesh and an example shell segment mesh.

* solid_blade: This directory contains examples of solid segment meshes. A full solidblade mesh is not provided, as the file would be too large

