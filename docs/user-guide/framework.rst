.. role:: red

.. raw:: latex

    \clearpage

.. _software_framework:

Software framework and limitations
==================================

Before using OpenSG, it is helpful to understand the software framework.
OpenSG is a Python package, which contains several subpackages, listed below:
Each subpackage contains modules that contain classes, methods, and functions. 
The classes used to generate blade mesh models and 
run structural analysis are described in more detail below, followed by a list of software limitations.

.. only:: html

   See :ref:`api_documentation` for more information on the code structure.

.. only:: latex

   See the online API documentation at https://wenbinyugroup.github.io/OpenSG/ for more information on the code structure.
   
**OpenSG Subpackages:**

:mod:`~opensg.core`
   Contains classes and methods for core computational functions including ABD matrix computation, beam model implementations, and stress recovery.

:mod:`~opensg.mesh`
   Contains classes and methods to define blade mesh models, segment meshes, and mesh data structures for both shell and solid element analyses.

:mod:`~opensg.io`
   Contains classes and methods for input/output operations including YAML file handling, mesh file generation, and data serialization.

:mod:`~opensg.utils`
   Contains utility functions for mathematical operations, finite element computations, eigenvalue solving, and mesh utilities for both shell and solid analyses.

Blade mesh model
-----------------
The :class:`~opensg.mesh` subpackage contains classes to define blade mesh models, segment meshes, and their data structures.
These classes are listed below:
Blade mesh models can be built from YAML files or generated programmatically.
The package supports both shell and solid element analyses for wind turbine blade structural analysis.

**Mesh Classes:**

:class:`~opensg.mesh.segment.ShellSegmentMesh`
   Class to define individual shell-based blade segments for efficient segment-by-segment analysis without requiring the full blade mesh.

:class:`~opensg.mesh.segment.SolidSegmentMesh`
   Class to define individual solid-based blade segments for 3D structural analysis of blade segments.

Core computational modules
---------------------------
The :class:`~opensg.core` subpackage contains classes and functions for the core structural analysis computations.
These are listed below:

**Core Analysis Classes and Functions:**

:func:`~opensg.core.shell.compute_ABD_matrix`
   Function to compute ABD matrices for composite laminates. Relates forces/moments to strains/curvatures in shell structures.

:func:`~opensg.core.shell.compute_timo_boun`
   Function to compute boundary stiffness matrices for Euler-Bernoulli and Timoshenko beam theories on shell boundaries.

:func:`~opensg.core.shell.compute_stiffness`
   Function to compute segment stiffness matrices using shell element formulations.

:func:`~opensg.core.solid.compute_timo_boun`
   Function to compute boundary stiffness matrices for solid element analyses.

:func:`~opensg.core.solid.compute_stiffness`
   Function to compute segment stiffness matrices using solid element formulations.

:mod:`~opensg.core.stress_recov`
   Module containing functions for stress recovery and post-processing, including local strain computation and stress analysis.

Utility functions
------------------
The :class:`~opensg.utils` subpackage contains utility functions that support the core analysis capabilities.
These are organized by analysis type and listed below:

**Utility Modules:**

:mod:`~opensg.utils.shared`
   Contains shared utility functions including nullspace computation, linear system solvers, and constraint handling used across both shell and solid analyses.

:mod:`~opensg.utils.shell`
   Contains shell-specific utilities including local frame computations, strain measures, boundary condition handling, and shell element operations.

:mod:`~opensg.utils.solid`
   Contains solid-specific utilities including 3D strain measures, stress computations, boundary markers, and solid element operations.

:mod:`~opensg.utils.eigenvalue_solver`
   Contains eigenvalue solver utilities for advanced analysis including buckling and modal analysis capabilities.

.. _limitations:
   
Limitations
-----------
Current OpenSG limitations include:

TODO

.. _future_work:

Future work
-----------

TODO



