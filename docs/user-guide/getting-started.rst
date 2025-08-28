.. _getting_started:

Getting started
======================================

To start using OpenSG, open a Python console or IDE like Spyder and import the package::

	import opensg	

OpenSG comes with a simple `getting started example <https://github.com/wenbinyugroup/OpenSG/blob/main/examples/run_shell_analysis.py>`_, 
shown below, that uses `shell blade data <https://github.com/wenbinyugroup/OpenSG/tree/main/data/shell_blade>`_.

This script shows how to use the OpenSG package to:

1. Load mesh data from a YAML file
2. Create a BladeMesh object
3. Generate segment meshes
4. Compute ABD and stiffness matrices for each segment
5. Save the results

.. literalinclude:: ../../examples/run_shell_analysis.py

See :ref:`examples` for more information on downloading and running examples.