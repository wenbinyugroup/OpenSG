.. raw:: latex

    \clearpage

.. _examples:

Examples
========

This page provides examples demonstrating the capabilities of OpenSG for wind turbine blade structural analysis.
All examples are available in the ``examples/`` directory of the OpenSG repository.

Getting Started Example
-----------------------

The simplest way to get started with OpenSG is to run the basic blade segment analysis example:

.. literalinclude:: ../examples/run_shell_analysis.py
   :language: python
   :caption: Basic shell segment analysis

This example demonstrates:

* Loading a blade mesh from YAML file
* Generating segment files
* Computing ABD matrices for composite layups
* Computing stiffness matrices using beam theories
* Saving results for further analysis

Running the Examples
--------------------

To run the examples, first ensure OpenSG is properly installed:

.. code-block:: bash

   cd OpenSG/
   conda activate opensg_env
   cd examples/
   python run_shell_analysis.py
