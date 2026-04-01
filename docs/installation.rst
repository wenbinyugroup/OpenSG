.. _installation:

Installation
============

OpenSG requires `DOLFINx (FEniCSx) <https://github.com/fenics/dolfinx>`_, which has non-Python
dependencies and is not available as a standalone pip package. The recommended installation path is via conda,
which handles all system-level dependencies automatically.

Conda (recommended)
--------------------

If you are unfamiliar with conda, refer to the
`conda documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_.

1. Clone the OpenSG repository::

    git clone https://github.com/wenbinyugroup/OpenSG
    cd OpenSG

2. Create and activate the conda environment::

    conda env create -f environment.yml
    conda activate opensg_env

The ``environment.yml`` file installs DOLFINx, PETSc, and all other required dependencies.

.. note::

   On some systems, ``gmsh`` must be installed via pip rather than conda to interface correctly with DOLFINx.
   If you encounter gmsh-related errors, run::

       conda remove gmsh
       pip install gmsh

pip (advanced)
--------------

A pip-based install is possible but requires DOLFINx and its non-Python dependencies to already be present on
your system. OpenSG itself can then be installed in editable mode::

    pip install -e .

For additional instructions on installing DOLFINx, see the
`DOLFINx installation documentation <https://github.com/fenics/dolfinx>`_.
