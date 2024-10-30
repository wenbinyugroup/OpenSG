.. _intallation:

Installation 
============

Download pyNuMAD
----------------

The OpenSG source code is hosted on the `opensg GitHub repository <https://github.com/sandialabs/pyNuMAD>`_. 
OpenSG users are recommended to clone the Github repository.
Cloning the repository allows users to easily pull the latest updates to the pyNuMAD source code.
These updates may improve the code's speed, accuracy and add additional functionality or advanced features.

To download OpenSG using `git <https://git-scm.com/>`_, type the following in a git interface:: 

    git clone https://github.com/sandialabs/pyNuMAD

Installation
------------

To install OpenSG and create an environment where the code runs correctly, please execute the following steps:

1. Download the OpenSG repository::

    git clone https://github.com/sandialabs/pyNuMAD

2. Create the conda environment named `opensg_env` using the `environment.yml` file::

    conda env create -f environment.yml

3. On some systems, there is a bug where `gmsh` must be installed through pip, rather than conda, to interface with `dolfinx` correctly.
If this bug occurs for you please run the following commands::

    conda remove gmsh
    pip install gmsh

4. Install the OpenSG package by running the following command in the root of the OpenSG repository::

    pip install .

Alternatively, to perform a development installation run::

    pip install -e .


.. Developers are recommended to install using the instructions on
.. :ref:`contributing<contributing>` page.
