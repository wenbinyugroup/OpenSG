.. _intallation:

Installation 
============

Download opensg
----------------

The OpenSG source code is hosted on the `opensg GitHub repository <https://github.com/wenbinyugroup/opensg>`_. 
OpenSG users are recommended to clone the Github repository.
Cloning the repository allows users to easily pull the latest updates to the opensg source code.
These updates may improve the code's speed, accuracy and add additional functionality or advanced features.

To download OpenSG using `git <https://git-scm.com/>`_, type the following in a git interface:: 

    git clone https://github.com/wenbinyugroup/opensg


Fetching Large Files (Git LFS)
------------------------------

Some files in the OpenSG repository are stored using 
`Git LFS (Large File Storage) <https://git-lfs.com/>`_. 
If you want these large files (such as datasets or models), 
you need to install Git LFS and fetch them explicitly.

1. Install Git LFS (only once per system)::

    git lfs install

2. After cloning the repository, fetch the large files::

    cd opensg
    git lfs pull

This will download all LFS-tracked files so they are available locally.

Installation
------------

To install OpenSG and create an environment where the code runs correctly, please execute the following steps:

1. Download the OpenSG repository::

    git clone https://github.com/wenbinyugroup/opensg

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
