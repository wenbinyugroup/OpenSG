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

To install OpenSG and create an environment where the code runs correctly, is is recommended to 
use the `environment.yml` file to create the environment via conda. If you are unfamiliar with conda, 
please refer to the `conda documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_.

1. Download the OpenSG repository::

    git clone https://github.com/wenbinyugroup/opensg

2. Create the conda environment named `opensg_env` using the `environment.yml` file::

    conda env create -f environment.yml

3. Activate the environment::

    conda activate opensg_env

.. note::

   On some systems, there is a bug where `gmsh` must be installed through pip, rather than conda, to interface with `dolfinx` correctly.
   If this bug occurs for you please run the following commands::

       conda remove gmsh
       pip install gmsh


Troubleshooting
---------------

If you encounter issues during installation:

1. **Environment Issues**: Make sure you're using the correct conda environment
2. **Dependencies**: Check that all required packages are installed correctly
3. **GMSH Issues**: Try the pip installation method for gmsh if conda installation fails
4. **DOLFINx Compatibility**: Ensure you have a compatible version of DOLFINx installed

For additional help, please check the :ref:`contributing<contributing>` page or create an issue on the GitHub repository. 