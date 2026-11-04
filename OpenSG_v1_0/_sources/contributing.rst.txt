.. _contributing:


Contributing
============

Thank you for considering contributing to opensg. We welcome
contributions from the community in the form of bug fixes, feature
enhancements, documentation updates, etc. All contributions are
processed through pull-requests or issues on GitHub. Please follow these
guidelines for contributing.


Reporting issues and bugs
-------------------------

This section guides you through the process of submitting an issue for
opensg. To report issues or bugs please `create a new issue
<https://github.com/wenbinyugroup/opensg/issues/new>`_ on GitHub.

Following these guidelines will help maintainers understand your issue,
reproduce the behavior, and develop a fix in an expedient fashion.
Before submitting your bug report, please perform a cursory search to
see if the problem has been already reported. If it has been reported,
and the issue is still open, add a comment to the existing issue instead
of opening a new issue.

Tips for effective bug reporting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Use a clear descriptive title for the issue
-  Describe the steps to reproduce the problem, the behavior you
   observed after following the steps, and the expected behavior
-  Provide the SHA ID of the git commit that you are using
-  For runtime errors, provide a function call stack

Submitting pull-requests
^^^^^^^^^^^^^^^^^^^^^^^^

Contributions can take the form of bug fixes, feature enhancements,
documentation updates. All updates to the repository are managed via
`pull requests
<https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests>`_.
One of the easiest ways to get started is by looking at `open issues
<https://github.com/wenbinyugroup/opensg/issues>`_ and contributing fixes,
enhancements that address those issues. If your code contribution
involves large changes or additions to the codebase, we recommend
opening an issue first and discussing your proposed changes with the
core development team to ensure that your efforts are well directed, and
so that your submission can be reviewed and merged seamlessly by the
maintenance team.

Guidelines for preparing and submitting pull-requests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Use a clear descriptive title for your pull-requests

-  Describe if your submission is a bugfix, documentation update, or a
   feature enhancement. Provide a concise description of your proposed
   changes.

-  Provide references to open issues, if applicable, to provide the
   necessary context to understand your pull request

-  Make sure that your pull-request merges cleanly with the `main`
   branch of opensg. When working on a feature, always create your
   feature branch off of the latest `main` commit

-  Ensure that the code compiles without warnings, (leave for later? the
   unit tests and regression tests all pass without errors, and the
   documentation builds properly with your modifications)

-  New physics models and code enhancements should be accompanied with
   relevant updates to the documentation, supported by necessary
   verification and validation, as well as unit tests and regression
   tests

Once a pull-request is submitted you will iterate with opensg
maintainers until your changes are in an acceptable state and can be
merged in. You can push addditional commits to the branch used to create
the pull-request to reflect the feedback from maintainers and users of
the code.

Stylistic conventions
^^^^^^^^^^^^^^^^^^^^^

-  Please apply `black <https://black.readthedocs.io/en/stable/>`__ formatting to your code.
   This enforces a standard style across the project with minimal
   thought and effort

-  Please use `snake case <https://en.wikipedia.org/wiki/Snake_case>`__
   for function and variable names.

Developer Installation
----------------------

The standard installation method will install the package in editable mode, so no additional steps are required.

Running the Test Suite
-----------------------

To run the test suite locally:

1. **Install pytest** (if not already installed):

   .. code-block:: bash

      pip install pytest

2. **Run the test suite**:

   .. code-block:: bash

      pytest opensg/tests/

This will run all tests in the ``opensg/tests/`` directory, matching the same process used in the continuous integration workflow.

Documentation
-------------

Docstrings
^^^^^^^^^^

The documentation for opensg adheres to NumPy style docstrings. Not
only does this help to keep a consistent style, but it is also necessary
for the API documentation to be parsed and displayed correctly. For an
example of what this should look like:

.. code::

   def func(arg1, arg2):
   """Summary line.

   Extended description of function.

   Parameters
   ----------
   arg1 : int
       Description of arg1
   arg2 : str
       Description of arg2

   Returns
   -------
   bool
       Description of return value

   """
   return True

Additional examples can be found in the `napoleon documentation
<https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.
The following boilerplate can be copy-pasted into the top of a function
definition to help get things started:

.. code::

   """Summary line.

   Extended description of function.

   Parameters
   ----------

   Returns
   -------

   """

Extending opensg
-----------------

Below we explain what to do when adding a top level directory 
in the opensg source called ``new_mod/`` which contains a submodule called
``new_file.py``.

Exposing New Functionality to Users
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default, any functions in ``new_file.py`` would not be automatically
available to users operating outside of the source directory. This is 
to enforce a distinction between internal and external functionality.
Any functionality you wish to be accessed externally needs to be imported
inside the appropriate ``__init__.py`` file. For example, the ``__init__.py``
file inside of ``new_mod/`` might look like::

   from new_file import new_function

and the ``__init__.py`` inside of ``src/`` might look like::

   import new_mod

This would allow a user to access the functionality by importing ``opensg``
and then running ``opensg.new_mod.new_function``. If you wanted the call
to be accessed with ``opensg.new_mod.new_file.new_function``, you would
replace ``from new_file import new_function`` with ``import new_file``. If
you are extending an already existing module, please follow the
existing convention within the corresponding ``__init__.py``.

Expanding the API Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

New functionality for opensg should be properly documented
in the API documentation. A new folder-level module named ``new_mod/`` would 
require the creation of the file ``docs/apidoc/opensg.new_mod`` with
the contents recording any submodules that should be captured by the
API documentation. A new file named ``new_file.py`` added to an existing folder would
need the following code to capture its functionality in the auto api documentation::
   
   .. automodule:: opensg.new_mod.new_file
      :members:
      :no-undoc-members:
      :show-inheritance:

This will automatically capture all functions (internal and external) in ``new_file.py``.

Building the Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the documentation locally for testing and development:

1. **Install documentation dependencies**:

   .. code-block:: bash

      pip install -r docs/requirements.txt

2. **Navigate to the docs directory**:

   .. code-block:: bash

      cd docs/

3. **Build the HTML documentation**:

   .. code-block:: bash

      make html

   This will generate the documentation in the ``_build/html/`` directory.

4. **View the documentation**:

   Open ``_build/html/index.html`` in your web browser to view the generated documentation.

5. **Clean previous builds** (if needed):

   .. code-block:: bash

      make clean

**Adding Tutorials**

To add new Jupyter notebook tutorials:

1. Place your ``.ipynb`` file in the ``examples/`` directory
2. Add a notebook link to ``docs/tutorials/`` directory following the format of the other notebooks.
3. Add the notebook reference to ``docs/tutorials.rst`` in the appropriate ``.. toctree::`` section
4. Rebuild the documentation to verify the tutorial appears correctly
