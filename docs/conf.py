# Add the parent directory to Python path to find opensg package
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# from opensg import __version__
__version__ = "0.0.1"

# -- Project information -----------------------------------------------------
project = u'opensg'
copyright = u'2023 National Technology & Engineering Solutions of Sandia, LLC (NTESS)'
author = u'opensg Developers'

version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.autosummary',
    # 'sphinx.ext.viewcode', # commenting out for now b/c bad render width
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
]
napoleon_use_rtype = False
viewcode_import = True
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
autodoc_member_order = 'bysource'
autoclass_content = 'both'
bibtex_bibfiles = ['user-guide/references.bib']
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
language = "en"
numfig = True

autodoc_mock_imports = ["dolfinx", "basix", "ufl", "gmshio", "gmsh", "petsc4py", "scipy",
                        "mpi4py", "numpy", "slepc4py", "meshio", "pyvista"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# autosummary settings
import glob
autosummary_generate = ["api-doc.rst",] + glob.glob("apidoc/*.rst")
# autosummary_generate = True
autosummary_generate_overwrite = True


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '_user']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = 'pydata_sphinx_theme'

# Theme options for PyData Sphinx Theme
html_theme_options = {
    "show_toc_level": 2,
    "navbar_align": "content",
    "navbar_persistent": ["search-button"],
    "primary_sidebar_end": [],  # Remove ads for cleaner appearance
    "secondary_sidebar_items": ["page-toc"],
    "use_edit_page_button": False,
    "show_prev_next": True,
    "navigation_with_keys": True,
    "collapse_navigation": False,
    "navigation_depth": 4,
}

# Configure sidebar content
html_sidebars = {
    "**": ["sidebar-nav-bs"]  # Simplified sidebar for consistency
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
# html_style = 'css/my_style.css'  # Commented out until custom CSS is created

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'opensgdoc'