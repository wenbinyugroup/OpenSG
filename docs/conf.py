"""Sphinx configuration for the OpenSG docs.

Deliberately dependency-light: MyST markdown only, no autodoc, so the site
builds from a clean checkout without the JAX/basix runtime.
"""
project = "OpenSG"
author = "Akshat Bagla"
copyright = "2026, Akshat Bagla"

extensions = ["myst_parser", "sphinxcontrib.mermaid"]
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence",
                          "deflist", "attrs_inline"]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "OpenSG"
