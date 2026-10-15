"""Sphinx configuration for the OpenSG documentation.

MyST Markdown sources, the pydata-sphinx-theme, no autodoc: the site builds from a clean
checkout without the JAX or basix runtime (pip install -r docs/requirements.txt).
"""
project = "OpenSG"
author = "Akshat Bagla"
copyright = "2026, Akshat Bagla"
release = "2.2.0"

extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx_design",
    "sphinx_copybutton",
]
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "attrs_inline"]
myst_heading_anchors = 3
numfig = True
numfig_format = {"figure": "Figure %s", "table": "Table %s", "code-block": "Listing %s"}

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "latex"]

html_theme = "pydata_sphinx_theme"
html_title = "OpenSG"
html_static_path = ["_static"]
html_show_sourcelink = False
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/wenbinyugroup/OpenSG",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": [],
}
html_context = {
    "github_user": "wenbinyugroup",
    "github_repo": "OpenSG",
    "github_version": "main",
    "doc_path": "docs",
}

copybutton_prompt_text = r">>> |\$ "
copybutton_prompt_is_regexp = True
