# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from meqpy import __version__

# -- Project information -----------------------------------------------------

project = "meqpy"
author = "Nils Krane, Andres Ortega-Guerrero, Gonçalo Catarina"
copyright = f"2026, {author}"
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # pull in docstrings from the code
    "sphinx.ext.napoleon",  # understand NumPy/Google docstring style
    "sphinx.ext.viewcode",  # add [source] links next to each object
    "sphinx.ext.intersphinx",  # link to numpy/scipy docs
    "sphinx.ext.mathjax",  # render LaTeX math in docstrings
    "myst_nb",  # render Jupyter notebooks as doc pages
]

# Markdown (MyST) extras
myst_enable_extensions = ["dollarmath", "amsmath"]

# The notebooks use GitHub-style `
myst_fence_as_directive = ["math"]

# Notebooks: render the outputs already saved in the .ipynb files,
# don't re-execute them on Read the Docs (no compute needed there).
nb_execution_mode = "force"

# The tutorials use raw-HTML anchors (<a id='...'>) for internal navigation.
# They work in the built pages, but myst can't verify them and would warn.
suppress_warnings = [
    "myst.xref_missing",  # raw-HTML anchors work but can't be verified
    "myst.header",  # notebooks jump H1 -> H3 for chapter numbering
]

templates_path = ["_templates"]

# Data folders that live next to the tutorial notebooks but are not doc pages.
exclude_patterns = [
    "tutorials/tutorial_files",
    "tutorials/dyson_from_2pz_vector",
    "**/.ipynb_checkpoints",
]

# Cross-link types like `numpy.ndarray` to the external docs.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "ase": ("https://ase.gitlab.io/ase/", None),
}

# autodoc defaults: document members in source order, include __init__ docs.
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"meqpy {__version__}"
