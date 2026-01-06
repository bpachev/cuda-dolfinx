# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import sys
import cudolfinx

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cuDOLFINx'
now = datetime.datetime.now()
date = now.date()
copyright = f'{date.year}, Benjamin Pachev, James Trotter'
author = 'Benjamin Pachev, James Trotter'
release = cudolfinx.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ['_templates']
exclude_patterns = []

todo_include_todos = True
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []

napoleon_google_docstring = True
napoleon_use_admonition_for_notes = False

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "imported-members": True,
    "undoc-members": True,
}
autosummary_generate = True
autosummary_ignore_module_all = False
autoclass_content = "both"

codeautolink_concat_default = True
