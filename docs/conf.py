# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

os.environ['__SPHINX_BUILD__'] = 'True'

# -- Project information -----------------------------------------------------

project = 'BQSKit'
copyright = '2022, Lawrence Berkeley National Laboratory'
author = 'BQSKit Development Team'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.imgmath',
    'sphinx_rtd_theme',
    'myst_parser',
    'jupyter_sphinx',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {'navigation_depth': 2}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

autodoc_type_aliases = {
    'tuple': 'tuple',
    'list': 'list',
    'dict': 'dict',
}
# autodoc_type_aliases = {
#     'StateLike': 'bqskit.qis.state.StateLike',
#     'CircuitLocationLike': 'bqskit.ir.CircuitLocationLike',
#     'CircuitPointLike': 'bqskit.ir.CircuitPointLike',
#     'CircuitRegionLike': 'bqskit.ir.CircuitRegionLike',
#     'IntervalLike': 'bqskit.ir.IntervalLike',
# }
# napoleon_type_aliases = autodoc_type_aliases
autodoc_typehints = 'both'
autodoc_typehints_description_target = 'all'
autoclass_content = 'class'
nbsphinx_output_prompt = 'Out[%s]:'

add_module_names = False
modindex_common_prefex = ['bqskit.']
autosummary_generate = True
autosummary_generate_overwrite = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_references = True
napolean_custom_sections = ['Invariants']
napoleon_preprocess_types = True
napoleon_use_rtype = True
myst_enable_extensions = ['dollarmath', 'amsmath']
autodoc_mock_imports = [
    'scipy',
    'numpy',
    'qiskit',
    'lark-parser',
    'hypothesis',
    'pytket',
    'cirq',
    'qutip',
    'dill',
]
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'
