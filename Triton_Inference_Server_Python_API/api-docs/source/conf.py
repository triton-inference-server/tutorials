# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from __future__ import annotations

import os
import sys

sys.path.insert(
    0, os.path.abspath("/usr/local/lib/python3.10/dist-packages/tritonserver")
)

project = "Triton Inference Server In-Process Python API"
copyright = "2024, NVIDIA"
author = "NVIDIA"
release = "BETA"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
markdown_anchor_signatures = True
markdown_anchor_sections = True
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx_markdown_builder"]

templates_path = ["_templates"]
exclude_patterns = []
autodoc_typehints_format = "short"
autoclass_content = "both"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
