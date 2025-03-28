# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "malariagen_data API"
copyright = "2024, MalariaGEN"
author = "MalariaGEN"
version = os.environ.get("VERSION_TAG", "dev")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autosummary", "sphinx_design"]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navbar_center": ["version-switcher", "navbar-nav"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/malariagen/malariagen-data-python",
            "icon": "fa-brands fa-github",
        }
    ],
}
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.ico"
