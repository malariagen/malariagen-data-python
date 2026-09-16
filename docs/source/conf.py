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

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

# Inheritance diagrams are pure server-side Graphviz -> static SVG, with no
# browser-side JS That makes this approach reliable
# across browsers. "sphinx.ext.viewcode" gives the
# clickable-to-source part, via a "[source]" link on each class's
# own doc entry (see architecture.rst).
inheritance_graph_attrs = dict(rankdir="RL", fontsize=20, ratio="compress")
inheritance_node_attrs = dict(  # https://graphviz.org/docs/nodes/
    fontsize=20,
    shape="box",
    style='"filled,rounded"',
    fillcolor='"#dcfce7"',
    color='"#15803d"',
    fontcolor='"#14532d"',
)
inheritance_edge_attrs = dict(color='"#16a34a"')  # https://graphviz.org/docs/edges/

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "switcher": {
        "check_switcher": True,
        "version_match": version,
        "json_url": "https://malariagen.github.io/malariagen-data-python/latest/_static/switcher.json",
    },
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
html_css_files = ["custom.css"]
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.ico"

graphviz_output_format = "svg"
