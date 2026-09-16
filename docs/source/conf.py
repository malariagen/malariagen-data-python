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
# browser-side JS at all, unlike the mermaid.js diagrams we tried first
# (which hit a real, reproducible mermaid bug interacting with
# pydata-sphinx-theme's dark/light mode switch - a crash on every page load,
# confirmed in both Chrome and Firefox). That makes this approach reliable
# across every browser at the cost of some polish (no custom fullscreen
# button, method lists, or colour-by-depth) - "sphinx.ext.viewcode" gives the
# clickable-to-source part instead, via a "[source]" link on each class's
# own doc entry (see architecture.rst).
inheritance_graph_attrs = dict(rankdir="LR", fontsize=16, ratio="compress")
inheritance_node_attrs = dict(
    fontsize=16,
    shape="box",
    style='"filled,rounded"',
    # Hex colours need to be quoted - Graphviz's dot rejects a bare
    # "#dcfce7" with a syntax error (verified: this broke the build until
    # quoted).
    fillcolor='"#dcfce7"',
    color='"#15803d"',
    fontcolor='"#14532d"',
)
inheritance_edge_attrs = dict(color='"#16a34a"')

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
