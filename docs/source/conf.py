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
    "sphinxcontrib.mermaid",
]

autosummary_generate = True

# securityLevel must be "loose" for the `click ClassName href "..." "..."`
# links/tooltips in the mermaid class diagrams to work at all - mermaid
# silently drops that feature under the default "strict" level. fontSize
# is set here (rather than via a %%{init}%% line in each .mmd file) because
# sphinxcontrib-mermaid's raw-HTML template mangles the indentation of a
# diagram's first line, which can prevent that directive from being parsed.
# Both the top-level "fontSize" and "themeVariables.fontSize" are set because
# mermaid uses the top-level value to lay out/measure class diagram text, and
# themeVariables to generate the CSS - only setting one of the two had no
# visible effect.
mermaid_init_config = {
    "securityLevel": "loose",
    "fontSize": 20,
    "themeVariables": {"fontSize": "20px"},
}

# sphinxcontrib-mermaid's bundled CSS hard-codes `pre.mermaid > svg { height:
# {{ mermaid_height }} }` (default "500px"). Since the SVG preserves its
# aspect ratio, a large diagram (like the Ag3 class diagrams, ~30 classes)
# gets squashed to fit inside that fixed height, shrinking all the text with
# it regardless of fontSize above. "auto" lets the SVG take whatever height
# its own aspect ratio needs instead of being forced to shrink.
mermaid_height = "auto"

# Off by default; without this, scrolling/dragging on a diagram does nothing.
mermaid_d3_zoom = True

# mermaid_fullscreen (the "view in fullscreen" button) is a separate,
# top-level sphinxcontrib-mermaid config value - not a mermaid_init_config
# key - and is already enabled by default, so no setting is needed for it.

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
