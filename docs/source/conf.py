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
# silently drops that feature under the default "strict" level.
# Both the top-level "fontSize" and "themeVariables.fontSize" are set because
# mermaid uses the top-level value to lay out/measure class diagram text, and
# themeVariables to generate the CSS - only setting one of the two had no
# visible effect.
# theme is "base" (not e.g. "forest") because mermaid only fully honors
# custom themeVariables on top of "base" - other named themes precompute
# most of their own derived colours, so e.g. edges/borders/member text kept
# showing that theme's own accent colour (orange, with "forest") instead of
# ours no matter what we set here.
mermaid_init_config = {
    "securityLevel": "loose",
    "fontSize": 20,
    "theme": "base",
    "themeVariables": {
        "fontSize": "20px",
        "primaryColor": "#dcfce7",
        "primaryBorderColor": "#15803d",
        "primaryTextColor": "#14532d",
        "lineColor": "#16a34a",
        "secondaryColor": "#bbf7d0",
        "tertiaryColor": "#f0fdf4",
        "classText": "#14532d",
        "nodeBorder": "#15803d",
        "clusterBkg": "#f0fdf4",
        "clusterBorder": "#15803d",
    },
    # Class diagrams hard-code their own text sizes internally (member text
    # at 10px, the class name at 18px, "themeVariables.fontSize" above does
    # not reach either - verified by rendering with mermaid-cli and reading
    # the generated CSS). themeCSS is the one mechanism that reliably
    # overrides this: it's appended as a raw, literal stylesheet after
    # mermaid's own, so it wins regardless of theme.
    "themeCSS": ".classTitle { font-weight: 800 !important; } "
    ".classTitleText { font-size: 24px !important; font-weight: 800 !important; }",
}
# The class diagrams set their own per-node colours with `style` lines (see
# the .mmd files), so light/dark mode doesn't need to swap palettes - keep
# both the same to avoid the diagram jumping between two colour schemes.
mermaid_light_theme = "base"
mermaid_dark_theme = "base"

# sphinxcontrib-mermaid's bundled CSS hard-codes `pre.mermaid > svg { height:
# {{ mermaid_height }} }` (default "500px"). "auto" lets the SVG take whatever height
# its own aspect ratio needs instead of being forced to shrink.
mermaid_height = "auto"

# Off despite the name implying otherwise: enabling this crashes mermaid on
# this site specifically. pydata-sphinx-theme applies its light/dark theme
# choice just after page load, which the mermaid JS is watching for (to
# re-render diagrams in the right theme) - and that theme-triggered re-render
# throws "TypeError: Cannot read properties of null (reading 'firstChild')"
# inside mermaid's own bundle when the d3-zoom code has already wrapped the
# SVG's contents in an extra <g> beforehand. Confirmed with a headless
# Chrome + CDP session: this uncaught exception aborts the rest of
# runMermaid() before it gets to building the fullscreen button, which is
# why "mermaid_fullscreen = True" below had no visible effect while this
# was on. The fullscreen view still auto-fits the diagram to the viewport
# without this, just without extra scroll/drag pan-zoom on top of that.
mermaid_d3_zoom = False

# Fullscreen-view button ("⛶") in the corner of each diagram. Already the
# default, set explicitly here for clarity. Opacity raised from the default
# 50 - at 50% against a light diagram it's easy to miss.
mermaid_fullscreen = True
mermaid_fullscreen_button_opacity = "85"

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
