# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "ascribe-link"
copyright = "2026, The Regents of the University of California, through Lawrence Berkeley National Laboratory"
author = "Ronald Pandolfi"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_immaterial",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
# claude-agent-sdk is an optional extra; don't require it to build docs.
autodoc_mock_imports = ["claude_agent_sdk"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_immaterial"
html_static_path = ["_static"]
html_title = "ascribe-link"

html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/git-alt",
    },
    "site_url": "https://ascribe-link.readthedocs.io/en/latest/",
    "repo_url": "https://github.com/lbl-camera/ascribe-link",
    "repo_name": "ascribe-link",
    "edit_uri": "blob/master/docs",
    "globaltoc_collapse": True,
    "features": [
        "navigation.expand",
        "navigation.sections",
        "navigation.top",
        "search.highlight",
        "search.share",
        "toc.follow",
        "toc.sticky",
        "content.tabs.link",
        "announce.dismiss",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "blue",
            "accent": "light-blue",
            "toggle": {
                "icon": "material/brightness-7",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "blue",
            "accent": "light-blue",
            "toggle": {
                "icon": "material/brightness-4",
                "name": "Switch to light mode",
            },
        },
    ],
    "toc_title_is_page_title": True,
}

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
