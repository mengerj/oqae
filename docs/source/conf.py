"""Sphinx configuration for the OQAE documentation.

Builds the HTML docs with autodoc + napoleon so the NumPy-style docstrings
across the public ``omvqvae`` API are rendered automatically. The package is
imported from the in-tree ``src/`` layout, so the docs always document the
checked-out source.
"""

from __future__ import annotations

import os
import sys

# ``omvqvae`` lives under ``src/`` (src layout). Make it importable for autodoc
# without requiring an editable install of the docs build environment.
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------

project = "OQAE"
author = "mengerj"
project_copyright = "2026, mengerj"

# Keep the documented version in lock-step with the package metadata.
try:
    from importlib.metadata import version as _pkg_version

    release = _pkg_version("oqae")
except Exception:  # pragma: no cover - fallback when not installed
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

exclude_patterns: list[str] = []

# -- Autodoc / Napoleon ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
# Move fully-annotated type hints out of the signature and into the parameter
# descriptions so the rendered signatures stay readable.
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Heavy / optional third-party deps are lazy-imported inside functions, but
# mock the ones that can be slow or environment-specific to import so the docs
# build stays fast and robust (e.g. on a docs-only CI runner).
autodoc_mock_imports = [
    "cellxgene_census",
    "tiledbsoma",
    "tiledbsoma_ml",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = True
napoleon_use_param = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = "OQAE — Omics Quantized Auto Encoder"
