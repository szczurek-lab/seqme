# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "seqme"
author = "Møller-Larsen et al."

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

extensions = [
    "myst_nb",
    # "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",  # needs to be after napoleon
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
    "nbsphinx",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}

nbsphinx_execute = "never"
nb_execution_mode = "off"

autosummary_generate = True
autosummary_imported_members = True
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_custom_sections = [("Params", "Parameters")]
todo_include_todos = False
numpydoc_show_class_members = False
annotate_defaults = True
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]

templates_path = ["_templates"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "scanpydoc"  # "sphinx_rtd_theme"
html_title = "seqme"
html_static_path = ["_static"]
html_css_files = ["css/overwrite.css", "css/sphinx_gallery.css"]

html_logo = "_static/logo.svg"
html_show_sphinx = False

nbsphinx_thumbnails = {
    "tutorials/getting_started": "_static/logo.svg",
    "tutorials/time_series": "_static/logo.svg",
    "tutorials/benchmark_peptides": "_static/logo.svg",
    "tutorials/benchmark_rna": "_static/logo.svg",
    "tutorials/diagnostics": "_static/logo.svg",
    "tutorials/third_party": "_static/logo.svg",
    "tutorials/beyond_sequences": "_static/logo.svg",
}
