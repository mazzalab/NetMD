# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NetMD'
copyright = '2025, Manuel & Michele'
author = 'Manuel & Michele'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.duration',
    'sphinx_copybutton',
    "sphinx_inline_tabs",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

#html_theme = 'alabaster'
html_static_path = ['_static']

html_title = "NetMD Documentation"
html_favicon = '_static/img/favicon.ico'
html_scaled_image_link = False

html_theme_options = {
    "light_logo": "/img/netmd_logo_def.png",
    "dark_logo": "/img/netmd_logo_def.png",
}

def setup(app):
    app.add_css_file('custom.css')