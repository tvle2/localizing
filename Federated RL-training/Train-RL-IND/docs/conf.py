# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


from sys import path
from os.path import abspath
from sphinx_pyproject import SphinxConfig

project = 'qsensoropt'

path.insert(0, abspath("../src"))
path.insert(0, abspath("../examples"))

config = SphinxConfig("../pyproject.toml", globalns=globals())

copyright = '2022, Federico Belliardo, Fabio Zoratti'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.imgmath',
    # 'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


add_module_names = False
math_number_all = True
imgmath_image_format = "svg"
imgmath_font_size = 15
autoclass_content = 'both'

numfig = True
math_numfig = True
numfig_secnum_depth = 2
math_eqref_format = "Eq.{number}"