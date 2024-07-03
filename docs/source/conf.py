import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../../src'))
os.environ["SPHINX_BUILD"] = "1"

# -- Project information -----------------------------------------------------

project = u"TissueMosaic"
copyright = u""
author = u"Sandeep Kambhampati, Luca Dalessio, and Fedor Grab"

version = ""

if "READTHEDOCS" not in os.environ:
    from tissuemosaic import __version__  # noqaE402
    version = __version__
    html_context = {"github_version": "master"}
# release version
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.imgconverter',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    #'sphinx_search.extension',
    'sphinx.ext.napoleon',
    #'sphinx_autodoc_typehints',
]


autodoc_inherit_docstrings = True

# Napoleon settings (for google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Make the return section behave like the args section
napoleon_custom_sections = [('Returns', 'params_style')]

PACKAGE_MAPPING = {
    "PyYAML": "yaml",
    "neptune-client": "neptune",
    "google-cloud": "google",
    "scikit-learn": "sklearn",
    "umap_learn": "umap",
    "pytorch-lightning": "pytorch_lightning",
    "lightning-bolts": "pl_bolts",
}
MOCK_PACKAGES = [
    'numpy',
    'anndata',
    'scanpy',
    'leidenalg',
    'igraph',
    'google-cloud',
    'scikit-learn',
    'pytorch-lightning',
    'torch',
    'torchvision',
    'matplotlib',
    'umap_learn',
    'neptune-client',
    'scipy',
    'pandas',
    'lightly',
    'lightning-bolts',
    'seaborn',
]
MOCK_PACKAGES = [PACKAGE_MAPPING.get(pkg, pkg) for pkg in MOCK_PACKAGES]
autodoc_mock_imports = MOCK_PACKAGES

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = [".rst", ".ipynb"]
nbsphinx_execute = "never"
html_sourcelink_suffix = ""
master_doc = "index"
language = "en"
exclude_patterns = [
    ".ipynb_checkpoints",
    "notebooks/*ipynb",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# do not prepend module name to functions
add_module_names = False

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = ""

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"  # "alabaster" #"sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_style = "css/tissuemosaic.css"
htmlhelp_basename = "tissuemosaicdoc"
