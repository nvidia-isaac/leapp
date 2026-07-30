# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'LEAPP'
author = 'NVIDIA Corporation'
copyright = f'{datetime.now().year}, NVIDIA Corporation'

try:
    from importlib.metadata import version as _pkg_version
    release = _pkg_version('leapp')
except Exception:
    release = '0.6.0'
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    'nvidia_sphinx_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_tabs.tabs',
    'sphinxcontrib.spelling',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

sphinx_tabs_disable_tab_closing = True

copybutton_prompt_text = r'>>> |\$ '
copybutton_prompt_is_regexp = True
copybutton_copy_empty_lines = False

todo_include_todos = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}

# -- Spell checking ----------------------------------------------------------

spelling_lang = 'en_US'
tokenizer_lang = 'en_US'
spelling_word_list_filename = 'spelling_wordlist.txt'
spelling_show_suggestions = True
spelling_warning = True

# -- Link checking -----------------------------------------------------------

linkcheck_ignore = [
    # local file references in examples that aren't published URLs
    r'.*\.gitlab-master\.nvidia\.com.*',
]
linkcheck_anchors = False
linkcheck_workers = 10
linkcheck_timeout = 15
linkcheck_retries = 2

# -- HTML output -------------------------------------------------------------

html_theme = 'nvidia_sphinx_theme'

html_theme_options = {
    'copyright_override': {
        'start': 2025,
    },
    'collapse_navigation': False,
    'navigation_depth': -1,
}

html_static_path = ['_static']
html_css_files = ['css/leapp.css']

html_last_updated_fmt = '%b %d, %Y'
