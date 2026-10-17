# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'Isaac LEAPP'
author = 'NVIDIA Corporation'
copyright = f'{datetime.now().year}, NVIDIA Corporation'

try:
    from importlib.metadata import version as _pkg_version
    release = _pkg_version('leapp')
except Exception:
    release = '0.7.2'
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
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '_redirects']

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
html_title = 'Isaac LEAPP: machine learning pipeline export'
html_short_title = 'Isaac LEAPP'
html_baseurl = 'https://nvidia-isaac.github.io/leapp/'

html_theme_options = {
    'copyright_override': {
        'start': 2025,
    },
    'collapse_navigation': False,
    'navigation_depth': -1,
}

html_static_path = ['_static']
html_extra_path = ['_redirects', 'llms.txt']
html_css_files = ['css/leapp.css']

html_last_updated_fmt = '%b %d, %Y'


def package_agent_skill(app, exception):
    """Bundle the repository skill and its references with the published HTML."""
    if exception is not None or app.builder.format != 'html':
        return

    repo_root = Path(__file__).resolve().parents[2]
    download_dir = Path(app.outdir) / '_downloads'
    download_dir.mkdir(parents=True, exist_ok=True)
    files = [repo_root / 'SKILL.md', *sorted((repo_root / 'references').rglob('*.md'))]
    with ZipFile(download_dir / 'leapp-integrate.zip', 'w', ZIP_DEFLATED) as bundle:
        for source in files:
            name = (Path('leapp-integrate') / source.relative_to(repo_root)).as_posix()
            bundle.writestr(name, source.read_bytes())


def setup(app):
    app.connect('build-finished', package_agent_skill)
