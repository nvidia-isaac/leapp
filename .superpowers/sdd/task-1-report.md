#
# Task 1 Report: Add Dependency Direction And Visualization Package Shell
#

## What I Implemented

- Raised the project Python floor to `>=3.11` in `pyproject.toml`.
- Kept the existing `matplotlib` and `networkx` dependencies in place.
- Added the new visualization dependencies:
  - `fast-sugiyama>=0.5.3`
  - `Pillow>=10.0.0`
- Removed the Python 3.8, 3.9, and 3.10 classifiers so only 3.11 and 3.12 remain.
- Added `leapp/leapp_graph/visualization/` as an importable package shell:
  - `__init__.py` exports `visualize_graph` and the visual model types.
  - `model.py` defines the requested dataclasses and helper types.
  - `visualize.py` provides the temporary entrypoint that raises a clear runtime error if called.
- Added the import regression test in `tests/unit_tests/test_graph_visualization_imports.py`.
- Added `tests/conftest.py` to put the repo root on `sys.path` for pytest in this environment so the package can be imported during test runs.
- Updated `uv.lock` after the dependency change.

## Test Results

RED:

- First `uv run pytest tests/unit_tests/test_graph_visualization_imports.py -q` failed, but the failure was a dependency-resolution error caused by the existing Python floor being too low for the current dependency set.
- After the dependency update, the same test failed with `ModuleNotFoundError: No module named 'leapp'`, which exposed a pytest path issue in this repo environment.

GREEN:

- `uv run pytest tests/unit_tests/test_graph_visualization_imports.py -q`
  - Result: `2 passed`
- `uv run python - <<'PY' ... PY`
  - Result: `visualization deps import`
- Regression check:
  - `uv run pytest tests/unit_tests/test_conversion.py -q`
  - Result: `19 passed`

## Files Changed

- `pyproject.toml`
- `leapp/leapp_graph/visualization/__init__.py`
- `leapp/leapp_graph/visualization/model.py`
- `leapp/leapp_graph/visualization/visualize.py`
- `tests/unit_tests/test_graph_visualization_imports.py`
- `tests/conftest.py`
- `uv.lock`

## Self-Review Findings

- The new package shell matches the task brief and stays isolated from `graph_gui.py`.
- The model definitions are frozen dataclasses with the requested helper types and `visual_id()` normalization.
- The temporary renderer stub is intentionally unreachable unless the new package entrypoint is called directly.
- The pytest path shim is test-only and does not affect runtime package behavior.

## Concerns

- `tests/conftest.py` is a repo-level test harness adjustment that was necessary because `uv run pytest` could not import the local `leapp` package in this environment without it.
- `visualize.py` is still a stub by design; wiring the renderer remains for the next task.
