<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LEAPP (this repository)

Instructions for coding agents working **in** nvidia-isaac/leapp: where
code lives, which tests to run, and which skill to load. This is not a
guide for annotating a downstream policy; that lives in
`docs/source/` and at https://nvidia-isaac.github.io/leapp/

Python 3.10+ and PyTorch 2.6.0+. Visualization extras install only on
3.11+. Environment is a local venv (typically `uv`), never conda.

## Load this first

| If you are changing… | Read this before editing |
|---|---|
| Tracing, `TracedData` / `TracedTensor` / `TracedNpArray`, indexing, mutation, node-boundary identity, FX proxies, NumPy dispatch | `.cursor/skills/leap_agent/SKILL.md` |
| Public annotate/lifecycle API or docs examples | `docs/source/api/index.rst` plus the matching guide under `docs/source/` |
| Graph PNG layout | `packages/leapp-visualization/` |
| Export backends | `leapp/backends/` |
| YAML / `InferenceManager` | `leapp/leapp.py` (`compile_graph`), `leapp/inference_manager.py`, `docs/source/generated_configs.rst` |

Do not mix the skill's current `(context_obj, output_port)` identity with
legacy `leapp_tag` helpers. The skill states which checkout you have.

## Tree (start here)

```
leapp/                      # installable package
  leapp.py                  # start / stop / compile_graph / annotate facade
  export_manager.py         # annotate.* implementations
  inference_manager.py      # runtime loader for exported YAML
  backends/                 # jit, onnx-dynamo, onnx-torchscript, pt2, none
  leapp_graph/
    traced_node.py          # per-node FX capture and compile
    leapp_graph.py          # pipeline wiring, feedback, visualize hook
    datatypes/              # TracedData, TracedTensor, TracedNpArray, patching
  utils/                    # logging, TensorSemantics, enums, GraphConfigs
packages/leapp-visualization/   # static PNG layout (ships with leapp)
docs/source/                # Sphinx (NVIDIA theme)
examples/                   # runnable samples; CI runs tests/test_examples/
tests/
  unit_tests/               # tracer, conversion, export-format helpers
  functional_tests/         # annotate, backends, state, inference, failures
  test_examples/            # examples/* must keep passing
```

Public imports are only what `leapp/__init__.py` exports. Annotation names
allowed on `leapp.annotate` are listed in `AnnotateAPI._ALLOWED_APIS` in
`leapp/leapp.py`. Do not add a consumer API without updating both, the
docs, and tests.

## Commands

```bash
python -m pip install -e ".[dev]"
pytest tests/ packages/leapp-visualization/tests/ -v
```

Narrower (use these while iterating):

| Change | Run |
|---|---|
| Torch tracing / indexing | `pytest tests/unit_tests/test_traced_tensor.py tests/unit_tests/export_format_validation.py -v` |
| NumPy tracing | `pytest tests/unit_tests/test_traced_np_array.py tests/unit_tests/test_numpy_compatibility.py tests/unit_tests/test_conversion.py -v` |
| Annotate / I/O / state | `pytest tests/functional_tests/ -v` |
| Examples | `pytest tests/test_examples/ -v` |
| Visualization | `pytest packages/leapp-visualization/tests/ -v` (Python 3.11+) |

Docs (CI uses `-W`; spelling needs `enchant-2`):

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -b html -W --keep-going docs/source docs/build/html
python -m sphinx -b spelling -W --keep-going docs/source docs/build/spelling
python -m sphinx -b linkcheck docs/source docs/build/linkcheck
```

Add unknown words to `docs/source/spelling_wordlist.txt`. New screenshots:
`docs/source/_static/images/`. `*.png` and `*.gif` are Git LFS (see
`.gitattributes`).

Commits must be DCO signed (`git commit -s`). See `CONTRIBUTING.md`.

## Version bump

Keep these in lockstep: `leapp/__init__.py` (`__version__`),
`pyproject.toml`, `packages/leapp-visualization/pyproject.toml`,
`docs/source/conf.py` fallback, and example YAML `leapp version` in
`docs/source/api/index.rst` and `docs/source/generated_configs.rst`.
`__config_version__` is the YAML schema, not the PyPI version; bump it
only when the config format changes.

`main` is often a **squash** of `develop`. `git log main..develop` can
list hundreds of SHAs that are already in the squash; the merge-base
diff (`git diff origin/main...HEAD`) is the true delta.

## Docs map (`docs/source/`)

| Page | Topic |
|---|---|
| `getting_started.rst` | First annotated pipeline |
| `ecosystem.rst` | Isaac Lab / Isaac ROS Deploy links |
| `guides/nodes.rst` | `method` vs `input_tensors` / `output_tensors` |
| `guides/export.rst` | Backends and `export_with` |
| `guides/graph.rst` | State, `annotate.module`, feedback |
| `guides/buffers.rst` | `static_outputs`, `mirror_leapp_tags` |
| `guides/debugging.rst` | Logs, dry_run, `non_traced` |
| `guides/runtime.rst` | `compile_graph(validate=...)` |
| `leapp_runtime.rst` | `InferenceManager` |
| `semantics/` | `TensorSemantics`, kinds, `TemporalAxis` |
| `generated_configs.rst` | YAML shape |
| `api/index.rst` | Public signatures |

User-facing integration recipes belong here, not in this file.

## Guardrails when changing behavior

- Prefer extending `TracedData` once (`_map_structure`, index lowering,
  port preserve/clear) over copying logic into tensor and NumPy classes.
- After shared indexing changes: both torch and NumPy unit suites, plus
  FX / jit / exported-program coverage as the skill describes.
  `onnx-dynamo` for modern index cases; do not require legacy ONNX if
  that exporter cannot lower the op.
- Do not invent graph edges for data a node did not publish. Clearing
  `output_port` is the correct default after a transforming op.
- Examples in `examples/` are CI. If you change annotate semantics,
  update the matching `tests/test_examples/` case.
- Conventional commits (`feat`, `fix`, `docs`, `chore`, …). Do not bump
  the version unless the user asked for a release.
