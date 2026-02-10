# LEAPP - Lightweight Export Annotations for Policy Pipelines

## Overview

LEAPP is a Python package for tracing and exporting computational graphs from PyTorch code. It is designed for robotics and autonomous agent applications where efficient policy export is crucial.

The core workflow: annotate PyTorch code with lightweight markers, trace the computation, and export to formats like TorchScript, ONNX, with automatic graph visualization and YAML specifications.

## Repository Structure

- `leapp/` — Core library
  - `export_manager.py` — Main `annotate` singleton, user-facing API (start, stop, input_tensors, output_tensors, state_tensors, etc.)
  - `leapp_graph/` — Graph construction and node types
    - `leapp_graph.py` — Graph class, connection detection via leapp_tag matching
    - `leapp_node.py` — Base node class, I/O descriptions, backend setup
    - `traced_node.py` — TracedTensorNode for tensor-level tracing via torch.fx
    - `datatypes.py` — TracedTensor wrapper
  - `backends/` — Export backends (onnx, torchscript, etc.)
  - `protocols.py` — StatefulModel protocol for recurrent model export
  - `utils.py` — TensorDescription dataclass, flatten/resolve helpers
- `examples/` — Usage examples (getting_started, wbc, stateful_policy, etc.)
- `tests/` — Unit and functional tests
  - `tests/functional_tests/` — End-to-end tests including state tensor tests

## Key Concepts

- **Three annotation methods**: `@annotate.method()` decorator, `annotate.block()` context manager, `annotate.input_tensors()`/`annotate.output_tensors()` traced tensors
- **State tensors**: Tensors that are both inputs and outputs (RNN hidden states, history buffers). Use `annotate.state_tensors()` and `annotate.update_state()`.
- **leapp_tag**: Tag on tensors for automatic connection detection between nodes. Format: `{node_name}/{tensor_name}/`
- **StatefulModel protocol**: Interface for generic export of recurrent models (`get_state_spec`, `stateless_forward`)
- **ONNX RNN export**: Use `export_with="onnx-torchscript"` for models with nn.GRU/nn.LSTM. The default dynamo exporter decomposes RNNs into invalid Slice ops.

## Rules

- **Commit messages**: Do NOT add `Co-Authored-By: Claude` or any Claude attribution to commit messages.
- **Tests**: Run functional tests with `python -m pytest tests/functional_tests/ -v`
- **Examples**: Examples should be self-contained and runnable with `python examples/<name>.py`
