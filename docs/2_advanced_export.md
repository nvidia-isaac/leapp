# Advanced Export Configurations in LEAPP

This guide covers export backend selection and advanced export options in the current LEAPP API.

## Backend Names and Aliases

LEAPP supports these public backend names:

| `export_with` value | Actual backend | Output |
|---|---|---|
| `"jit"` | `jit-script` | TorchScript `.pt` |
| `"jit-script"` | `jit-script` | TorchScript `.pt` |
| `"jit-trace"` | `jit-trace` | TorchScript `.pt` |
| `"onnx"` | `onnx-dynamo` | ONNX `.onnx` |
| `"onnx-dynamo"` | `onnx-dynamo` | ONNX `.onnx` |
| `"onnx-torchscript"` | `onnx-torchscript` | ONNX `.onnx` |
| `None` | `NoneExportBackend` | No compilation |

Recommended defaults:
- start with `"jit"` for the fastest bring-up
- use `"onnx"` when you want the default ONNX exporter
- use `"onnx-torchscript"` for recurrent models such as `nn.GRU` and `nn.LSTM`

## TorchScript Export

`"jit"` and `"jit-script"` select the TorchScript scripting backend.
`"jit-trace"` selects the tracing backend.

```python
import torch
import leapp
from leapp import annotate

leapp.start("torchscript_example")

x = annotate.input_tensors("normalize", {"x": torch.randn(16)})
y = torch.relu((x - x.mean()) / (x.std() + 1e-6))
annotate.output_tensors("normalize", {"y": y}, export_with="jit")

leapp.stop()
leapp.compile_graph(validate=True)
```

## ONNX Export: `onnx-dynamo` vs `onnx-torchscript`

LEAPP exposes two ONNX backends:

- `onnx-dynamo` is the default behind `export_with="onnx"`
- `onnx-torchscript` is the TorchScript-based ONNX path

Use `onnx-dynamo` for typical feedforward models.
Use `onnx-torchscript` when the dynamo path produces unstable graphs or when exporting recurrent models (e.g. `nn.GRU`, `nn.LSTM`). See `examples/stateful_gru_export.py` for a complete example.

```python
import torch
import leapp
from leapp import annotate

leapp.start("onnx_example")

x = annotate.input_tensors("policy", {"obs": torch.randn(1, 32)})
action = torch.tanh(x[..., :12])
annotate.output_tensors("policy", {"action": action}, export_with="onnx")

leapp.stop()
leapp.compile_graph(validate=True)
```

### ONNX backend parameters

All ONNX backend parameters are passed through `backend_params`.

| Parameter | Default | Used by | Description |
|---|---|---|---|
| `opset_version` | PyTorch default | both ONNX backends | Override the ONNX opset version |
| `report` | `False` | both ONNX backends | Emit exporter diagnostics/reporting |
| `verify` | `False` | `onnx-dynamo` | Enable exporter-side verification |
| `optimize` | `True` | `onnx-dynamo` | Enable dynamo ONNX optimization |
| `fallback` | `None` | `onnx-dynamo` | Configure exporter fallback behavior |
| `prescript` | `False` | `onnx-torchscript` | Script the module before ONNX export |
| `skip_validation` | `False` | both ONNX backends at save time | Skip `onnx.checker.check_model()` when saving |

Example:

```python
annotate.output_tensors(
    "policy",
    {"action": action},
    export_with="onnx-dynamo",
    backend_params={
        "opset_version": 17,
        "verify": False,
        "optimize": True,
        "report": True,
    },
)
```

### Validation guidance

Exporter-side options like `verify` are optional and backend-specific.
The main LEAPP validation path is still:

```python
leapp.compile_graph(validate=True, rtol=1e-3, atol=1e-5, strict=True)
```

That validation compares exported model outputs against the captured traced outputs.

## Export Without Compilation: `export_with=None`

Set `export_with=None` when you want the node to appear in the graph without asking LEAPP to compile it.
This is useful for:
- prebuilt `.pt` or `.onnx` artifacts
- placeholder nodes that will be filled in later
- flows where LEAPP should capture I/O and graph edges but not produce a model

```python
import torch
import leapp
from leapp import annotate

leapp.start("precompiled_example")

x = annotate.input_tensors("precompiled_inference", {"input_data": torch.randn(1, 10)})
predictions = x  # representative traced output shape

annotate.output_tensors(
    "precompiled_inference",
    {"predictions": predictions},
    export_with=None,
    backend_params={
        "model_path": "/models/my_optimized_model.pt",
        "copy_original_model": True,
    },
)

leapp.stop()
leapp.compile_graph()
```

### `None` backend parameters

- `model_path` (optional): Path to an existing model artifact
- `copy_original_model` (optional, default `False`): Copy the provided artifact into the LEAPP output directory
- `device` (optional): Device hint used when loading the referenced artifact

### Important behavior

- LEAPP still records input/output shapes and graph connectivity from the traced example tensors
- if `model_path` is provided, LEAPP verifies the file and stores checksums in the YAML
- the model path written into the YAML is made relative to the YAML directory when possible
- `InferenceManager` currently only runs referenced `.pt` and `.onnx` artifacts

## Dry Run and Selective Non-Traced Nodes

Sometimes you want LEAPP to preserve graph structure and connectivity without fully compiling every node.
LEAPP provides three related options, and they behave differently:

- `leapp.start(..., dry_run=True)` makes the entire trace metadata-only from the start
- `leapp.start(..., non_traced=[...])` disables tracing/export for only selected nodes
- `leapp.compile_graph(..., dry_run=True)` keeps an already-captured trace but skips compile/save/validate

### `start(dry_run=True)`: whole-graph metadata-only mode

Use this when you want to explore graph boundaries, graph I/O, and connectivity without paying export cost.

In this mode:
- `input_tensors()` and related APIs return normal tensors instead of `TracedTensor`
- LEAPP still tags outputs so graph connectivity can be detected
- YAML and graph structure are still produced
- model files are not exported

```python
leapp.start("debug_graph", dry_run=True)
# ... run your traced code ...
leapp.stop()
leapp.compile_graph()
```

This is useful for:
- debugging node boundaries
- checking graph inputs/outputs quickly
- validating connectivity before expensive export

### `non_traced=[...]`: selective not-compiled / not-traced nodes

This is the selective option when only some nodes should stay in the graph but should not be traced through or compiled.

This is especially useful because traced-tensor nodes normally try to trace through the computation inside the node. For some nodes, that is exactly what you do **not** want:
- the code may call into functionality that is not trace-friendly
- the node may intentionally act as a placeholder or opaque stage
- tracing through that node may raise errors even though you still want it represented in the graph

With `non_traced=[...]`, LEAPP lets that node run on normal tensors, skips export for it, but still tags its outputs so downstream traced nodes can connect to it.

```python
import torch
import leapp
from leapp import annotate

leapp.start("mixed_graph", non_traced=["raw_node"])

x = annotate.input_tensors("raw_node", {"x": torch.tensor([1.0, 2.0, 3.0])})
raw_y = x * 2.0
annotate.output_tensors("raw_node", {"y": raw_y}, export_with="jit")

traced_y = annotate.input_tensors("traced_node", {"y": raw_y})
traced_z = traced_y + 1.0
annotate.output_tensors("traced_node", {"z": traced_z}, export_with="jit")

leapp.stop()
leapp.compile_graph(validate=True)
```

Result:
- `raw_node` appears in the graph
- `raw_node` outputs still connect to `traced_node`
- `raw_node` does not produce a compiled model artifact
- `traced_node` is still traced and exported normally

### `compile_graph(dry_run=True)`: skip export after tracing

Use this when you want a normal trace session first, but want to skip compile/save/validate at the final export step.

```python
leapp.start("captured_graph")
# ... normal tracing ...
leapp.stop()
leapp.compile_graph(dry_run=True, validate=False)
```

This differs from `start(dry_run=True)`:
- tracing still happens normally during the session
- FX graphs and node traces are still built
- compile/save/validate are skipped only at the end

### Choosing the right option

- Use `start(dry_run=True)` when the whole graph should be metadata-only
- Use `non_traced=[...]` when only specific nodes should stay uncompiled / untraced
- Use `compile_graph(dry_run=True)` when you already did a real trace and only want to skip final export work