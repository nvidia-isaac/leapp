# Advanced Export Configurations in LEAPP

This guide covers advanced export configurations in LEAPP. LEAPP relies on PyTorch's built-in export backends to generate models. For more details on the underlying export mechanisms, see:

- [torch.jit.script](https://docs.pytorch.org/docs/stable/generated/torch.jit.script.html) — Converts Python code to TorchScript via static analysis
- [torch.jit.trace](https://docs.pytorch.org/docs/stable/generated/torch.jit.trace.html) — Converts Python code to TorchScript by tracing execution
- [torch.onnx.export](https://docs.pytorch.org/docs/stable/onnx_export.html) — Exports models to ONNX format


## ONNX Export: Dynamo vs TorchScript

LEAPP supports two methods for exporting models to ONNX format:

1. **Dynamo-based export** (default) — Uses PyTorch 2.0's TorchDynamo for modern, graph-based export
2. **TorchScript-based export** (legacy) — Uses the traditional `torch.jit.script` approach

### Dynamo Export (Default)

Dynamo export is the modern approach introduced in PyTorch 2.0. It captures the computation graph using TorchDynamo and converts it to ONNX. TorchDynamo is the backend used for the new torch.export module.

```python
import torch
import leapp
from leapp import annotate

@annotate.method(export_with="onnx")  # Uses dynamo by default
def process_data(input_tensor: torch.Tensor):
    normalized = (input_tensor - input_tensor.mean()) / input_tensor.std()
    return torch.relu(normalized)

leapp.start(name="dynamo_example")
process_data(torch.randn(10))
leapp.stop()
leapp.compile_graph()
```
These options can be configured by passing a dict with the expected keys to `backend_params`.

**Advantages of Dynamo export:**
- Generally produces more optimized ONNX graphs
- Automatically performs testing
- Significantly reduces memory usage during export

### Using Pre-Scripted Models with ONNX Export

In some situations, especially when your logic involves many submodules, you may need to pre-script your function first. on top of setting `dynamo` to false This wraps your entire logic and scripts it using `torch.jit.script` first. This may help eliminate some bugs but generally produces a less optimal graph. ⚠️ **Use with caution**


### Backend Parameters for ONNX Export

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prescript` | `False` | Pre-script the model before ONNX export (only applies when `dynamo=False`) |
| `opset_version` | `None` | ONNX opset version (e.g., `17`). Uses PyTorch default if not specified |
| `verify` | `True` | Verify the exported model (dynamo only) |
| `optimize` | `True` | Apply ONNX optimizations (dynamo only) |
| `fallback` | `None` | Whether to fallback to the TorchScript exporter if the dynamo exporter fails. (dynamo only)|
| `report` | `False` | Generates a detailed export report showing which operations were converted and any issues encountered. (dynamo only)|


## Using Pre-Compiled Models: Export Without a Backend

Sometimes you have a pre-compiled model (e.g., a ONNX model, or TensorRT engine) that you want to include in your LEAPP graph without recompiling it, or there is a section of code that you want to add to the graph and provide the model details offline. LEAPP supports this use case by allowing you to set `export_with=None`.

### When to Use This Approach

Use this when you have:
- Pre-compiled models from external sources
- Models that were optimized with custom compilation pipelines
- Legacy models that you want to integrate into your graph
- Models that require special compilation flags or tools not supported by LEAPP's backends
- Models or processes that you cannot export

### Basic Usage

Instead of providing a backend name like `"jit"` or `"onnx"`, you can set `export_with=None` and specify the model path in `backend_params`:

```python
import torch
import leapp
from leapp import annotate

def use_precompiled_model():
    leapp.start(name="precompiled_example")
    
    # Some input data
    input_data = torch.randn(1, 10)
    
    # Reference a pre-compiled model without recompiling
    x = annotate.input_tensors("precompiled_inference", {"input_data": input_data})

    @annotate.method(export_with=None, backend_params={"model_path": "/path/to/model.pt"})
    def precompiled_inference(input_data):
        model = torch.jit.load("/path/to/model.pt")
        return model(input_data)

    predictions = precompiled_inference(x)
    
    leapp.stop()
    leapp.compile_graph()
    return predictions
```

### Backend Parameters for None Export

When using `export_with=None`, the following `backend_params` are available:

- **`model_path`** (optional): Path to the pre-compiled model file. If the model path is not provided a warning will be printed and the model is expected to be manually filled in at a later time.
- **`copy_original_model`** (optional): If `True`, copies the model file to the LEAPP output directory. If `False` (default), the YAML will reference the original path.

```python
@annotate.method(
    export_with=None,
    backend_params={
        "model_path": "/models/my_optimized_model.pt",
        "copy_original_model": True  # Copy model to output directory
    }
)
def inference_with_precompiled(data):
    model = torch.jit.load("/models/my_optimized_model.pt")
    return model(data)
```

### Important Considerations

**⚠️ No Compilation or Validation**

When using `export_with=None`:
- LEAPP does **not** compile or modify the model
- LEAPP does **not** validate that the model exists during tracing
- The model path is only verified during `compile_graph()` when generating the YAML

**⚠️ Input/Output Shape Tracing**

LEAPP still traces the input and output shapes based on the actual data flowing through your code. Make sure:
- The tensor shapes you use during tracing match what your pre-compiled model expects
- You provide representative data shapes for accurate graph generation

**⚠️ Model Path in Generated YAML**

If `copy_original_model=False` (default):
- The YAML will contain the original absolute path to the model
- Ensure the model is accessible at that path in your deployment environment
- Consider using relative paths or environment variables for portability

If `copy_original_model=True`:
- The model is copied to the LEAPP output directory
- The YAML references the copied model
- This increases output directory size but improves portability

### Best Practices

1. **Use absolute paths during development**: Makes debugging easier
2. **Copy models for deployment**: Set `copy_original_model=True` for production bundles
3. **Verify model compatibility**: Ensure your pre-compiled model's input/output shapes match the traced shapes
4. **Document model provenance**: Add comments explaining where the pre-compiled model came from and how it was created