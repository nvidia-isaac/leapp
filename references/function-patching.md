# Function patches for unsupported operations

Use this when an existing module-level function cannot be traced, including native-library calls nested inside helpers. Preserve the original application implementation and register a small adapter in the integration layer. Do not rewrite the whole pipeline, replace every call site, or convert all NumPy code to Torch merely because one operation is unsupported.

Consult the online [function-patching guide](https://nvidia-isaac.github.io/leapp/guides/patching.html) and [public API](https://nvidia-isaac.github.io/leapp/api/index.html); no source checkout is needed. NumPy, SciPy, scikit-learn, scikit-image, and OpenCV calls can return untraced values. Patch calls that actually break the trace, not every call from those libraries. Scikit estimator methods are not module-level targets; find the unsupported module function or a pure module-level helper enclosing the computation.

## Contract

```python
leapp.start(
    name="pipeline",
    patching=[leapp.FunctionPatch(module, "function_name", traceable_replacement)],
)
```

- `FunctionPatch` is a public `leapp` import. Check that it exists in the active installation before relying on it. Missing support is a version constraint, not a reason to invent a private patch API.
- The target is a callable attribute on a **Python module**, not an arbitrary object or bound instance method. Target the smallest unsupported function that explains the trace break.
- The adapter is called when arguments contain actively traced data (including nested supported containers); ordinary-array calls still use the original function. The adapter's traced operations become the exported computation.
- `global_patching=True` is required (the default). `stop()` restores session patches. Scope them with `try/finally`; do not leave a session active after an exception.
- Patches affect the whole Python process, including other threads. Avoid concurrent sessions with different patch lists. Stop and restart before changing patches.
- Calls must resolve the patched module attribute. References captured earlier by `from package import function`, default arguments, closures, or caches are not automatically rebound. Inspect the actual call path. Prefer patching the module attribute actually used, or a minimal integration-level binding adjustment, over changing every application call.

## Adapter requirements

Implement the same computation with supported traced Torch/NumPy operations. Preserve the signature used by the application, accepted argument combinations, output structure, shapes, dtype, device, layout, and numerical semantics (including relevant clipping, interpolation endpoints, or rounding).

A replacement must not call the patched function. Reject unsupported modes explicitly rather than silently approximating them. A patch is not an escape hatch for file I/O, simulator side effects, or stateful native code.

Keep genuinely dynamic values as traced data; do not extract them into Python scalars or copy raw buffers. Do not call the original unsupported operation on traced inputs and then attach metadata to its result: that still loses the computation.

## Example: preserve nested OpenCV callers

Existing helpers may call `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` in several places. This adapter supports that precise three-channel HWC operation without changing those helpers:

```python
import cv2
import torch
import leapp
from leapp import annotate


def traced_cvt_color(src, code, dst=None, dstCn=0):
    if code != cv2.COLOR_BGR2RGB or dst is not None or dstCn not in (0, 3):
        raise ValueError("Only BGR2RGB without a destination buffer is supported")
    if src.ndim != 3 or src.shape[-1] != 3:
        raise ValueError("Expected a three-channel HWC image")
    return torch.from_numpy(src).flip((-1,)).numpy()


leapp.start(
    name="vision",
    patching=[leapp.FunctionPatch(cv2, "cvtColor", traced_cvt_color)],
)
try:
    image = annotate.input_tensors("preprocess", {"bgr": sample_bgr})
    features = existing_image_pipeline(image)  # Existing helpers stay unchanged.
    annotate.output_tensors("preprocess", {"features": features}, export_with="onnx")
finally:
    leapp.stop()
leapp.compile_graph(validate=True, strict=True)
```

`sample_bgr` and `existing_image_pipeline` are application-provided. The online guide links a complete example; the [upstream example source](https://github.com/nvidia-isaac/leapp/blob/main/examples/function_patch.py) is also available without cloning LEAPP. Match examples to the package version used by the application.

## Verify before accepting

1. Compare the adapter with the original function on ordinary representative inputs, including relevant edge cases. Reject unsupported arguments deliberately.
2. Exercise the real nested application call path during tracing. Confirm it reaches the adapter and retains dependence on inputs.
3. Confirm ordinary calls still use the original behavior during the session and the module attribute is restored afterward, including failure cleanup where relevant.
4. Export and replay on **new values** against the unmodified original pipeline. Same-input validation can pass when an unsupported result was captured as a constant.

If a faithful pure adapter is impractical, inspect whether a compatible exporter can handle an already correctly traced graph. Otherwise expose the operation as an explicit external runtime boundary or propose a narrowly scoped application change, with the unsupported behavior and parity requirements stated. Never label a partial graph as a complete export.
