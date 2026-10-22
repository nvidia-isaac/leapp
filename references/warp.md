# Warp inside a LEAPP pipeline

This is the complete Warp integration reference bundled with `leapp-integrate`. Read it before tracing Warp; a separate skill or source checkout is unnecessary. Warp support is experimental. Consult the online [setup](https://nvidia-isaac.github.io/leapp/tensor_libraries/installation.html), [usage](https://nvidia-isaac.github.io/leapp/tensor_libraries/warp.html), [capture model](https://nvidia-isaac.github.io/leapp/tensor_libraries/warp_how_it_works.html), and [limitations](https://nvidia-isaac.github.io/leapp/tensor_libraries/warp_limitations.html) for the release used by the application.

LEAPP records Warp regions as `leapp::warp_runner` custom operations, embedding APIC archives and compiled kernel modules in an ONNX or PT2 model. It does not translate kernel bodies to Torch operators. Keep the full compute pipeline in one node by default; several Warp regions and Torch operations can share that node.

## Set up the export environment

Current export/capture requirements are Linux, a CUDA GPU, Warp >=1.18, and a matching CUDA toolkit with `nvcc`. Native builds also need a C++17 compiler, CMake >=3.18, and network access to download ONNX Runtime headers. PTX is currently targeted at compute capability 7.5: older GPUs and kernels relying on newer hardware features are outside this export contract. Verify version-specific requirements before installing.

Use the application's environment and package manager. A default `leapp` installation does not install Warp support. Select **one** extra matching the CUDA toolkit (not merely the driver's maximum supported CUDA version):

```bash
python -m pip install "leapp[warp-cu12]"  # CUDA 12; respect the project's LEAPP version pin
# For CUDA 13, use "leapp[warp-cu13]" instead.
leapp-build-warp-runtime
leapp-build-warp-runtime --status
```

For an actual LEAPP source checkout managed by uv, `uv sync --extra test --extra warp-cu12` (or `warp-cu13`) is the alternative. Do not assume a downstream project defines these extras.

The CUDA extras select compatible `cupti-python` and `onnxruntime-gpu` packages with CUDA/cuDNN runtime dependencies. Do not independently install a mismatched CUPTI major or add a competing CPU ONNX Runtime package. CUDA 13's GPU runtime package requires Python >=3.11. Check `nvcc --version`, `python -m pip check`, and `onnxruntime.get_available_providers()` in the export environment; Warp ONNX replay needs `CUDAExecutionProvider`.

The build command builds **both ONNX and PT2 native adapters**, checking rather than installing dependencies. Default discovery uses the user cache:

- Linux: `${XDG_CACHE_HOME:-~/.cache}/leapp/warp-runtime/build`
- Windows native runtime builds: `%LOCALAPPDATA%\leapp\warp-runtime\build`

No environment variable is required for a valid cached build. For isolation between environments, overrides are available:

```bash
export LEAPP_WARP_ONNX_CUSTOM_OP_LIBRARY="/path/to/libleapp_wrp_onnx_custom_op.so"
export LEAPP_WARP_PT2_CUSTOM_OP_LIBRARY="/path/to/libleapp_wrp_torch_custom_op.so"
```

Use actual built libraries in place of these example paths. An invalid override is an error, not a fallback to the cache. The shared cache is not versioned per environment. After changing Torch, Warp, ONNX Runtime, or CUDA, rebuild and restart processes that loaded the libraries. Account for other environments using that cache before replacing a shared build.

Windows library builds use corresponding `.dll` names and need `LEAPP_WARP_IMPORT_LIBRARY` pointing to `warp.lib`, which the Warp wheel does not ship. This build capability does **not** enable Linux-only LEAPP tracing on Windows. Capture uses CUPTI; replay of an exported bundle is a separate dependency check and is not inherently CUPTI-dependent.

## Capture and export

1. Initialize Warp with `wp.init()` before `leapp.start(..., global_patching=True)` and before compiling application kernels under incompatible settings. Keep global patching enabled.
2. Pass application inputs through `annotate.input_tensors` and use returned values. Public Warp calls involving actively traced Warp arrays participate in capture. Work on unrelated raw arrays may be absent from the export.
3. Execute the annotated path **at least twice** before stopping: discovery, then APIC capture. Calls and region ordering must agree. Repeated identical inputs alone are not a correctness test; also validate fresh values on the supported path.
4. Export with `"onnx"` / `"onnx-dynamo"` by default. `"pt2"` / `"exported-program"` is the other supported route. Legacy ONNX and JIT do not support these Warp bundles.

Save this runnable example to a Python file so Warp can inspect its kernel source. It assumes setup is complete and CUDA device 0 is available:

```python
import torch
import warp as wp
import leapp
from leapp import annotate, InferenceManager


@wp.kernel
def scale_kernel(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i] = 2.0 * src[i]


wp.init()
leapp.start("warp_graph", global_patching=True)
try:
    for _ in range(2):
        source = wp.ones(8, dtype=wp.float32, device="cuda:0")
        x = annotate.input_tensors("pipeline", {"x": source})
        with annotate.warp_op("pipeline"):
            y = wp.empty_like(x)
            wp.launch(scale_kernel, dim=x.size, inputs=[x, y], device="cuda:0")
        result = torch.tanh(wp.to_torch(y))
        annotate.output_tensors("pipeline", {"result": result}, export_with="onnx")
finally:
    leapp.stop()
leapp.compile_graph(visualize=False, validate=True, strict=True)

manager = InferenceManager("warp_graph/warp_graph.yaml")
inputs = manager.get_mock_input()
fresh = torch.linspace(-1, 1, 8, device=inputs["pipeline/x"].device)
inputs["pipeline/x"] = fresh
actual = manager.run_policy(inputs)["pipeline/result"]
torch.testing.assert_close(actual, torch.tanh(2 * fresh).to(actual.device), rtol=1e-3, atol=1e-5)
```

In the application, use its existing kernels and compare replay with the original Warp pipeline on fresh inputs; the Torch expression above is just a simple oracle for this example.

## Automatic versus explicit regions

With global patching, public Warp calls touching active traced Warp values automatically open/continue a region. `wp.to_torch()`, Warp `.numpy()`, synchronization, and consumption by traced Torch operations close automatic regions. `wp.from_torch()` / `wp.from_numpy()` do not themselves close them. Use automatic regions when they match existing code; an explicit context is optional.

`annotate.warp_op("node")` protects a region until context exit. Only its node name currently controls capture; `inputs`, `outputs`, and extra keyword arguments do not select its boundary. Capture currently uses `cuda:0`, so this context is not a device-selection API. Keep synchronization, host readback, and unrelated CUDA work outside explicit regions.

A Warp operation must not consume active arrays owned by different nodes. Finish producers and rebind inputs in the consumer first. Equivalent full-range `wp.copy` can preserve a finished node's output identity; transformations/partial copies belong inside annotated compute.

## Constraints and process effects

- Boundary arrays need scalar or densely packed homogeneous dtypes. Logical `(B, N)` `wp.vec3` arrays are represented as `(B, N, 3)` float32. Heterogeneous `@wp.struct` arrays are unsupported at these boundaries.
- CPU Warp arrays can execute eagerly but cannot replay in the exported CUDA runner. Successful discovery does not prove export compatibility.
- Private `warp._*` entrypoints are not intercepted. Kernel-language symbols inside `@wp.kernel` belong to the captured kernel, not independent Python patches.
- Data-dependent changes to the sequence of host Warp calls can diverge between passes. This concerns captured call structure, not every conditional inside a kernel body.
- Capture changes process-wide PTX configuration and Torch CUDA caching allocator behavior that can persist after `stop()`. Prefer an isolated export process for a long-lived application; stopping does not restore every CUDA/Warp setting.
- Driver compilation of embedded PTX can add first-load latency; separate it from steady-state replay when measuring performance.

## Troubleshooting and delivery

| Symptom | Check |
|---------|-------|
| No Warp backend / runtime missing | Matching extra, `leapp-build-warp-runtime --status`, initialization before start, global patching, and valid overrides. |
| CUPTI initialization fails | CUPTI and toolkit major versions match; do not randomly upgrade packages. |
| Missing second APIC capture pass | Execute the same annotated path twice in one session. Dry-run still needs a second discovery-path replay even though APIC capture is skipped. |
| More regions / segment diverged | Compare host call sequence and boundaries between passes; reset relevant application state without changing intended semantics. |
| Unsupported export backend | Use ONNX Dynamo or PT2; legacy ONNX is not a Warp fallback. |
| Missing CUDA provider / CPU validation fails | Check GPU runtime/provider and device placement; CPU fallback cannot run Warp. |
| Cross-context / owner-token error | Keep one node per active operation; leave explicit regions before another node starts. |
| Replay mismatch | Inspect inputs, mutations, region boundaries, and logs; compare fresh inputs with the original before changing tolerances. |

Deliver the full LEAPP bundle plus native-runtime setup information. Embedded kernels do not make the model self-contained: the destination needs the adapter for its ONNX/PT2 backend and compatible CUDA/runtime dependencies. Test loading and replay there when available. Do not claim portability from a CPU-only smoke test.

## Only when maintaining LEAPP itself

For internal changes, use the tracing-maintenance skill if available. Traced Torch operations (including views/indexing) must close a pending automatic Warp region **before eager execution or extracting FX argument proxies**. Closing only at FX insertion can preserve stale proxies; view operations may have no CUDA callback. Internal paused work and explicit owner-protected regions retain their lifecycle. In source checkouts, relevant regression coverage includes `tests/unit_tests/test_warp_torch_boundaries.py` and `test_torch_view_consumes_pending_warp_output` in the Warp functional suite. This is maintainer guidance, not an application rewrite requirement.
