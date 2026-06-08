"""Mixed PyTorch + Warp compute graph, captured and run with LEAPP.

You write one ordinary function that mixes PyTorch ops and a Warp kernel. You mark only the
input and output tensors. LEAPP traces everything in between and automatically splits the
region at the ``wp.from_torch`` / ``wp.to_torch`` bridges into native single-kind nodes:

    compute_graph.01_torch   — the PyTorch preprocessing  (exported as ONNX/TorchScript)
    compute_graph.02_warp    — the Warp kernel            (exported as a native APIC .wrp)
    compute_graph.03_torch   — the PyTorch postprocessing (exported as ONNX/TorchScript)

connected by auto-discovered data-flow edges and run through the standard InferenceManager.

Run:
    PYTHONPATH=$PWD python3.12 examples/warp_pytorch_mixed_compute_graph.py

Requires: warp-lang>=1.13 (APIC), torch (CUDA), a CUDA GPU.
"""
import os
import sys
import tempfile

import torch
import warp as wp

import leapp
from leapp import annotate, InferenceManager

DEVICE = "cuda:0"
N = 6  # number of 3-vectors


# ============================================================================
# 1. General setup — the Warp kernel and the PyTorch function
# ============================================================================
# A Warp kernel: normalize each 3-vector on the GPU. This is your existing,
# unmodified Warp code — LEAPP does not require any changes to it.
@wp.kernel
def normalize3(x: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    out[i] = wp.normalize(x[i])


# A PyTorch function: ordinary torch ops, used to pre-process the input.
def pytorch_preprocess(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 0.5


def mixed_compute_graph(obs: torch.Tensor) -> torch.Tensor:
    """The computation we want to capture: PyTorch -> Warp -> PyTorch.

    The SAME function is used both to define the LEAPP graph (when tracing is
    active) and to compute the eager reference (when it is not).
    """
    # --- PyTorch part: preprocess the input ---
    x = pytorch_preprocess(obs)

    # --- Warp part: hand the tensor to Warp, run the kernel, take it back ---
    x_wp = wp.from_torch(x.contiguous().reshape(-1, 3), dtype=wp.vec3f)
    y_wp = wp.zeros(N, dtype=wp.vec3f, device=DEVICE)
    wp.launch(normalize3, dim=N, inputs=[x_wp], outputs=[y_wp], device=DEVICE)
    y = wp.to_torch(y_wp)

    # --- back in PyTorch: postprocess (reshape to [N, 3]) ---
    return y.reshape(N, 3)


def build_and_run(save_path: str) -> int:
    wp.init()
    example_input = torch.randn(N, 3, device=DEVICE, dtype=torch.float32)

    # ========================================================================
    # 2. Set up LEAPP — start tracing
    # ========================================================================
    leapp.start("compute_graph", save_path=save_path)

    # ========================================================================
    # 3. Run the compute graph — mark inputs, run the mixed code, mark outputs
    # ========================================================================
    obs = annotate.input_tensors("compute_graph", {"obs": example_input})
    result = mixed_compute_graph(obs)  # traces PyTorch + Warp automatically
    annotate.output_tensors("compute_graph", {"result": result},
                            export_with="onnx-torchscript")

    # ========================================================================
    # 4. Post-process LEAPP — stop tracing and compile the graph
    # ========================================================================
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True)

    # ========================================================================
    # 5. Run the compiled graph via LEAPP and compare to the original result
    # ========================================================================
    yaml_path = os.path.join(save_path, "compute_graph", "compute_graph.yaml")
    inference = InferenceManager(yaml_path)

    test_input = torch.randn(N, 3, device=DEVICE, dtype=torch.float32)
    in_key = [k for k in inference.inputs if k.endswith("/obs")][0]
    out_key_outputs = inference({in_key: test_input})
    out_key = [k for k in out_key_outputs if k.endswith("/result")][0]
    leapp_result = out_key_outputs[out_key]

    # The "original result": run the same function eagerly, without LEAPP.
    eager_result = mixed_compute_graph(test_input)

    max_abs_err = float((leapp_result - eager_result).abs().max())
    ok = max_abs_err < 1e-4

    print("Mixed compute graph:  PyTorch preprocess -> Warp normalize -> PyTorch reshape")
    print(f"  nodes: compute_graph.01_torch -> compute_graph.02_warp -> compute_graph.03_torch")
    print(f"  max_abs_err (LEAPP vs eager) = {max_abs_err:.2e}")
    print()
    print("PASS — LEAPP graph reproduces the original mixed PyTorch + Warp computation"
          if ok else f"FAIL — max_abs_err {max_abs_err:.2e} exceeds 1e-4")
    return 0 if ok else 1


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="leapp_warp_mixed_") as tmp:
        return build_and_run(tmp)


if __name__ == "__main__":
    sys.exit(main())
