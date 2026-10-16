<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LEAPP — Lightweight Export Annotations for Policy Pipelines

**Export the policy code you already run — preprocessing, model inference, and action processing together.**

Deploying a trained policy takes more than its weights. Observation normalization, feature construction, recurrent state, and action transforms must travel with it. LEAPP captures supported tensor computation from your existing Python code and exports it to ONNX, TorchScript, or PyTorch ExportedProgram (`.pt2`).

Mark where inputs enter and outputs leave, then run your code. The captured computation can span helper functions, class methods, and supported PyTorch and NumPy operations. **Export the whole path as one model**, or choose separate models when your deployment benefits from independent stages. You do not need to organize your code around an RL environment interface or move the computation into a single `forward()` method.

**Isaac Lab** provides a [native LEAPP integration](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/policy_deployment/05_leapp/exporting_policies_with_leapp.html) that exports observation processing, the policy, action processing, and recurrent state. For supported manager-based environments, Isaac Lab adds the annotations automatically. For custom research pipelines, use the same annotations directly in your own code.

## What you can export

- **Existing computation across functions and objects.** Collect inputs at multiple call sites and capture the path to the final output in one model. [Annotation patterns](https://nvidia-isaac.github.io/leapp/guides/nodes.html)
- **Mixed tensor code.** Trace supported NumPy preprocessing through conversions into PyTorch inference. Experimental NVIDIA Warp support also captures kernel regions into ONNX or PT2 graphs, with an additional runtime adapter. [NumPy](https://nvidia-isaac.github.io/leapp/tensor_libraries/numpy.html) · [Warp](https://nvidia-isaac.github.io/leapp/tensor_libraries/warp.html)
- **Stateful policies.** Declare feedback state explicitly, or track reassigned registered PyTorch buffers with `annotate.module()`. Export the state inputs, outputs, and initial values needed for replay. [State capture](https://nvidia-isaac.github.io/leapp/guides/graph.html)
- **Your choice of model format and boundaries.** Use one model for the complete computation, or export stages independently with different backends and include prebuilt model artifacts. [Export options](https://nvidia-isaac.github.io/leapp/guides/export.html)
- **A deployment interface with meaning.** Attach and preserve tensor names, joint names, state and command roles, and temporal metadata alongside the models so downstream systems can bind their inputs and outputs. [Tensor semantics](https://nvidia-isaac.github.io/leapp/semantics/usage.html)

Export validation compares the compiled model outputs with captured Python outputs. The generated YAML describes the interface and any state feedback or connections; a graph visualization makes the capture inspectable.

## Installation

```bash
pip install leapp
```

LEAPP requires Python 3.10+ and PyTorch 2.6.0+.

## Export a complete pipeline into one model

This runnable example keeps preprocessing in a NumPy helper, inference in a PyTorch module, and action processing at the call site. Replace the small example network with your trained policy.

```python
import numpy as np
import torch
import leapp
from leapp import annotate

def make_observation(position, velocity):
    return np.concatenate([np.clip(position, -1.0, 1.0), velocity * 0.1])

policy = torch.nn.Sequential(
    torch.nn.Linear(12, 32), torch.nn.Tanh(), torch.nn.Linear(32, 6),
).eval()

leapp.start(name="robot_policy")
position, velocity = annotate.input_tensors("control", {
    "position": np.zeros(6, dtype=np.float32),
    "velocity": np.ones(6, dtype=np.float32),
})

with torch.no_grad():
    observation = torch.from_numpy(make_observation(position, velocity))
    targets = torch.clamp(policy(observation) * 0.25, -1.0, 1.0)

annotate.output_tensors("control", {"targets": targets}, export_with="onnx")

leapp.stop()
leapp.compile_graph(validate=True)
```

The `robot_policy/` directory contains **one `control.onnx` model** covering NumPy preprocessing, network weights and inference, and action scaling and clipping. It also contains `robot_policy.yaml` describing the interface and, on Python 3.11+, a graph PNG. On Python 3.10, export completes without the PNG.

For this stateless example, an ONNX Runtime session can run `control.onnx` directly. LEAPP's Python `InferenceManager` can instead load the YAML bundle, including feedback state and model connections when present.

For your own pipeline, check tracing warnings and compare exported outputs on varied inputs: agreement on one captured sample can hide values frozen during tracing. See [validation guidance](https://nvidia-isaac.github.io/leapp/guides/debugging.html).

## From export to deployment

Use [InferenceManager](https://nvidia-isaac.github.io/leapp/leapp_runtime.html) to replay and inspect a bundle in Python. For ROS 2 deployment, [Isaac ROS Deploy](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_deploy/index.html) maps policy terms to topics or `ros2_control`. Other applications can load the exported models with the corresponding backend runtime and implement the declared I/O and state contract. See the [ecosystem guide](https://nvidia-isaac.github.io/leapp/ecosystem.html).

## Capture scope

LEAPP captures supported tensor computation along the executed path. Data-dependent Python branches and loops are not exported as dynamic control flow; extracting Python scalars can freeze values at trace time. Operator coverage depends on the tensor library and export backend. Warp support is experimental and requires additional installation and runtime support. See the [PyTorch](https://nvidia-isaac.github.io/leapp/tensor_libraries/torch_limitations.html), [NumPy](https://nvidia-isaac.github.io/leapp/tensor_libraries/limitations.html), and [Warp](https://nvidia-isaac.github.io/leapp/tensor_libraries/warp_limitations.html) limitations before integrating a new pipeline.

## Documentation

Full guides and API reference live at **[https://nvidia-isaac.github.io/leapp/](https://nvidia-isaac.github.io/leapp/)**:

- [Getting started](https://nvidia-isaac.github.io/leapp/getting_started.html) — install, first pipeline, generated output
- [Annotation patterns](https://nvidia-isaac.github.io/leapp/guides/nodes.html) — capture across call sites, function decorators, and nested data
- [API reference](https://nvidia-isaac.github.io/leapp/api/index.html)

## License

Copyright (c) 2026, NVIDIA Corporation & affiliates. Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [thirdparty.txt](thirdparty.txt).
