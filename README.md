<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LEAPP — Lightweight Export Annotations for Policy Pipelines

**Export the policy code you already run — preprocessing, model inference, and action processing together.**

LEAPP traces supported PyTorch and NumPy computation across existing functions and objects. Export the complete path as **one ONNX, TorchScript, or PT2 model**, or choose separate stages when useful.

- **[State](https://nvidia-isaac.github.io/leapp/guides/graph.html):** export explicit feedback or track reassigned registered PyTorch buffers.
- **[Deployment metadata](https://nvidia-isaac.github.io/leapp/semantics/usage.html):** attach tensor roles, joint names, and timing information.
- **[FunctionPatch](https://nvidia-isaac.github.io/leapp/guides/patching.html):** supply traceable replacements for unsupported module-level calls while keeping existing callers.
- **[Validation](https://nvidia-isaac.github.io/leapp/guides/debugging.html):** compare exported outputs against captured samples and replay bundles in Python.
- **[Warp](https://nvidia-isaac.github.io/leapp/tensor_libraries/warp.html):** capture kernel regions with an additional runtime adapter.

**Isaac Lab** [adds annotations automatically](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/policy_deployment/05_leapp/exporting_policies_with_leapp.html) for supported manager-based environments. Custom pipelines use the annotations directly.

## Quick start

```bash
pip install leapp
```

Requires Python 3.10+ and PyTorch 2.6.0+.

This example exports NumPy preprocessing, a small PyTorch policy, and action scaling and clipping together:

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

Produces **`robot_policy/control.onnx`**, a YAML interface specification, and a graph PNG on Python 3.11+. This stateless model runs directly in ONNX Runtime; [InferenceManager](https://nvidia-isaac.github.io/leapp/leapp_runtime.html) loads the bundle for Python replay. See [Isaac ROS Deploy](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_deploy/index.html) for ROS 2 deployment.

## Capture scope

LEAPP captures the executed tensor path. Data-dependent Python branches are not preserved dynamically, and unsupported operations or scalar extraction can freeze values. Check tracing warnings and validate on fresh inputs. See [PyTorch](https://nvidia-isaac.github.io/leapp/tensor_libraries/torch_limitations.html) and [NumPy](https://nvidia-isaac.github.io/leapp/tensor_libraries/limitations.html) limitations.

[Getting started](https://nvidia-isaac.github.io/leapp/getting_started.html) · [Export options](https://nvidia-isaac.github.io/leapp/guides/export.html) · [API reference](https://nvidia-isaac.github.io/leapp/api/index.html)

## License

Copyright (c) 2026, NVIDIA Corporation & affiliates. Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [thirdparty.txt](thirdparty.txt).
