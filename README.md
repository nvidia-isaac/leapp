<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LEAPP: Lightweight Export Annotations for Policy Pipelines

[Full LEAPP docs](https://nvidia-isaac.github.io/leapp/)

**Export the policy code you already run: preprocessing, model inference, and action processing together.**

LEAPP traces supported PyTorch and NumPy computation across existing functions and objects. Export the complete path as **one ONNX, TorchScript, or PT2 model**, or choose separate stages when useful.

Mark where inputs enter and outputs leave, then run your application. Inputs can enter at different call sites; helper functions and class boundaries do not force separate models. No prescribed RL environment interface is required. [See annotation patterns](https://nvidia-isaac.github.io/leapp/guides/nodes.html).

- **Keep processing with the policy.** Capture supported NumPy preprocessing, conversions into PyTorch, and action transforms together. [Warp kernel regions](https://nvidia-isaac.github.io/leapp/tensor_libraries/warp.html) can also be captured into ONNX or PT2 with an additional runtime adapter.
- **Adapt unsupported calls with [FunctionPatch](https://nvidia-isaac.github.io/leapp/guides/patching.html).** Register a faithful replacement built from supported operations; existing module-level callers can stay in place.
- **Describe the deployment interface.** Attach tensor roles, joint names, and timing information with [semantic metadata](https://nvidia-isaac.github.io/leapp/semantics/usage.html), so downstream applications can interpret model inputs and outputs.
- **Inspect and validate the export.** Compare exported outputs, including state updates, against cached samples from consecutive Python iterations. Configure sample coverage, inspect the graph, and use [sequential replay](https://nvidia-isaac.github.io/leapp/guides/debugging.html#stateful-transitions-and-sequential-replay) to check accumulated drift and runtime feedback.

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

Produces **`robot_policy/control.onnx`**, a YAML interface specification, and a graph PNG on Python 3.11+. This stateless model runs directly in ONNX Runtime; [InferenceManager](https://nvidia-isaac.github.io/leapp/leapp_runtime.html) loads the bundle for Python replay. Other applications can consume the exported artifacts through their chosen backend runtime.

## Capture scope

LEAPP captures the executed tensor path. Data-dependent Python branches are not preserved dynamically, and unsupported operations or scalar extraction can freeze values. Check tracing warnings and validate on fresh inputs. See [PyTorch](https://nvidia-isaac.github.io/leapp/tensor_libraries/torch_limitations.html) and [NumPy](https://nvidia-isaac.github.io/leapp/tensor_libraries/limitations.html) limitations.

[Getting started](https://nvidia-isaac.github.io/leapp/getting_started.html) · [Export options](https://nvidia-isaac.github.io/leapp/guides/export.html) · [API reference](https://nvidia-isaac.github.io/leapp/api/index.html)

## License

Copyright (c) 2026, NVIDIA Corporation & affiliates. Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [thirdparty.txt](thirdparty.txt).
