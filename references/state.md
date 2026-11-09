# State and episode lifecycle

Read this for recurrent policies, history windows, or episode resets. State controls values carried between calls; semantic metadata describes tensor meaning and is optional independently of state. See [deployment semantics](semantics.md) only when a consumer needs that metadata.

Online references: [state](https://nvidia-isaac.github.io/leapp/guides/graph.html), [API](https://nvidia-isaac.github.io/leapp/api/index.html), [runtime](https://nvidia-isaac.github.io/leapp/leapp_runtime.html).

## Explicit state across an episode

`state_tensors` takes a flat dict of named Torch tensors inside an existing node. Nested state is unsupported; expose tensor leaves under distinct stable names. `update_state` uses those same names and preserves shape and dtype. An unupdated state remains an ordinary input. Constants belong in `annotate.register_buffer`, not feedback.

This runnable example demonstrates state ownership and sequence parity without modifying the original step function:

```python
import torch
import leapp
from leapp import annotate, InferenceManager


def original_step(obs, hidden):
    next_hidden = torch.tanh(obs + 0.5 * hidden)
    return 2.0 * next_hidden, next_hidden


samples = [torch.randn(1, 3) for _ in range(5)]
hidden = torch.zeros(1, 3)
expected = []
for obs in samples:
    action, hidden = original_step(obs, hidden)
    expected.append(action.clone())

hidden = torch.zeros(1, 3)
leapp.start("stateful_graph", max_cached_io=5)
try:
    for obs, reference in zip(samples, expected):
        x = annotate.input_tensors("pipeline", {"obs": obs})
        h = annotate.state_tensors("pipeline", {"hidden": hidden})
        action, next_hidden = original_step(x, h)
        annotate.update_state("pipeline", {"hidden": next_hidden})
        annotate.output_tensors("pipeline", {"action": action}, export_with="onnx")
        torch.testing.assert_close(action, reference)
        hidden = next_hidden
finally:
    leapp.stop()
leapp.compile_graph(visualize=False, validate=True, strict=True)

# Each new manager starts a new episode with the saved feedback initial values.
for episode in range(2):
    manager = InferenceManager("stateful_graph/stateful_graph.yaml")
    template = manager.get_mock_input()["pipeline/obs"]
    hidden = torch.zeros_like(template)
    for step in range(4):
        fresh = torch.randn_like(template) + episode * 0.2
        expected_action, hidden = original_step(fresh, hidden)
        actual = manager.run_policy({"pipeline/obs": fresh})["pipeline/action"]
        torch.testing.assert_close(actual, expected_action.to(actual.device), rtol=1e-3, atol=1e-5)
```

For an actual application, preserve its initial state, reset rules, and mutation semantics. Clone inputs or use separate model instances when baseline and traced execution could mutate shared data. Do not reset every trace iteration if deployment carries state between ticks. Compare first-tick and later-tick behavior on fresh sequences.

## Existing module state

After opening a node, register the module before executing it:

```python
annotate.module("pipeline", model, buffer_names=["h_state", "encoder.running_mean"])
```

Replace the example names with real registered buffers; omit `buffer_names` to track all. Reassigned buffers become feedback; read-only buffers are constants. Plain attributes are not captured, and an in-place update such as `self.h.copy_(next_h)` of a tracked buffer raises an error when LEAPP detects it (the graph guide gives the exact rule), so leave that buffer out of `buffer_names` and declare it with `state_tensors()` and `update_state()`. Use explicit state annotations where appropriate rather than rewriting the model or assuming a pure function patch captures mutation.

Explicit state/module feedback can be declared in one trace. **Inferred cross-node feedback** needs at least two graph iterations in the same session. Validate `pipeline.feedback_flow` and `pipeline.initial_values`; carry the referenced safetensors file with the bundle. Use `manager.feedback_inputs` and `set_input_value(node, port, value)` to inspect/seed feedback. A fresh manager is the portable episode-reset recipe; if using an installed `reset()` method, verify its contract—buffer reset does not necessarily reset private state inside model objects.
