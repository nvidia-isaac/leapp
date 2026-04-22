# Advanced Graph Operations

This guide explores advanced graph-level operations in LEAPP, focusing on cycles, feedback connections, and tracing continuity across graph edges.

## Cycle Detection and Feedback Connections

LEAPP automatically detects cycles in your computational graph through a concept called **feedback connections**. A feedback connection occurs when data flows from a later node back to an earlier node, creating a loop in the graph.

### How LEAPP Detects Cycles

LEAPP assigns each node an index when that node completes its initial trace. In practice, this is the node completion/finalization order, not the order a node name first appeared. When analyzing connections:
- **Normal connections**: Data flows from a lower-indexed node to a higher-indexed node (forward flow)
- **Feedback connections**: Data flows from a higher-indexed node back to a lower-indexed node (backward flow)

Feedback connections are visualized in red in the graph visualization, while normal connections appear in black.
### Capturing Feedback Behavior

For graph-level feedback that is inferred from re-entry across nodes, you need to **run your graph multiple times** within the same tracing session. This allows LEAPP to observe data flowing from a later node back into an earlier node on a later iteration.

This is different from explicit state APIs such as `annotate.state_tensors()` / `annotate.update_state()` and `annotate.module()`, which can produce feedback metadata in a single trace.

#### Example: Processing with Feedback Loop

A complete runnable version of this example lives in `examples/feedback_example.py`.

```python
import torch
import leapp
from leapp import annotate

def mix_with_feedback(data: torch.Tensor, feedback: torch.Tensor) -> torch.Tensor:
    centered = data - 0.5
    return torch.tanh(centered + 0.25 * feedback)

def blend_feedback(hidden: torch.Tensor, previous_feedback: torch.Tensor) -> torch.Tensor:
    return 0.8 * previous_feedback + 0.2 * hidden

def main():
    leapp.start(name="sample_feedback_graph")

    policy_memory = torch.tensor([0.0])

    for _ in range(2):  # needed for inferred cross-node feedback detection
        policy_inputs = annotate.input_tensors("policy_step", {
            "observation_scalar": torch.tensor([1.0]),
            "policy_memory_in": policy_memory,
        })
        policy_context = mix_with_feedback(policy_inputs[0], policy_inputs[1])
        control_action = torch.clamp(policy_context * 2.0, min=-1.0, max=1.0)
        annotate.output_tensors(
            "policy_step",
            {"policy_context": policy_context, "control_action": control_action},
            export_with="jit",
        )

        feedback_inputs = annotate.input_tensors("feedback_update", {
            "policy_context": policy_context,
            "policy_memory_prev": policy_memory,
        })
        policy_memory = blend_feedback(feedback_inputs[0], feedback_inputs[1])
        annotate.output_tensors(
            "feedback_update",
            {"policy_memory_out": policy_memory},
            export_with="jit",
        )

    leapp.stop()
    leapp.compile_graph()
```

In this example, `policy_memory_out` flows from `feedback_update` (completed later) back into the `policy_memory_in` input of `policy_step` on the next iteration, creating a feedback connection. The standalone script exports to `sample_feedback_graph/`, which is ignored by git via the repository's `sample_*` rule.

![Feedback Example Graph](images/feedback_example_graph.png)

Inspect detected feedback connection details by verifying the `feedback_flow` field under pipeline.

```yaml
    feedback_flow:
        feedback_update/policy_memory_out: [policy_step/policy_memory_in, feedback_update/policy_memory_prev]
```

### Important Considerations

**⚠️ Minimum Two Iterations Required**

You must run the loop **at least twice** for LEAPP to detect inferred cross-node feedback connections:
- **First iteration**: LEAPP traces all nodes and establishes their direct connections
- **Second iteration**: LEAPP observes data flowing back to earlier nodes, confirming the feedback connection

Explicit feedback declared with `state_tensors()` / `update_state()` or detected via `annotate.module()` does not require a second iteration.

**⚠️ feedback i/o names are not reconciled**
- LEAPP currently reconciles names only for forward internal connections in `data_flow`. Feedback connections in `feedback_flow` are emitted as-is, so downstream frameworks should not assume the source and target port names will match.

## Maintaining Tracing with `mirror_leapp_tags`

When working with tensor data in LEAPP, proper tracing requires that LEAPP can track how data flows through your computational graph. However, sometimes you need to duplicate tensor data without using PyTorch's standard `clone()` or `detach()` methods - for example, when using in-place assignment operations like `tensor[:] = other_tensor` or if the tensor was temporarily converted to a different datatype `tensor=np.array(tagged_tensor)`.

In these cases, LEAPP's internal tags that track data provenance won't automatically transfer to the copied data. The `annotate.mirror_leapp_tags()` function solves this problem by explicitly transferring tracing tags from a source tensor to a target tensor.

### When to Use `mirror_leapp_tags`

Use `mirror_leapp_tags` when you:
- Copy tensor data using in-place operations (e.g., `self._prev_action[:] = self._action`)
- Need to maintain tracing continuity across manual data duplication
- Want to ensure LEAPP recognizes that two tensors contain the same logical data

### How It Works

The `mirror_leapp_tags()` function performs two critical operations:

1. **Verifies Data Equivalence**: First checks that the source and target tensors contain exactly the same values
2. **Transfers Tags**: If verification passes, copies all LEAPP internal tracking tags from source to target

If the data doesn't match, LEAPP logs an error and raises instead of copying incorrect tracing metadata.

### Example: Using Preallocated Buffer

A common use case is duplicating data into a preallocated buffer:

```python
import torch
import leapp
from leapp import annotate

class DataProcessor:
    def __init__(self):
        self._buffer = torch.zeros(10)
    
    @annotate.method(export_with="jit")
    def process(self, input_data: torch.Tensor):
        # input_data comes from an upstream node and is tagged to trace graph connections
        # Copy data using in-place assignment
        self._buffer[:] = input_data
        
        # Mirror LEAPP tags to maintain proper tracing
        annotate.mirror_leapp_tags(input_data, self._buffer)
        
        # Now use the buffer
        result = self._buffer * 2.0
        return result
```

### API Signature

```python
annotate.mirror_leapp_tags(source, target)
```

**Parameters:**
- `source`: The tensor containing the original data and LEAPP tags
- `target`: The tensor that should receive the tags (must have identical values to source)

### Important Considerations

**⚠️ Data Must Match Exactly**

The function raises if the values in `source` and `target` differ:

```python
source = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.0, 2.0, 4.0])  # Different value!

# This raises because the tensor values differ:
annotate.mirror_leapp_tags(source, target)
```

**⚠️ Only Works During Tracing**

The function only has an effect when LEAPP is actively tracing. Outside of `leapp.start()` / `leapp.stop()` blocks, it will safely no-op.

### When to not use `mirror_leapp_tags`

You **don't** need to use this function when:

- Using the PyTorch operations such as `clone()` and `detach()`, or assignment that create new tensors
- LEAPP can automatically track data flow through normal operations
- You're not manually duplicating data with in-place operations

```python
# These don't need mirror_leapp_tags:
new_tensor = old_tensor.clone()  # LEAPP tracks this automatically
detached = old_tensor.detach()   # LEAPP tracks this automatically
copied = old_tensor              # Simple reference, no duplication
new_tensor = [old_tensor]        # variable structure change but underlying tensor is not changed
```

**LEAPP tags are automatically preserved through these operations:**
- `.clone()` - Creates a copy with the same tags
- `.detach()` - Detaches from computation graph, preserves tags
- `.contiguous()` - Returns contiguous tensor, preserves tags
- `.cpu()` / `.cuda()` - Device transfers, preserves tags

You **shouldn't** use this function when:

- computation is performed such that the value of source and target are going to be **different**
```python
# These should be tracked as a node
new_tensor = old_tensor+10       # computation performed
new_tensor[:5] = old_tensor      # a subsection of new_tensor is replaced with old_tensor
```

## IO Reconciliation

IO reconciliation happens during graph construction, when LEAPP tries to connect node outputs to downstream node inputs even if the names do not already match.

This is useful as a fallback, but it can also create ambiguous or conflicting graph interfaces. As a rule, it is better to keep names consistent yourself and treat reconciliation as a last resort.

### Practical Guidance

- Keep output names and downstream input names consistent when you can
- Prefer explicit names with `input_tensors()` / `output_tensors()` for graph boundaries
- If `compile_graph()` warns that names were changed, inspect the generated YAML and verify the final interface names

### Common Failure Pattern

This kind of graph is risky:

```python
@annotate.method()
def detect(x):
    detections = x + 1
    return detections

@annotate.method()
def consume(detections, x):
    return detections + x
```

If a graph has multiple candidate names that LEAPP tries to reconcile into the same slot, `compile_graph()` can fail with an unrecoverable naming conflict.

The simplest fix is usually to rename the node inputs or outputs so the intended graph wiring is already explicit before reconciliation runs.

## Summary

Understanding graph-level operations helps you build more sophisticated computational graphs:

- **Feedback connections** capture cyclic behavior by running your graph multiple times during tracing
- **Mirror LEAPP tags** (`annotate.mirror_leapp_tags()`) maintains proper tracing when duplicating tensor data with in-place operations
These advanced features give you fine-grained control over how LEAPP interprets and optimizes your computational graphs for deployment.

