# Advanced Graph Operations

This guide explores advanced graph-level operations in LEAPP, focusing on how LEAPP handles complex graph structures including cycles and optimization through node merging.

## Cycle Detection and Feedback Connections

LEAPP automatically detects cycles in your computational graph through a concept called **feedback connections**. A feedback connection occurs when data flows from a later node back to an earlier node, creating a loop in the graph.

### How LEAPP Detects Cycles

LEAPP assigns each node an index based on the order it's first traced. When analyzing connections:
- **Normal connections**: Data flows from a lower-indexed node to a higher-indexed node (forward flow)
- **Feedback connections**: Data flows from a higher-indexed node back to a lower-indexed node (backward flow)

Feedback connections are visualized in red in the graph visualization, while normal connections appear in black.
\
### Capturing Feedback Behavior

To properly capture feedback connections, you need to **run your graph multiple times** within the same tracing session. This allows LEAPP to observe the data flowing back to earlier nodes.

#### Example: Processing with Feedback Loop

```python
@annotate.method()
def process_input(data: torch.Tensor, feedback: torch.Tensor):
    """Combines current input with feedback from previous iteration."""
    return data + feedback


@annotate.method()
def transform_data(data: torch.Tensor):
    """Applies transformation to the data."""
    return data * 2.0


@annotate.method()
def generate_feedback(data: torch.Tensor, previous_feedback: torch.Tensor):
    """Generates feedback for next iteration."""
    return data + previous_feedback * 0.1


@annotate.method()
def final_output(data: torch.Tensor):
    if torch.sum(data > 10):
        retval = torch.tensor([False])
    else:
        retval = torch.tensor([True])
    return retval


def main():
    leapp.start(name="feedback_example")

    # Initialize feedback to zero
    feedback_value = torch.tensor([0])
    final_output_value = torch.tensor([False])

    # Run the graph multiple times to capture the feedback loop
    for i in range(2):  # Minimum 2 iterations required
        # Process input with feedback from previous iteration
        processed = process_input(torch.tensor([1]), feedback_value)

        # Transform the data
        transformed = transform_data(processed)

        # Generate feedback for next iteration
        feedback_value = generate_feedback(transformed, feedback_value)

        final_output_value = final_output(feedback_value)

    leapp.stop()
    leapp.compile_graph()
```

In this example, `feedback_value` flows from `generate_feedback` (traced later) back to `process_input` (traced earlier), creating a feedback connection. The detected graph will look like the following where red curved lines represent feedback connections. 

![Sample Robot Pipeline Graph](images/feedback_example.png)

Inspect detected feedback connection details by verifying the `feedback_flow` field under pipeline.

```yaml
    feedback_flow:
        generate_feedback/data: [process_input/feedback, generate_feedback/previous_feedback]
```

### Important Considerations

**⚠️ Minimum Two Iterations Required**

You must run the loop **at least twice** for LEAPP to detect feedback connections:
- **First iteration**: LEAPP traces all nodes and establishes their direct connections
- **Second iteration**: LEAPP observes data flowing back to earlier nodes, confirming the feedback connection

**⚠️ name matching not guaranteed**
- When feedback loops are established, LEAPP does not attempt to reconcile the i/o names. The downstream framework would need to take this into consideration when reconnecting feedback loops.

## Maintaining Tracing with `mirror_leapp_tags`

When working with tensor data in LEAPP, proper tracing requires that LEAPP can track how data flows through your computational graph. However, sometimes you need to duplicate tensor data without using PyTorch's standard `clone()` or `detach()` methods - for example, when using in-place assignment operations like `tensor[:] = other_tensor`.

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

If the data doesn't match, it logs an error and does nothing to prevent incorrect tracing.

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

The function will log an error and do nothing if the values in source and target differ:

```python
source = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.0, 2.0, 4.0])  # Different value!

# This will log an error and do nothing:
annotate.mirror_leapp_tags(source, target)  # Error logged: source and target do not match
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

## Node Merging

Node merging has been removed from LEAPP and is no longer part of `compile_graph()`.

## Summary

Understanding graph-level operations helps you build more sophisticated computational graphs:

- **Feedback connections** capture cyclic behavior by running your graph multiple times during tracing
- **Mirror LEAPP tags** (`annotate.mirror_leapp_tags()`) maintains proper tracing when duplicating tensor data with in-place operations
These advanced features give you fine-grained control over how LEAPP interprets and optimizes your computational graphs for deployment.

