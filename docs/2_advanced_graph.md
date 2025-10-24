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
    annotate.start(name="feedback_example")

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

    annotate.stop()
    annotate.compile_graph()
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

## Node Merging: Optimizing Graph Structure

In many cases, it is advantageous to merge interconnected nodes to simplify the final graph structure. LEAPP can automatically merge nodes to create more efficient graph structures. This optimization combines multiple nodes into single computational units when it's safe to do so.

### Understanding Node Merging Strategies

LEAPP provides different strategies for node merging through the `MergeCfgEnum`:

```python
from leapp import annotate, MergeCfgEnum

annotate.compile_graph(merge_nodes=MergeCfgEnum.NO_MERGE)      # Default: No merging
annotate.compile_graph(merge_nodes=MergeCfgEnum.AUTOMATIC)     # Automatic merging
```

### Automatic Merging: Completely Sequential Nodes

The `AUTOMATIC` strategy merges nodes that are **completely sequential** - meaning one node's outputs connect exclusively to another node's inputs with no branching or external connections.

#### Criteria for Automatic Merging

Two nodes can be merged automatically only if **ALL** of these conditions are met:

1. **Single target per output**: Each output from the source node goes to exactly one target
2. **Complete output consumption**: ALL outputs from the source node connect to the target node
3. **Exclusive input source**: ALL inputs to the target node come from the source node
4. **Same backend**: Both nodes use the same export backend (e.g., both use "torch")

#### Example: Nodes That Can Be Merged

```python
import torch
from leapp import annotate, MergeCfgEnum

@annotate.method(export_with="torch")
def step_one(input_data: torch.Tensor):
    """First step produces one output."""
    result = input_data * 2.0
    return result

@annotate.method(export_with="torch")
def step_two(data: torch.Tensor):
    """Second step consumes ALL outputs from step_one."""
    result = data + 1.0
    return result

annotate.start(name="mergeable_example")

# These two nodes form a perfect sequential chain
intermediate = step_one(torch.tensor([1.0, 2.0, 3.0]))
output = step_two(intermediate)  # step_two ONLY uses step_one's output

annotate.stop()
annotate.compile_graph(merge_nodes=MergeCfgEnum.AUTOMATIC)
# Result: step_one and step_two are merged into a single node
```

### When Nodes CANNOT Be Merged

Nodes with outputs that have **multiple targets** or inputs from **multiple sources** cannot be automatically merged.

#### Example: Branching Outputs (Cannot Merge)

```python
import torch
from leapp import annotate, MergeCfgEnum

@annotate.method(export_with="torch")
def shared_processing(input_data: torch.Tensor):
    """This node's output goes to TWO different targets."""
    return input_data * 2.0

@annotate.method(export_with="torch")
def branch_a(data: torch.Tensor):
    return data + 1.0

@annotate.method(export_with="torch")
def branch_b(data: torch.Tensor):
    return data - 1.0

annotate.start(name="branching_example")

shared = shared_processing(torch.tensor([1.0, 2.0, 3.0]))

# shared_processing's output goes to MULTIPLE targets
result_a = branch_a(shared)
result_b = branch_b(shared)

annotate.stop()
annotate.compile_graph(merge_nodes=MergeCfgEnum.AUTOMATIC)
# Result: No merging occurs - shared_processing has multiple output targets
```

**Why this is ignored by automerge:**
- `shared_processing` output goes to both `branch_a` AND `branch_b`
- This violates the "single target per output" rule
- The graph must preserve this branching structure

#### Example: Multiple Input Sources (Cannot Merge)

```python
import torch
from leapp import annotate, MergeCfgEnum

@annotate.method(export_with="torch")
def source_a(input_a: torch.Tensor):
    return input_a * 2.0

@annotate.method(export_with="torch")
def source_b(input_b: torch.Tensor):
    return input_b * 3.0

@annotate.method(export_with="torch")
def combine(data_a: torch.Tensor, data_b: torch.Tensor):
    """This node receives inputs from TWO different sources."""
    return data_a + data_b

annotate.start(name="multiple_sources_example")

out_a = source_a(torch.tensor([1.0, 2.0]))
out_b = source_b(torch.tensor([3.0, 4.0]))

# combine receives inputs from MULTIPLE sources
result = combine(out_a, out_b)

annotate.stop()
annotate.compile_graph(merge_nodes=MergeCfgEnum.AUTOMATIC)
# Result: No merging occurs - combine has multiple input sources
```

**Why this is ignored by automerge:**
- `combine` receives inputs from both `source_a` AND `source_b`
- Neither source node can merge with `combine` because `combine` has external inputs
- The graph must preserve these multiple data paths

### Visualizing Merged Nodes

When nodes are merged, the resulting node is named by combining the original node names:

```python
# Original nodes: "preprocessing", "normalization", "scaling"
# Merged node name: "preprocessing-normalization-scaling"
```

The generated YAML file will show this merged node as a single computational unit with:
- Combined inputs from the first node in the chain
- Combined outputs from the last node in the chain
- Internal execution order preserved

### Benefits of Node Merging

**Performance advantages:**
- Reduced overhead from node-to-node data transfers
- More efficient memory usage
- Simplified deployment graph
- Potential for better optimization by downstream frameworks

**When to use AUTOMATIC merging:**
- Long sequential chains of simple operations
- When you want to minimize the number of deployed models
- When intermediate outputs aren't needed externally

**When to use NO_MERGE (default):**
- When you need access to intermediate outputs
- For better debugging and monitoring of individual stages
- When nodes use different optimization settings
- When graph structure clarity is important

## Summary

Understanding graph-level operations helps you build more sophisticated computational graphs:

- **Feedback connections** capture cyclic behavior by running your graph multiple times during tracing
- **Automatic node merging** optimizes completely sequential chains where all outputs go to a single target
- Nodes with **branching outputs** or **multiple input sources** cannot be automatically merged
- Choose merging strategies based on your performance needs and debugging requirements

These advanced features give you fine-grained control over how LEAPP interprets and optimizes your computational graphs for deployment.

