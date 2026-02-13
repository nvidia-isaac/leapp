# LEAPP API Reference

This document provides a comprehensive reference for LEAPP's core API methods. LEAPP uses a global `annotate` instance that provides methods for tracing and exporting computational graphs.

## Quick Reference

```python
from leapp import annotate, MergeCfgEnum

# Core workflow
annotate.start(name="my_graph", save_path=".", verbose=False)
# ... trace your code using @annotate.method(), with annotate.block(), 
#     or annotate.input_tensors() / annotate.output_tensors()
annotate.stop()
annotate.compile_graph(visualize=True, merge_nodes=MergeCfgEnum.NO_MERGE)
```

---

## `annotate.start()`

Initialize and start LEAPP graph interpretation.

### Signature

```python
annotate.start(name: str, save_path: str = ".", verbose: bool = False)
```

### Parameters

- **`name`** (str, required): The name of the graph to be created. This will be used as the directory name where graph artifacts are saved.
- **`save_path`** (str, optional): The base directory path where the graph directory will be created. Defaults to `"."` (current directory).
- **`verbose`** (bool, optional): If `True`, enables verbose logging output. Defaults to `False`.

### Behavior

- Creates the save directory `{save_path}/{name}/` if it doesn't exist
- Configures logging based on the `verbose` flag. If set to true, logs will also print to console. Logs can always be accessed at `{save_path}/{name}/log*`
- Enables graph interpretation mode for tracing
- If called while interpretation is already active, it will reset the graph (this is discouraged)

### Notes

- Must be called before any `@annotate.method()` decorators or `annotate.block()` context managers are used
- Creates a directory structure: `{save_path}/{name}/`
- All traced nodes and outputs will be saved in this directory

---

## `annotate.stop()`

Stop LEAPP graph interpretation and disable tracing.

### Signature

```python
annotate.stop()
```

### Parameters

None

### Behavior

- Disables graph interpretation mode that was enabled by `start()`
- Performs safety checks to ensure no active tracing is in progress
- Must be called after all tracing operations are complete

### Notes

- Should only be called after `start()` has been called
- Ensure all active tracing operations (decorated functions or blocks) have completed before calling `stop()`
- Must be called before `compile_graph()`

---

## `@annotate.method()`

Create a decorator for tracing functions/methods in the computational graph.

### Signature

```python
@annotate.method(**params)
def your_function(...):
    ...
```

### Parameters

All parameters are optional keyword arguments (`**params`):

- **`node_name`** (str): Custom name for the node. If not provided, uses the function's name.
- **`export_with`** (str): Backend to use for exporting the model (e.g., `"torch"`, `"onnx"`).
- **`backend_params`** (dict): Parameters specific to the export backend.
- **`inputs`** (list[str]): Input specifications for the node.
- **`outputs`** (list[str]): Output specifications for the node.
- **`environment_constants`** (list[str]): External variables to capture as constants. Two main use cases: (1) capturing external dependencies like models or configs, (2) freezing variables that change over time (e.g., loop counters) to their value at node creation. See [Advanced Node Operations](1_advanced_nodes.md#environment-constants-referencing-external-data) for details.
- **`register_buffers`** (list[str]): Buffers to register with the model (for mutable state).
- **`enable_fp16`** (bool): Enable FP16 precision mode.
- **`enable_cuda_graphs`** (bool): Enable CUDA graphs optimization.

### Behavior

- Wraps the function to trace its execution when called
- Captures function inputs, outputs, and execution details
- Creates a node in the LEAPP computational graph
- If graph interpretation is disabled, the function executes normally without tracing

### Notes

- Graph interpretation must be enabled via `start()` before decorated functions are called
- Functions decorated with `method()` should not contain nested `block()` or `method()` annotations
- Input and output names are automatically derived from function parameters and return values
- Class member variables (`self.*`) are automatically available as constants in class methods

---

## `annotate.block()`

Create a context manager for tracing a block of code in the computational graph.

### Signature

```python
with annotate.block(node_name: str, **kwargs):
    # code to trace
```

### Parameters

- **`node_name`** (str, required): The unique name to identify this node in the computational graph.
- **`**kwargs`**: Additional parameters (same as `@annotate.method()`)
  - `export_with`: Backend for exporting
  - `backend_params`: Backend-specific parameters
  - `inputs`: Input specifications (list of variable names)
  - `outputs`: Output specifications (list of variable names)
  - `environment_constants`: External variables to capture as constants (also used to freeze changing variables like loop counters)
  - `register_buffers`: Buffers for mutable state
  - `enable_fp16`: Enable FP16 precision
  - `enable_cuda_graphs`: Enable CUDA graphs

### Behavior

- Traces a specific block of code when used with a `with` statement
- Captures inputs, outputs, and execution details of the code block
- Creates a node in the LEAPP computational graph
- Must declare input and output variable names explicitly

### Notes

- Must be used with a `with` statement to demarcate the code block
- Graph interpretation must be enabled via `start()` before using this method
- Input and output variable names must be explicitly declared
- Variable names in `inputs` and `outputs` must match actual Python variable names in scope
- The traced code block should not contain nested `block()` or `method()` annotations

---

## `annotate.input_tensors()`

Create traced tensor inputs for programmatic node definition. Returns TracedTensor objects that record all subsequent operations.

### Signature

```python
traced_tensors = annotate.input_tensors(node_name: str, tensors: dict)
```

### Parameters

- **`node_name`** (str, required): The unique name to identify this node in the computational graph.

- **`tensors`** (dict, required): A dictionary mapping input names to tensor values. Keys become the input names in the exported model. Values can be:
  - `torch.Tensor`: Regular tensors
  - Nested structures: `dict`, `list`, or `tuple` containing tensors (will be flattened)

### Returns

- **TracedTensor(s)**: Returns TracedTensor object(s) that wrap the input tensors and record all operations performed on them.
  - Single tensor input: Returns a single TracedTensor
  - Multiple tensor inputs: Returns a tuple of TracedTensors (in dict key order)
  - Nested structures: Returns the same structure with TracedTensors replacing regular tensors

### Behavior

- Creates a new traced tensor node context (or reuses existing if called again with same `node_name`)
- Wraps input tensors in TracedTensor objects that use `torch.fx` to record operations
- All operations on TracedTensors are automatically captured in the computation graph
- Operations can span multiple function calls - helper functions work seamlessly
- Must be paired with `output_tensors()` to finalize the node

### Notes

- Graph interpretation must be enabled via `start()` before calling
- TracedTensors support most PyTorch tensor operations
- Cannot mix TracedTensors from different node contexts in a single operation
- Cannot pass TracedTensors to `@annotate.method()` decorated functions (call `output_tensors()` first)
- Calling `input_tensors()` multiple times with the same `node_name` will reuse the existing and add another input to the node context

---

## `annotate.output_tensors()`

Mark traced tensor outputs and finalize a traced tensor node.

### Signature

```python
annotate.output_tensors(node_name: str, tensors: dict, **kwargs)
```

### Parameters

- **`tensors`** (dict, required): A dictionary mapping output names to TracedTensor values. Keys become the output names in the exported model. Values should be TracedTensors (or nested structures containing them) that were derived from `input_tensors()`.

- **`node_name`** (str, required): The node name. Must match the name used in the corresponding `input_tensors()` call.

- **`**kwargs`**: Export configuration options:
  - **`export_with`** (str): Backend for exporting. **Currently only `"torch"` is supported** for traced tensor nodes.
  - **`backend_params`** (dict): Backend-specific parameters.

### Behavior

- Finalizes the traced tensor node by marking which tensors are outputs
- Compiles the recorded computation graph into an exportable model
- Automatically prunes unused inputs (inputs that don't contribute to outputs)
- Returns the underlying raw tensors (unwrapped from TracedTensor)
- After calling, the TracedTensors for this node are no longer tracing (operations return regular tensors)

### Notes

- Must be called after `input_tensors()` with the same `node_name`
- All output tensors must be derived from the corresponding input TracedTensors
- Unused inputs are automatically detected and removed from the exported model

---

## `annotate.compile_graph()`

Compile and save the computational graph from traced nodes.

### Signature

```python
annotate.compile_graph(visualize: bool = True, 
                       merge_nodes: MergeCfgEnum = MergeCfgEnum.NO_MERGE)
```

### Parameters

- **`visualize`** (bool, optional): If `True`, generates a visual representation of the graph structure and saves it to the output directory. Visualization errors are logged but don't stop compilation. Defaults to `True`.

- **`merge_nodes`** (MergeCfgEnum, optional): Strategy for merging nodes in the graph. Options:
  - `MergeCfgEnum.NO_MERGE`: Keep all nodes separate (default)
  - `MergeCfgEnum.AUTOMATIC`: Automatically merge completely sequential nodes
  - `MergeCfgEnum.SIGNATURE`: Merge nodes by signature (not yet implemented)
  
  Defaults to `MergeCfgEnum.NO_MERGE`.

### Behavior

This method performs the complete pipeline:

1. **Compiles models**: Converts traced nodes into exportable models using configured backends
2. **Builds connections**: Analyzes data flow to connect nodes in the graph
3. **Merges nodes**: Applies the specified node merging strategy (if not `NO_MERGE`)
4. **Saves models**: Exports compiled models to `{save_path}/{name}/` directory
5. **Generates YAML**: Creates `{name}.yaml` with complete graph description
6. **Creates visualization**: Generates graph visualization (if `visualize=True`)
7. **Logs statistics**: Prints graph statistics (nodes, connections, etc.)

### Generated Artifacts

When called, this method creates:

- **Compiled models**: Individual model files for each node (e.g., `.pt` files)
- **YAML descriptor**: `{name}.yaml` containing:
  - Model descriptions (inputs, outputs, parameters)
  - Pipeline connections (`data_flow` and `feedback_flow`)
  - System information
- **Visualization**: `{name}.png` showing the graph structure (if `visualize=True`)


### Output YAML Structure

The generated YAML file contains:

```yaml
models:
  node_name:
    backend: torch
    inputs:
      - name: input_name
        shape: [10, 3]
        dtype: float32
    outputs:
      - name: output_name
        shape: [10, 3]
        dtype: float32
    parameters:
      model_path: ./node_name.pt

pipeline:
  data_flow:
    - source: source_node/output_name
      targets:
        - target_node/input_name
  feedback_flow:
    - source: later_node/output_name
      targets:
        - earlier_node/input_name
  inputs:
    node_name: [input1, input2]
  outputs:
    node_name: [output1, output2]

system_info:
  python_version: "3.10.0"
  pytorch_version: "2.0.0"
  # ... more system information
```

### Graph Statistics

The method logs statistics including:
- **Computation nodes**: Total number of nodes in the graph
- **Dangling inputs**: Inputs with no connections (graph-level inputs)
- **Dangling outputs**: Outputs with no connections (graph-level outputs)
- **Internal connections**: Number of node-to-node connections
- **Total edges**: Total number of edges in the graph

### Notes

- Must be called after `stop()` has been called
- All traced nodes must have completed successfully
- Visualization errors are logged but don't stop the compilation
- The YAML file can be used by downstream deployment frameworks
- Automatic merging only works for completely sequential nodes (see Advanced Graph Operations documentation)

---

## `annotate.mirror_leapp_tags()`

Transfer LEAPP's internal tracing tags from one tensor to another when data is duplicated without using standard PyTorch operations like `clone()` or `detach()`.

### Signature

```python
annotate.mirror_leapp_tags(source, target)
```

### Parameters

- **`source`** (Tensor, required): The tensor containing the original data and LEAPP internal tags.
- **`target`** (Tensor, required): The tensor that should receive the tags. Must contain exactly the same values as `source`.

### Behavior

1. **Verifies data equivalence**: Checks that `source` and `target` contain exactly the same values
2. **Transfers tags**: If verification passes, copies all LEAPP internal tracking tags from `source` to `target`
3. **Logs error on mismatch**: If data doesn't match exactly, logs an error and does nothing to prevent incorrect tracing


- **See detailed guide**: For more information and use cases, see [Advanced Graph Operations](2_advanced_graph.md#maintaining-tracing-with-mirror_leapp_tags)

---

## Complete API Workflow Example

Here's a complete example demonstrating all API methods:

```python
import torch
import torch.nn as nn
from leapp import annotate, MergeCfgEnum

# Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.linear(x)

# Create model instance
model = SimpleNet()
model.eval()

# Preprocessing function
@annotate.method(export_with="jit", node_name="preprocess")
def preprocess(raw_data):
    """Normalize input data."""
    normalized = (raw_data - raw_data.mean()) / (raw_data.std() + 1e-6)
    return normalized

# Model inference function
@annotate.method(
    export_with="jit",
    node_name="inference",
    environment_constants=["model"]
)
def run_inference(data):
    """Run model inference."""
    with torch.no_grad():
        output = model(data)
    return output

# Main pipeline
def main():
    # 1. Start tracing
    annotate.start(
        name="complete_pipeline",
        save_path="./exports",
        verbose=True
    )
    
    # Create sample input
    raw_input = torch.randn(1, 10)
    
    # 2. Trace preprocessing
    preprocessed = preprocess(raw_input)
    
    # 3. Trace inference
    predictions = run_inference(preprocessed)
    
    # 4. Trace postprocessing with a block
    with annotate.block(
        "postprocess",
        inputs=["predictions"],
        outputs=["final_output"],
        export_with="jit"
    ):
        probabilities = torch.softmax(predictions, dim=1)
        final_output = torch.argmax(probabilities, dim=1)
    
    print(f"Final output: {final_output}")
    
    # 5. Stop tracing
    annotate.stop()
    
    # 6. Compile and export the graph
    annotate.compile_graph(
        visualize=True,
        merge_nodes=MergeCfgEnum.AUTOMATIC
    )
    
    print("Pipeline exported successfully!")
    print("Check ./exports/complete_pipeline/ for outputs")

if __name__ == "__main__":
    main()
```

This will create:
```
exports/complete_pipeline/
├── complete_pipeline.yaml    # Graph description
├── complete_pipeline.png     # Visualization
├── preprocess.pt             # Preprocessing model
├── inference.pt              # Inference model
└── postprocess.pt            # Postprocessing model
└── log.txt                   # generated logs for the export process
```

---


## Best Practices

1. **Always call methods in order**: `start()` → trace code → `stop()` → `compile_graph()`
2. **Use meaningful node names**: Makes debugging and deployment easier
3. **Specify `export_with`**: Explicitly declare your export backend
4. **Run feedback loops twice**: Required for cycle detection
5. **Use verbose mode during development**: Helps debug tracing issues
6. **Choose the right annotation method**:
   - Use `@annotate.method()` for self-contained functions
   - Use `annotate.block()` for inline code with explicit I/O
   - Use `input_tensors()`/`output_tensors()` for operations spanning multiple functions or dynamic scenarios
7. **Always pair `input_tensors()` with `output_tensors()`**: Forgetting to call `output_tensors()` leaves the node incomplete
8. **Don't mix TracedTensors across contexts**: Complete one traced tensor node before starting another

---

## See Also

- [Getting Started Guide](0_getting_started.md) - Learn the basics of LEAPP
- [Advanced Node Operations](1_advanced_nodes.md) - Advanced node tracing options
- [Advanced Graph Operations](2_advanced_graph.md) - Advanced graph crafting operations

