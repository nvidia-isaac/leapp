# LEAPP API Reference

This document provides a reference for LEAPP's public runtime and annotation APIs. Use the `leapp` module for graph lifecycle control and the global `annotate` instance for node annotations.

## Quick Reference

```python
import leapp
from leapp import annotate

# Core workflow
leapp.start(name="my_graph", save_path=".", verbose=False)
# ... trace your code using annotate.method()
#     or annotate.input_tensors() / annotate.output_tensors()
leapp.stop()
leapp.compile_graph(visualize=True)
```

---

## `leapp.start()`

Initialize and start LEAPP graph interpretation.

### Signature

```python
leapp.start(name: str, save_path: str = ".", verbose: bool = False, dry_run: bool = False, non_traced = None, max_cached_io: int = 5, global_patching: bool = True)
```

### Parameters

- **`name`** (str, required): The name of the graph to be created. This will be used as the directory name where graph artifacts are saved.
- **`save_path`** (str, optional): The base directory path where the graph directory will be created. Defaults to `"."` (current directory).
- **`verbose`** (bool, optional): If `True`, enables verbose logging output. Defaults to `False`.
- **`dry_run`** (bool, optional): If `True`, skips model compilation and export. Used to verify graph structure and I/O without the cost of compilation. Defaults to `False`.
- **`non_traced`** (list[str], optional): List of node names to exclude from tracing/export. These nodes still capture inputs/outputs, contribute to graph connectivity, and appear in the YAML — they just won't have compiled models. Defaults to `None`.
- **`max_cached_io`** (int, optional): Controls how many re-entry I/O examples LEAPP caches per node for multi-example validation. Higher values improve confidence for looped/stateful pipelines at the cost of memory. Defaults to `5`.
- **`global_patching`** (bool, optional): If `True`, patches torch numpy functions for TracedTensor compatibility. Defaults to `True`. Set to `False` if global patching causes environment issues.

  > **Warning:** Setting `global_patching=False` disables patches that allow traced tensors to pass through `torch.from_numpy()` and related numpy-interop functions. If your pipeline calls any such functions on traced tensors, they will silently return untraced results and those operations will be invisible to LEAPP. Only disable this if you are certain your pipeline does not use numpy-interop on traced values.

### Behavior

- Creates the save directory `{save_path}/{name}/` if it doesn't exist
- Configures logging based on the `verbose` flag. If set to true, logs will also print to console. Logs can always be accessed at `{save_path}/{name}/log*`
- Enables graph interpretation mode for tracing
- If called while interpretation is already active, it will reset the graph (this is discouraged)

### Notes

- Must be called before annotated functions are invoked and before `annotate.input_tensors()` begins tracing
- Creates a directory structure: `{save_path}/{name}/`
- All traced nodes and outputs will be saved in this directory

---

## `leapp.stop()`

Stop LEAPP graph interpretation and disable tracing.

### Signature

```python
leapp.stop()
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

## `annotate.input_tensors()`

Create traced tensor inputs for programmatic node definition. Returns TracedTensor objects that record all subsequent operations.

### Signature

```python
traced_tensors = annotate.input_tensors(node_name: str, tensors)
```

### Parameters

- **`node_name`** (str, required): The unique name to identify this node in the computational graph.

- **`tensors`** (required): The top-level payload must be either a dictionary of named raw tensors or a `TensorSemantics` object / list of `TensorSemantics`. Bare tensors and other unnamed top-level collections are not supported because LEAPP requires explicit tensor names. See `5_semantic_data_annotation.md` for more details.

### Returns

- **TracedTensor(s)**: Returns traced values that record all subsequent operations.
  - Single input: Returns a single traced value
  - Multiple named inputs: Returns a tuple in dict key order

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
- Cannot pass TracedTensors from one node context into another active traced node
- Calling `input_tensors()` multiple times with the same `node_name` will reuse the existing and add another input to the node context

---

## `annotate.output_tensors()`

Mark traced tensor outputs and finalize a traced tensor node.

### Signature

```python
annotate.output_tensors(node_name: str, tensors, static_outputs = None, **kwargs)
```

### Parameters

- **`tensors`** (required): The top-level payload must be either a dictionary of named raw tensors or a `TensorSemantics` object / list of `TensorSemantics`. Bare tensors and other unnamed top-level collections are not supported because LEAPP requires explicit tensor names. See `5_semantic_data_annotation.md` for more details.

- **`node_name`** (str, required): The node name. Must match the name used in the corresponding `input_tensors()` call.

- **`static_outputs`** (optional): Constant outputs that should be emitted but are not derived from traced inputs. The top-level payload follows the same naming contract as `tensors`: pass either a dictionary of named raw tensors or a `TensorSemantics` object / list of `TensorSemantics`. The referenced values must still be plain `torch.Tensor` values, not `TracedTensor` values.

- **`**kwargs`**: Export configuration options:
  - **`export_with`** (str | None): Backend for exporting. Common values are `"jit"` and `"onnx"`. You can also specify `"jit-script"`, `"jit-trace"`, `"onnx-dynamo"`, or `"onnx-torchscript"` to control the exact backend — see [Advanced Export](2_advanced_export.md) for details.
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

## `annotate.method()`

Create a decorator for tracing functions/methods in the computational graph. method is a shorthand for annotating tensors for modules that follow a function structure. It sets up input_tensors and output_tensors in the backend.

### Signature

```python
@annotate.method(**params)
def your_function(...):
    ...
```

### Parameters

All parameters are optional keyword arguments (`**params`):

- **`node_name`** (str): Custom name for the node. If not provided, uses the function's name.
- **`export_with`** (str | None): Backend to use for exporting the model. Common values are `"jit"` and `"onnx"`.
- **`backend_params`** (dict): Backend-specific parameters forwarded to the selected export backend.

### Behavior

- Wraps the function to trace its execution when called
- Captures tensor inputs from the function signature and tensor outputs from the return value
- Creates a `TracedTensorNode` in the LEAPP computational graph
- If graph interpretation is disabled, the function executes normally without tracing
- Tensor-valued default arguments that are not explicitly passed are automatically registered as buffers

### Notes

- Graph interpretation must be enabled via `start()` before decorated functions are called
- Input and output names are automatically derived from function parameters and return values
- This is the recommended shorthand for self-contained functions
- even when called, you may still use other annotate. api to annotate things like other inputs or state values inside the method.

---

## `annotate.register_buffer()`

Register a preallocated tensor for a traced node.

### Signature

```python
annotate.register_buffer(node_name: str, tensors)
```

### Parameters

- **`node_name`** (str, required): Name of an existing traced tensor node. Call `annotate.input_tensors()` first to create the node.
- **`tensors`** (required): Buffer payload to register. Supported forms:
  - a single tensor
  - a `list` or `tuple` of tensors
  - a `dict` mapping buffer names to tensors

### Returns

- A single traced value for a single tensor input
- A tuple of traced values for multi-value inputs

### Behavior

- Wraps the preallocated tensor so that subsequent in-place writes (`buffer[:] = value`) are traced
- Supports repeated calls; unnamed list/tuple entries receive auto-generated names such as `buffer_0`, `buffer_1`, ...
- The return value must be reassigned to the variable or attribute you intend to mutate

### Notes

- `register_buffer()` is only supported for traced tensor nodes created with `input_tensors()` or `method()`
- The tensor passed in must be raw (not already traced)
- Use this for fixed-location staging buffers that are updated in-place each call — not for constants
- Use `state_tensors()` or `annotate.module()` when the value should behave as recurrent feedback across calls

---

## `annotate.state_tensors()`

Declare recurrent/state tensors for a traced node. State tensors behave as both inputs and outputs of the node.

### Signature

```python
annotate.state_tensors(node_name: str, tensors: dict[str, torch.Tensor])
```

### Parameters

- **`node_name`** (str, required): Name of an existing traced tensor node. Call `annotate.input_tensors()` first to create the node.
- **`tensors`** (dict, required): Mapping of state names to initial tensor values.

### Returns

- A single traced state tensor for a one-entry dict
- A tuple of traced state tensors for a multi-entry dict, in dict key order

### Behavior

- Registers state placeholders as additional node inputs
- Marks those values as feedback-capable state for graph compilation
- Only states that are later passed to `update_state()` become feedback outputs. If `update_state()` is omitted, the declared state remains a regular input and does not create feedback.
- Nested state structures are not supported. Each state name must map to a single tensor.

### Notes

- `state_tensors()` is only supported for traced tensor nodes created with `input_tensors()` or `method()`
- State tensor names must be unique within the node
- If you need structured state, explicitly list each state tensor with its own name, or use `input_tensors()` and rely on LEAPP feedback detection.
- Use this for hidden state, rolling history, or other explicit recurrent values

---

## `annotate.update_state()`

Update the output values for state tensors previously declared with `annotate.state_tensors()`.

### Signature

```python
annotate.update_state(node_name: str, tensors: dict[str, TracedTensor])
```

### Parameters

- **`node_name`** (str, required): Name of an existing traced tensor node.
- **`tensors`** (dict, required): Mapping from previously declared state names to their updated traced values.

### Returns

- Passthrough of the provided updated state values:
  - single-entry dict returns a single value
  - multi-entry dict returns a tuple in dict key order

### Behavior

- Binds new output values to state tensors declared with `state_tensors()`
- Validates that the updated values match the original state shape and dtype
- During graph export, these become feedback outputs

### Notes

- Call `state_tensors()` before `update_state()`
- Omitted state updates fall back to passthrough behavior

---

## `annotate.module()`

Register an `nn.Module` for automatic buffer tracking inside a traced tensor node.

### Signature

```python
annotate.module(node_name: str, model: torch.nn.Module, buffer_names: list[str] | None = None)
```

### Parameters

- **`node_name`** (str, required): Name of an existing traced tensor node. Call `annotate.input_tensors()` first to create the node.
- **`model`** (`torch.nn.Module`, required): Module whose registered buffers should be tracked.
- **`buffer_names`** (`list[str] | None`, optional): Optional subset of buffer names to track. If omitted, all registered buffers are tracked.

### Behavior

- Temporarily injects tracked versions of model buffers so the forward pass is traced through them
- Detects buffer reassignment during execution
- Emits mutated buffers as feedback state and preserves untouched buffers as frozen constants

### Notes

- Call `annotate.module()` after `input_tensors()` and before the module forward pass
- Reassignment is tracked, but in-place mutation like `self.h.copy_(...)` is not
- Use explicit `state_tensors()` / `update_state()` if you need in-place state handling

---

## `leapp.compile_graph()`

Compile and save the computational graph from traced nodes.

### Signature

```python
leapp.compile_graph(visualize: bool = True, verbose: bool = None, validate: bool = True, dry_run: bool = False, rtol: float = 1e-3, atol: float = 1e-5, strict: bool = True)
```

### Parameters

- **`visualize`** (bool, optional): If `True`, generates a visual representation of the graph structure and saves it to the output directory. Visualization errors are logged but don't stop compilation. Defaults to `True`.
- **`verbose`** (bool | None, optional): Override verbose logging for the compile step. `None` leaves the current setting unchanged. Defaults to `None`.
- **`validate`** (bool, optional): If `True`, validates exported models by comparing their outputs against the captured traced outputs. Defaults to `True`.
- **`dry_run`** (bool, optional): If `True`, skips model compile/save/validate at compile time while still tracing graph structure and generating the YAML. Useful for CI/headless runs when artifacts are not needed. Defaults to `False`.
- **`rtol`** (float, optional): Relative tolerance used in `torch.allclose` during model validation. Defaults to `1e-3`.
- **`atol`** (float, optional): Absolute tolerance used in `torch.allclose` during model validation. Defaults to `1e-5`.
- **`strict`** (bool, optional): If `True`, raises an exception when any model fails validation. Defaults to `True`.

### Behavior

This method performs the complete pipeline:

1. **Compiles models**: Converts traced nodes into exportable models using configured backends
2. **Builds connections**: Analyzes data flow to connect nodes in the graph
3. **Saves models**: Exports compiled models to `{save_path}/{name}/` directory
4. **Generates YAML**: Creates `{name}.yaml` with complete graph description
5. **Creates visualization**: Generates graph visualization (if `visualize=True`)
6. **Logs statistics**: Prints graph statistics (nodes, connections, etc.)

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
    inputs:
      - name: input_name
        shape: [10, 3]
        dtype: float32
    outputs:
      - name: output_name
        shape: [10, 3]
        dtype: float32
    parameters:
      backend: jit
      model_path: ./node_name.pt
      md5sum: ...
      sha256sum: ...

pipeline:
  data_flow:
    source_node/output_name: [target_node/input_name]
  feedback_flow:
    later_node/output_name: [earlier_node/input_name]
  inputs:
    node_name: [input1, input2]
  outputs:
    node_name: [output1, output2]

system information:
  python version: "3.12.9"
  torch version: "2.7.0+cu126"
  leapp version: "0.4.0"
  leapp config version: "1.0"
  cuda version: "12.6"
  os: Linux
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
3. **Raises on mismatch**: If data doesn't match exactly, LEAPP logs an error and raises instead of copying incorrect tracing metadata


- **See detailed guide**: For more information and use cases, see [Advanced Graph Operations](3_advanced_graph.md#maintaining-tracing-with-mirror_leapp_tags)

---

## Representative Workflow Example

Here's a representative example showing the core public workflow:

```python
import torch
import torch.nn as nn
import leapp
from leapp import annotate

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

# Main pipeline
def main():
    # 1. Start tracing
    leapp.start(
        name="complete_pipeline",
        save_path="./exports",
        verbose=True
    )
    
    # Create sample input
    raw_input = torch.randn(1, 10)
    
    # 2. Trace preprocessing
    preprocessed = preprocess(raw_input)
    
    # 3. Trace inference with traced tensors
    pred_traced = annotate.input_tensors("inference", {"features": preprocessed})
    predictions = model(pred_traced)
    annotate.output_tensors("inference", {"predictions": predictions}, export_with="jit")

    # 4. Trace postprocessing with traced tensors
    pred_traced = annotate.input_tensors("postprocess", {"predictions": predictions})
    probabilities = torch.softmax(pred_traced, dim=1)
    final_output = torch.argmax(probabilities, dim=1)
    annotate.output_tensors("postprocess", {"final_output": final_output}, export_with="jit")

    print(f"Final output: {final_output}")
    
    # 5. Stop tracing
    leapp.stop()
    
    # 6. Compile and export the graph
    leapp.compile_graph(visualize=True)
    
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
├── postprocess.pt            # Postprocessing model
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
   - Use `input_tensors()`/`output_tensors()` for operations spanning multiple functions, inline code, or dynamic scenarios
7. **Always pair `input_tensors()` with `output_tensors()`**: Forgetting to call `output_tensors()` leaves the node incomplete
8. **Don't mix TracedTensors across contexts**: Complete one traced tensor node before starting another

---

## `InferenceManager`

Load an exported LEAPP graph from YAML and run the full pipeline at inference time.

### Signature

```python
from leapp import InferenceManager

manager = InferenceManager(model_path: str)
```

### Parameters

- **`model_path`** (str, required): Path to the `.yaml` graph description generated by `leapp.compile_graph()`.

### Behavior

- Loads the exported graph description from YAML
- Loads the referenced node models from `model_path`
- Validates pipeline routing and shape/dtype compatibility
- Preallocates node input buffers
- Automatically prepopulates feedback inputs from `pipeline.initial_values` when present

### Common Usage

```python
from leapp import InferenceManager

manager = InferenceManager("my_graph/my_graph.yaml")

print(manager.inputs)
print(manager.outputs)

sample_inputs = manager.get_mock_input()
outputs = manager.run_policy(sample_inputs)

# Equivalent shorthand:
outputs = manager(sample_inputs)
```

### Methods

| Method | Description |
|---|---|
| `run_policy(inputs)` | Run the full pipeline. `inputs` is a dict of `'node_name/input_name'` to `torch.Tensor`. Returns a dict of final pipeline outputs. `manager(inputs)` is an equivalent shorthand. |
| `get_mock_input()` | Generate random tensors for every external graph input with the correct shape, dtype, and device. |
| `set_input_value(node_name, input_name, value)` | Overwrite a specific node input buffer, useful for manually overriding feedback state. |

### Properties

| Property | Description |
|---|---|
| `inputs` | List of expected graph input keys in `node_name/input_name` format. |
| `outputs` | List of final graph output keys in `node_name/output_name` format. |
| `feedback_inputs` | List of feedback input targets taken from `pipeline.feedback_flow`. |

### Notes

- Input dictionaries passed to `run_policy()` must use keys in `node_name/input_name` format
- `InferenceManager` currently runs exported or referenced `jit` and `onnx` models
- Feedback state persists across successive `run_policy()` calls unless you overwrite it manually

---

## See Also

- [Getting Started Guide](0_getting_started.md) - Learn the basics of LEAPP
- [Advanced Node Operations](1_advanced_nodes.md) - Advanced node tracing options
- [Advanced Graph Operations](3_advanced_graph.md) - Advanced graph crafting operations
- [Runtime And Validation Guide](4_runtime_and_validation.md) - Export validation and runtime verification
- [Semantic Data Guide](5_semantic_data_annotation.md) - Injecting semantic data into the LEAPP config

