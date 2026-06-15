=============
API reference
=============

Public runtime and annotation APIs. Use the ``leapp`` module for graph
lifecycle control and the global ``annotate`` instance for node
annotations.

.. code-block:: python

   import leapp
   from leapp import annotate

   leapp.start(name="my_graph", save_path=".", verbose=False)
   # ... trace your code using annotate.method()
   #     or annotate.input_tensors() / annotate.output_tensors()
   leapp.stop()
   leapp.compile_graph(visualize=True)

Graph lifecycle
===============

.. _api-leapp-start:

``leapp.start()``
-----------------

Initialize and start LEAPP graph interpretation.

Signature
~~~~~~~~~

.. code-block:: python

   leapp.start(
       name: str,
       save_path: str = ".",
       verbose: bool = False,
       dry_run: bool = False,
       non_traced=None,
       max_cached_io: int = 5,
       global_patching: bool = True,
   )

Parameters
~~~~~~~~~~

* ``name`` (str, required): Graph name; used as the directory for
  artifacts.
* ``save_path`` (str, optional): Base directory where the graph
  directory is created. Defaults to ``"."``.
* ``verbose`` (bool, optional): Enable verbose logging. Defaults to
  ``False``.
* ``dry_run`` (bool, optional): Skip model compilation and export.
  Useful to verify graph structure without paying export cost. Defaults
  to ``False``.
* ``non_traced`` (list[str], optional): Node names to exclude from
  tracing/export. They still capture I/O, contribute to graph
  connectivity, and appear in the YAML. Defaults to ``None``.
* ``max_cached_io`` (int, optional): How many re-entry I/O examples
  LEAPP caches per node for multi-example validation. Defaults to ``5``.
* ``global_patching`` (bool, optional): Patch ``torch`` numpy interop
  functions for ``TracedTensor`` compatibility. Defaults to ``True``.

  .. warning::

     Disabling ``global_patching`` disables patches that allow traced
     tensors to pass through ``torch.from_numpy()`` and related numpy
     interop functions. If your pipeline calls any such functions on
     traced tensors, they silently return untraced results and those
     operations become invisible to LEAPP.

Behavior
~~~~~~~~

* Creates ``{save_path}/{name}/`` if it does not exist.
* Configures logging based on the ``verbose`` flag. Logs are always
  written to ``{save_path}/{name}/log*``.
* Enables graph interpretation mode for tracing.
* If called while interpretation is already active, the graph resets
  (this is discouraged).

Notes
~~~~~

* Must be called before annotated functions are invoked and before
  ``annotate.input_tensors()`` begins tracing.
* All traced nodes and outputs are saved under
  ``{save_path}/{name}/``.

``leapp.stop()``
----------------

Stop LEAPP graph interpretation and disable tracing.

Signature
~~~~~~~~~

.. code-block:: python

   leapp.stop()

Behavior
~~~~~~~~

* Disables graph interpretation mode that was enabled by ``start()``.
* Performs safety checks to ensure no active tracing is in progress.
* Must be called after all tracing operations are complete and before
  ``compile_graph()``.

``leapp.compile_graph()``
-------------------------

Compile and save the computational graph from traced nodes.

Signature
~~~~~~~~~

.. code-block:: python

   leapp.compile_graph(
       visualize: bool = True,
       verbose: bool | None = None,
       validate: bool = True,
       dry_run: bool = False,
       rtol: float = 1e-3,
       atol: float = 1e-5,
       strict: bool = True,
       graph_configs: GraphConfigs | None = None,
   )

Parameters
~~~~~~~~~~

* ``visualize`` (bool, optional): Generate a graph visualization.
  Defaults to ``True``. Visualization errors are logged but do not stop
  compilation.
* ``verbose`` (bool | None, optional): Override verbose logging for the
  compile step. ``None`` leaves the current setting unchanged.
* ``validate`` (bool, optional): Validate exported models against
  captured traced outputs. Defaults to ``True``.
* ``dry_run`` (bool, optional): Skip model compile/save/validate while
  still tracing graph structure and generating YAML. Defaults to
  ``False``.
* ``rtol`` (float, optional): Relative tolerance for
  ``torch.allclose()``. Defaults to ``1e-3``.
* ``atol`` (float, optional): Absolute tolerance for
  ``torch.allclose()``. Defaults to ``1e-5``.
* ``strict`` (bool, optional): Raise on validation failure. Defaults
  to ``True``.
* ``graph_configs`` (GraphConfigs, optional): Graph-level metadata to serialize
  under the generated YAML ``pipeline.configs`` entry.

Behavior
~~~~~~~~

The method performs the complete pipeline:

#. Compile traced nodes into exportable models using configured
   backends.
#. Build connections by analyzing data flow.
#. Save compiled models to ``{save_path}/{name}/``.
#. Generate ``{name}.yaml`` with the complete graph description.
#. Generate ``{name}.png`` if ``visualize=True``.
#. Log graph statistics.

``GraphConfigs``
----------------

Graph-level metadata passed to :func:`~leapp.compile_graph`.

.. code-block:: python

   from leapp import GraphConfigs

   leapp.compile_graph(
       graph_configs=GraphConfigs(
           frequency=50,
           extra={"runtime": "isaac_lab"},
       ),
   )

This emits fields under the generated ``pipeline.configs`` entry:

.. code-block:: yaml

   pipeline:
     data_flow: {}
     feedback_flow: {}
     inputs: {}
     outputs: {}
     configs:
       frequency: 50
       runtime: isaac_lab

Parameters
~~~~~~~~~~

* ``frequency`` (float | int, optional): Graph-level execution frequency in Hz.
* ``extra`` (dict, optional): Additional graph-level fields. Keys are flattened
  into the ``pipeline.configs`` entry rather than serialized under an ``extra`` key.

Generated artifacts
~~~~~~~~~~~~~~~~~~~

* Compiled models --- one file per node (``.pt``, ``.onnx``).
* ``{name}.yaml`` --- model descriptions, pipeline connections
  (``data_flow`` and ``feedback_flow``), and system information.
* ``{name}.png`` --- graph visualization (when ``visualize=True``).

Output YAML structure
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

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
     leapp version: "0.5.2"
     leapp config version: "1.2"
     cuda version: "12.6"
     os: Linux

Graph statistics
~~~~~~~~~~~~~~~~

The method logs statistics including:

* Computation nodes --- total node count
* Dangling inputs --- inputs with no connections (graph-level inputs)
* Dangling outputs --- outputs with no connections (graph-level outputs)
* Internal connections --- node-to-node connection count
* Total edges --- total edge count

Semantic metadata
=================

``TemporalPeriodMs``
--------------------

Mark one ``element_names`` axis as temporal and attach an absolute period in
milliseconds to the tensor YAML entry.

.. code-block:: python

   from leapp import TensorSemantics, TemporalPeriodMs

   TensorSemantics(
       name="actions",
       ref=actions,
       element_names=[TemporalPeriodMs(100), ["hip", "knee", "ankle"]],
   )

This emits the reserved bare axis sentinel and the period metadata:

.. code-block:: yaml

   element_names:
   - __temporal_axis__
   - [hip, knee, ankle]
   temporal_period_ms: 100

``__temporal_axis__`` is reserved for LEAPP output; use
``TemporalPeriodMs`` rather than writing the sentinel directly.

``TensorSemantics``
-------------------

Describe a named tensor and optional semantic metadata for YAML serialization.
Use ``TensorSemantics`` with :func:`~leapp.annotate.input_tensors` and
:func:`~leapp.annotate.output_tensors` when tensor entries need meaning beyond
shape and dtype.

Signature
~~~~~~~~~

.. code-block:: python

   TensorSemantics(
       name: str = None,
       ref = None,
       kind: InputKindEnum | OutputKindEnum | str | None = None,
       element_names: list | None = None,
       temporal_period_ms: float | int | None = None,
       extra: dict | None = None,
   )

Parameters
~~~~~~~~~~

* ``name`` (str, required): Tensor name to use in the generated pipeline YAML.
* ``ref`` (Tensor or ndarray, required): Tensor value being annotated.
* ``kind`` (InputKindEnum | OutputKindEnum | str, optional): Semantic role for
  the tensor. Enum values are serialized to their string ``.value``.
* ``element_names`` (list | str, optional): Human-readable element names. A
  string becomes ``[[name]]``; a flat list of strings becomes ``[[...]]``; a
  per-dimension list is preserved with optional ``None`` entries. Include
  ``TemporalPeriodMs(...)`` as a bare axis item to mark that axis temporal.
* ``temporal_period_ms`` (float | int, optional): Temporal period in
  milliseconds serialized to YAML. Prefer setting this with
  ``TemporalPeriodMs(...)`` in ``element_names`` so the temporal axis is also
  marked.
* ``extra`` (dict, optional): Additional semantic fields. Keys are flattened
  into the tensor YAML entry rather than serialized under an ``extra`` key.

Behavior
~~~~~~~~

* ``TensorSemantics`` can be passed as a single object or as a list of objects.
* When using a list, every item must be ``TensorSemantics``; mixing raw tensors
  and ``TensorSemantics`` in the same list is not supported.
* ``TensorSemantics`` objects are not placed inside dicts. The ``name`` field
  provides the tensor key.
* Public semantic fields with non-``None`` values are serialized in the YAML.
* ``extra`` fields are flattened into the same YAML mapping as built-in fields.

.. warning::

   ``extra`` fields are non-standard, project-defined metadata. Downstream
   deployment frameworks should not be expected to understand or support them.
   Use ``extra`` only for project-specific integration data, and prefer
   standard LEAPP semantic fields whenever possible.

Enum values
~~~~~~~~~~~

``InputKindEnum`` currently defines:

.. code-block:: text

   JOINT_POSITION = state/joint/position
   JOINT_VELOCITY = state/joint/velocity
   JOINT_EFFORT = state/joint/effort
   BODY_POSE = state/body/pose
   BODY_VEL = state/body/velocity
   BODY_ACC = state/body/acceleration
   BODY_LINEAR_ACCELERATION = state/body/linear_acceleration
   BODY_LINEAR_VELOCITY = state/body/linear_velocity
   BODY_ANGULAR_ACCELERATION = state/body/angular_acceleration
   BODY_ANGULAR_VELOCITY = state/body/angular_velocity
   BODY_ROTATION = state/body/rotation
   BODY_POSITION = state/body/position
   WRENCH = state/wrench
   VECTOR3D = state/vector3d
   COMMAND_JOINT_POSITION = command/joint/position
   COMMAND_JOINT_VELOCITY = command/joint/velocity
   COMMAND_JOINT_TORQUES = command/joint/torques
   COMMAND_BODY_ROTATION = command/body/rotation
   COMMAND_BODY_VELOCITY = command/body/velocity
   COMMAND_POSE = command/body/pose

``OutputKindEnum`` currently defines:

.. code-block:: text

   KP = kp
   KD = kd
   JOINT_POSITION = target/joint/position
   JOINT_VELOCITY = target/joint/velocity
   JOINT_TORQUES = target/joint/torques
   JOINT_EFFORT = target/joint/effort
   BODY_POSITION = target/body/position
   BODY_LINEAR_ACCELERATION = target/body/linear_acceleration
   BODY_ORIENTATION = target/body/orientation
   BODY_LINEAR_VELOCITY = target/body/linear_velocity
   BODY_ANGULAR_ACCELERATION = target/body/angular_acceleration

See :doc:`/guides/semantics` for examples and field guidance.

Annotations
===========

``annotate.input_tensors()``
----------------------------

Create traced tensor inputs for programmatic node definition.

Signature
~~~~~~~~~

.. code-block:: python

   traced_tensors = annotate.input_tensors(node_name: str, tensors)

Parameters
~~~~~~~~~~

* ``node_name`` (str, required): Unique node identifier.
* ``tensors`` (required): Either a dict of named raw tensors or a
  ``TensorSemantics`` / list of ``TensorSemantics``. Bare tensors and
  other unnamed top-level collections are not supported. See
  :doc:`/guides/semantics`.

Returns
~~~~~~~

* A single traced value for a single input, or a tuple in dict key order
  for multiple inputs.

Behavior
~~~~~~~~

* Creates a new traced tensor node context (or reuses the existing one
  if called again with the same ``node_name``).
* Wraps input tensors in ``TracedTensor`` objects that use ``torch.fx``
  to record operations.
* Operations on ``TracedTensor`` objects are captured even across helper
  functions.
* Must be paired with ``output_tensors()`` to finalize the node.

Notes
~~~~~

* ``leapp.start()`` must have been called.
* ``TracedTensor`` supports most PyTorch tensor operations.
* Do not mix ``TracedTensor`` values from different node contexts in a
  single operation.

``annotate.output_tensors()``
-----------------------------

Mark traced tensor outputs and finalize a traced tensor node.

Signature
~~~~~~~~~

.. code-block:: python

   annotate.output_tensors(
       node_name: str,
       tensors,
       static_outputs=None,
       **kwargs,
   )

Parameters
~~~~~~~~~~

* ``node_name`` (str, required): Must match the corresponding
  ``input_tensors()`` call.
* ``tensors`` (required): Same form as ``input_tensors``.
* ``static_outputs`` (optional): Constant outputs not derived from
  traced inputs. Same naming contract as ``tensors``. Underlying values
  must be plain ``torch.Tensor``, not ``TracedTensor``.
* ``**kwargs``:

  * ``export_with`` (str | None): Export backend. Common values
    ``"jit"`` and ``"onnx"``. Other variants are ``"jit-script"``,
    ``"jit-trace"``, ``"onnx-dynamo"``, and ``"onnx-torchscript"``. See
    :doc:`/guides/export`.
  * ``backend_params`` (dict): Backend-specific parameters.

Behavior
~~~~~~~~

* Finalizes the traced node by marking which tensors are outputs.
* Compiles the recorded computation graph into an exportable model.
* Automatically prunes inputs that do not contribute to outputs.
* Returns the underlying raw tensors (unwrapped from ``TracedTensor``).
* After this call, the node's ``TracedTensor`` instances are no longer
  tracing.

``annotate.method()``
---------------------

Decorator that traces a function or method as a single node.

Signature
~~~~~~~~~

.. code-block:: python

   @annotate.method(**params)
   def your_function(...):
       ...

Parameters
~~~~~~~~~~

* ``node_name`` (str, optional): Custom node name. Defaults to the
  function name.
* ``export_with`` (str | None, optional): Export backend. Common values
  ``"jit"`` and ``"onnx"``.
* ``backend_params`` (dict, optional): Backend-specific parameters.

Behavior
~~~~~~~~

* Wraps the function to trace its execution.
* Captures inputs from the function signature and outputs from the
  return value.
* Creates a ``TracedTensorNode`` internally.
* If interpretation is disabled, the function runs normally without
  tracing.
* Tensor-valued default arguments that are not explicitly passed are
  captured as node-local constants.

Notes
~~~~~

* Recommended shorthand for self-contained functions.
* You may still use other ``annotate.*`` APIs inside a decorated method
  to annotate additional inputs or state values.

``annotate.state_tensors()``
----------------------------

Declare recurrent / state tensors for a traced node.

Signature
~~~~~~~~~

.. code-block:: python

   annotate.state_tensors(node_name: str,
                          tensors: dict[str, torch.Tensor])

Parameters
~~~~~~~~~~

* ``node_name`` (str, required): Existing traced node.
* ``tensors`` (dict, required): Map of state names to initial tensor
  values.

Returns
~~~~~~~

* A single traced state tensor for a one-entry dict, or a tuple in dict
  key order for multi-entry dicts.

Behavior
~~~~~~~~

* Registers state placeholders as additional node inputs.
* Marks them as feedback-capable.
* Only states later passed to ``update_state()`` become feedback
  outputs. Otherwise they remain regular inputs.
* Nested state structures are not supported.

``annotate.update_state()``
---------------------------

Update output values for state tensors declared with
``annotate.state_tensors()``.

Signature
~~~~~~~~~

.. code-block:: python

   annotate.update_state(node_name: str,
                         tensors: dict[str, TracedTensor])

Parameters
~~~~~~~~~~

* ``node_name`` (str, required): Existing traced node.
* ``tensors`` (dict, required): Map from declared state names to their
  updated traced values.

Returns
~~~~~~~

* Passthrough of the provided values: single value for a one-entry dict,
  tuple for multi-entry.

Behavior
~~~~~~~~

* Binds new output values to state tensors declared with
  ``state_tensors()``.
* Validates that updated values match the original state shape and
  dtype.
* During graph export, these become feedback outputs.

``annotate.module()``
---------------------

Register an ``nn.Module`` for automatic buffer tracking inside a traced
node.

Signature
~~~~~~~~~

.. code-block:: python

   annotate.module(node_name: str,
                   model: torch.nn.Module,
                   buffer_names: list[str] | None = None)

Parameters
~~~~~~~~~~

* ``node_name`` (str, required): Existing traced node.
* ``model`` (``torch.nn.Module``, required): Module whose registered
  buffers should be tracked.
* ``buffer_names`` (list[str] | None, optional): Subset of buffer names
  to track. If omitted, all registered buffers are tracked.

Behavior
~~~~~~~~

* Temporarily injects tracked versions of model buffers so the forward
  pass is traced through them.
* Detects buffer reassignment during execution.
* Emits mutated buffers as feedback state and preserves untouched
  buffers as frozen constants.

Notes
~~~~~

* Call after ``input_tensors()`` and before the module forward pass.
* Reassignment is tracked; in-place mutation like
  ``self.h.copy_(...)`` is not.

``annotate.mirror_leapp_tags()``
--------------------------------

Transfer LEAPP's internal tracing tags from one tensor to another when
data is duplicated without using standard PyTorch operations.

Signature
~~~~~~~~~

.. code-block:: python

   annotate.mirror_leapp_tags(source, target)

Parameters
~~~~~~~~~~

* ``source`` (Tensor, required): Original tensor with LEAPP tags.
* ``target`` (Tensor, required): Tensor that receives the tags. Must
  contain exactly the same values as ``source``.

Behavior
~~~~~~~~

#. Verifies that ``source`` and ``target`` contain exactly the same
   values.
#. If verification passes, copies all LEAPP internal tracking tags from
   ``source`` to ``target``.
#. If data does not match, logs an error and raises rather than copying
   incorrect tracing metadata.

See :doc:`/guides/graph` for context.

Runtime
=======

``InferenceManager``
--------------------

Load an exported LEAPP graph from YAML and run the full pipeline from
Python. Intended as a convenient testing, smoke-validation, and
lightweight deployment utility.

Signature
~~~~~~~~~

.. code-block:: python

   from leapp import InferenceManager

   manager = InferenceManager(model_path: str)

Parameters
~~~~~~~~~~

* ``model_path`` (str, required): Path to the ``.yaml`` graph
  description generated by ``leapp.compile_graph()``.

Behavior
~~~~~~~~

* Loads the exported graph description from YAML.
* Loads the referenced node models.
* Validates pipeline routing and shape/dtype compatibility.
* Preallocates node input buffers.
* Auto-populates feedback inputs from ``pipeline.initial_values`` when
  present.

Common usage
~~~~~~~~~~~~

.. code-block:: python

   from leapp import InferenceManager

   manager = InferenceManager("my_graph/my_graph.yaml")

   print(manager.inputs)
   print(manager.outputs)

   sample_inputs = manager.get_mock_input()
   outputs = manager.run_policy(sample_inputs)

   # Equivalent shorthand:
   outputs = manager(sample_inputs)

Typical use cases:

* validate an exported graph end to end from Python
* build a simple Python deployer around exported LEAPP artifacts
* inspect expected graph inputs, outputs, and feedback state

Methods
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Description
   * - ``run_policy(inputs)``
     - Run the full pipeline. ``inputs`` is a dict of
       ``"node_name/input_name"`` to ``torch.Tensor``. Returns a dict of
       final pipeline outputs. ``manager(inputs)`` is an equivalent
       shorthand.
   * - ``get_mock_input()``
     - Generate random tensors for every external graph input with the
       correct shape, dtype, and device.
   * - ``set_input_value(node_name, input_name, value)``
     - Overwrite a specific node input buffer; useful for manually
       overriding feedback state.

Properties
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Property
     - Description
   * - ``inputs``
     - Expected graph input keys in ``node_name/input_name`` format.
   * - ``outputs``
     - Final graph output keys in ``node_name/output_name`` format.
   * - ``feedback_inputs``
     - Feedback input targets from ``pipeline.feedback_flow``.

Notes
~~~~~

* Input dictionaries passed to ``run_policy()`` must use
  ``node_name/input_name`` keys.
* ``InferenceManager`` currently runs exported or referenced ``jit`` and
  ``onnx`` models.
* Feedback state persists across successive ``run_policy()`` calls
  unless you overwrite it manually.

Worked example
==============

A representative example showing the core public workflow:

.. code-block:: python

   import torch
   import torch.nn as nn
   import leapp
   from leapp import annotate

   class SimpleNet(nn.Module):
       def __init__(self):
           super().__init__()
           self.linear = nn.Linear(10, 5)

       def forward(self, x):
           return self.linear(x)

   model = SimpleNet()
   model.eval()

   @annotate.method(export_with="jit", node_name="preprocess")
   def preprocess(raw_data):
       """Normalize input data."""
       normalized = ((raw_data - raw_data.mean())
                     / (raw_data.std() + 1e-6))
       return normalized

   def main():
       leapp.start(
           name="complete_pipeline",
           save_path="./exports",
           verbose=True,
       )

       raw_input = torch.randn(1, 10)

       preprocessed = preprocess(raw_input)

       pred_traced = annotate.input_tensors("inference",
                                            {"features": preprocessed})
       predictions = model(pred_traced)
       annotate.output_tensors("inference",
                               {"predictions": predictions},
                               export_with="jit")

       pred_traced = annotate.input_tensors("postprocess",
                                            {"predictions": predictions})
       probabilities = torch.softmax(pred_traced, dim=1)
       final_output = torch.argmax(probabilities, dim=1)
       annotate.output_tensors("postprocess",
                               {"final_output": final_output},
                               export_with="jit")

       leapp.stop()
       leapp.compile_graph(visualize=True)

   if __name__ == "__main__":
       main()

This creates:

.. code-block:: text

   exports/complete_pipeline/
   |-- complete_pipeline.yaml    # Graph description
   |-- complete_pipeline.png     # Visualization
   |-- preprocess.pt             # Preprocessing model
   |-- inference.pt              # Inference model
   |-- postprocess.pt            # Postprocessing model
   `-- log.txt                   # Export log


Best practices
==============

#. Call methods in order: ``start()`` → trace → ``stop()`` →
   ``compile_graph()``.
#. Use meaningful node names --- makes debugging and deployment easier.
#. Always specify ``export_with`` explicitly.
#. Run feedback loops at least twice (see :doc:`/guides/graph`).
#. Use verbose mode during development to debug tracing issues.
#. Choose the right annotation method:

   * ``@annotate.method()`` for self-contained functions
   * ``input_tensors()`` / ``output_tensors()`` for operations spanning
     multiple functions, inline code, or dynamic scenarios

#. Always pair ``input_tensors()`` with ``output_tensors()``.
#. Do not mix ``TracedTensor`` values across node contexts.

See also
========

* :doc:`/getting_started` --- learn the basics
* :doc:`/guides/nodes` --- advanced node patterns
* :doc:`/guides/graph` --- graph and feedback operations
* :doc:`/guides/runtime` --- validation and runtime verification
* :doc:`/guides/semantics` --- semantic data annotation
