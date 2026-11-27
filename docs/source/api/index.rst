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

Initialize and start LEAPP graph tracing.

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
       patching=None,
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
* ``non_traced`` (list[str], optional): Node names to exclude from export.
  They are still traced, still capture I/O, contribute to graph
  connectivity, and appear in the YAML. Defaults to ``None``.
* ``max_cached_io`` (int, optional): Maximum number of captured input/output
  samples per node, including the initial trace. Repeated calls can record
  consecutive stateful transitions, including state inputs and updates.
  Defaults to ``5``.
* ``global_patching`` (bool, optional): Enable all process-global patches,
  including user-defined replacements. Defaults to ``True``.
* ``patching`` (sequence[FunctionPatch] | None, optional): User-defined
  replacements for module-level functions that LEAPP cannot otherwise trace.
  Defaults to ``None``.

  .. warning::

     Disabling ``global_patching`` disables patches that allow traced
     tensors to pass through ``torch.from_numpy()`` and related numpy
     interop functions, and skips definitions supplied through ``patching``.
     Unsupported operations can then become invisible to LEAPP.

User-defined function patches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``FunctionPatch`` maps a module attribute to a user-provided traceable
replacement:

.. code-block:: python

   import scipy.ndimage
   import leapp

   leapp.start(
       "pipeline",
       patching=[
           leapp.FunctionPatch(
               scipy.ndimage,
               "gaussian_filter",
               leapp_gaussian_filter,
           ),
       ],
   )

During the tracing session, calls containing active traced data use the
replacement. Calls containing only ordinary values, or traced values from a
finished node, use the original function. Positional arguments, keyword
arguments, and nested lists, tuples, and dictionaries are searched. User
patches require ``global_patching=True``.

The replacement receives traced arguments directly and must build its result
from operations LEAPP already traces. It must accept a compatible signature
and return the output structure, shape, and dtype expected by the caller.
A replacement must not call the patched function.

Call ``leapp.stop()`` before starting a session with a different patch list.
Stopping restores the exact function object captured when the patch was
installed.

.. warning::

   Patching a module attribute does not update references imported before
   ``leapp.start()``. Call ``scipy.ndimage.gaussian_filter(...)`` rather than a
   previously captured ``from scipy.ndimage import gaussian_filter`` alias.
   Installed patches are process-global and can affect other threads.

See :doc:`../guides/patching` for a walkthrough of choosing and registering a
replacement in an existing workflow.

For a complete export and replay using a user-defined patch, run
``python examples/function_patch.py`` from the repository root. It patches
``cv2.cvtColor`` once for a nested stereo-image workflow that converts both
camera frames from BGR to RGB, then crosses from NumPy into torch to extract
features. The adapter supports only three-channel BGR-to-RGB conversion without
a destination buffer. Replay uses new frames and compares against the original
OpenCV workflow. The example also verifies that ``leapp.stop()`` restores the
original OpenCV function. Install ``opencv-python-headless`` to run it.

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
       rtol: float = 1e-3,
       atol: float = 1e-5,
       strict: bool = True,
       graph_configs: GraphConfigs | None = None,
   )

Parameters
~~~~~~~~~~

* ``visualize`` (bool, optional): Generate a graph visualization.
  Defaults to ``True``. Requires Python 3.11 or later; earlier versions
  emit a warning and skip visualization. On supported Python versions,
  visualization errors stop compilation.
* ``verbose`` (bool | None, optional): Override verbose logging for the
  compile step. ``None`` leaves the current setting unchanged.
* ``validate`` (bool, optional): Validate exported models against all cached
  input/output samples, including exported state updates. Each sample uses
  its recorded inputs and state; see :ref:`stateful-validation` for the
  distinction from sequential runtime replay. Validation then runs sample 0
  again and fails if the result differs, which usually means the export
  keeps undeclared state between calls. Defaults to ``True``.
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
#. Generate ``{name}.png`` if ``visualize=True`` and Python 3.11 or later is
   running.
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

* Compiled models --- one file per node (``.pt``, ``.pt2``, ``.onnx``,
  ``.pte``), plus a ``.ptd`` weights file for an ExecuTorch CUDA node.
* ``{name}.yaml`` --- model descriptions, pipeline connections
  (``data_flow`` and ``feedback_flow``), and system information.
* ``{name}.png`` --- graph visualization (when ``visualize=True`` on
  Python 3.11+).

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
     warp version: null
     leapp version: "0.8.1"
     leapp config version: "1.4"
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

.. warning::

   ``Kind``, ``AxisKind``, ``ExpressionFrame``, ``Axis``, the quantity helpers,
   and the ``axes``, ``is_setpoint``, ``reference``, and ``expressed_in`` fields
   are experimental. They are imported directly from ``leapp``. Their contract
   may evolve in future releases.

``TensorSemantics`` describes a named tensor for
:func:`~leapp.annotate.input_tensors` and
:func:`~leapp.annotate.output_tensors`. Quantity helpers return the same class.

.. code-block:: python

   TensorSemantics(name=None, ref=None, kind=None, element_names=None, extra=None,
                   *, axes=None, is_setpoint=None, reference=None,
                   expressed_in=None)

* ``name`` supplies the port name; ``ref`` supplies the tensor value.
* ``kind`` accepts ``Kind``, ``InputKindEnum``, ``OutputKindEnum``,
  custom strings, or ``None``. The legacy input/output enums are deprecated
  and emit ``FutureWarning`` when used in an annotation.
* ``axes`` accepts ``None`` or a list of ``Axis`` / ``None`` entries, one per
  tensor dimension. Names must match the dimension size.
* ``is_setpoint`` accepts ``True``, ``False``, or ``None`` (the default).
  ``True`` marks an incoming desired value. Outputs accept ``False`` or ``None``
  but reject ``True``. Explicit boolean values are written to YAML; ``None``
  leaves the field unspecified and omitted.
* ``reference`` names the frame against which pose or motion is measured,
  or the coordinate frame for a point cloud.
* ``expressed_in`` accepts a literal frame name, ``ExpressionFrame.SELF``,
  ``ExpressionFrame.REFERENCE``, or ``None`` (global).
* ``extra`` is a dictionary flattened into the tensor YAML entry. Use it only
  for agreed consumer-specific metadata and avoid built-in field names.
* ``element_names`` supplies deprecated element labels and cannot accompany
  ``axes``. It emits ``FutureWarning``; use ``Axis`` labels instead.
  See :doc:`/semantics/legacy` for accepted formats and migration guidance.

.. warning::

   ``extra`` fields are non-standard, project-defined metadata. Downstream
   deployment frameworks should not be expected to understand or support them.
   Use ``extra`` only for project-specific integration data, and prefer
   standard LEAPP semantic fields whenever possible.

Pass a single annotation or a homogeneous list or tuple directly, never inside
a dict or mixed with raw tensors. See :doc:`/semantics/usage` for examples.

``Axis`` and quantity helpers
-----------------------------

.. code-block:: python

   Axis(kind=None, names=None, period_ms=None)
   joint_position(name, ref, *, axes=None, is_setpoint=None)
   joint_acceleration(name, ref, *, axes=None, is_setpoint=None)
   point_cloud(name, ref, *, reference=None, axes=None, is_setpoint=None)
   frame_pose(name, ref, *, axes=None, is_setpoint=None, reference=None)
   frame_twist(name, ref, *, axes=None, is_setpoint=None,
               reference=None, expressed_in=None)
   frame_wrench(name, ref, *, axes=None, is_setpoint=None, expressed_in=None)

``AxisKind`` defines ``SYSTEM``, ``ELEMENT``, ``COMPONENT``, and ``TIME``.
Axis kinds also accept custom strings. Time axes require a finite, positive
``period_ms``; other axis kinds cannot supply a period.

See :doc:`/semantics/kind_element_names` for the ``Kind`` catalog,
:doc:`/semantics/spatial` for spatial fields,
:doc:`/semantics/convenience` for helpers and component validation, and
:ref:`semantics-time-axes` for time axes.

Annotations
===========

``annotate.set_semantics()``
----------------------------

.. code-block:: python

   annotate.set_semantics(node_name, *, inputs=None, outputs=None, states=None,
                          allow_unregistered_ports=False)

* Declares or replaces metadata independently of tensor values. Call after
  ``start()`` and before ``compile_graph()``, including after ``stop()``.
* ``inputs``, ``outputs``, and ``states`` accept lists or tuples of named
  ``TensorSemantics`` descriptors. Input and output names are independent.
* By default, names must match registered ports in the specified direction.
  Unknown names raise ``ValueError``.
* ``allow_unregistered_ports=True`` permits predefinition and creates the node
  if needed, without adding executable ports. Unresolved names fail compilation.
* ``states`` describes both state ports; both must exist unless predefinition
  is enabled. Use ``state_tensors()`` and ``update_state()`` to create feedback.
* Each valid call replaces the full description of supplied ports, clearing
  omitted fields and preserving other ports. Inline annotations share this store
  and replacement policy.
* Missing or duplicate names, incompatible layouts, and output setpoints fail
  atomically. Known shapes validate immediately; otherwise, checks wait for
  registration.
* Metadata stays in the node's direction/name map. YAML generation looks it up
  for surviving ports. Registration does not consume it.
* Set metadata before compilation, which may rename ports. Later edits do not
  update saved artifacts. A new ``start()`` resets the store.

See :doc:`/semantics/usage` for examples.

``annotate.input_tensors()``
----------------------------

.. code-block:: python

   traced_tensors = annotate.input_tensors(node_name: str, tensors)

* Creates or reuses the named node during an active session.
* ``tensors`` accepts a dict of named raw tensors, a ``TensorSemantics``, or a
  homogeneous list/tuple of descriptors. Bare tensors and unnamed collections
  are unsupported.
* Returns one traced value for one input, otherwise a tuple in declaration order.
* Records operations across helper functions. Do not mix traced values from
  different node contexts in one operation.
* Pair with ``output_tensors()`` to finalize capture.

``annotate.output_tensors()``
-----------------------------

.. code-block:: python

   annotate.output_tensors(node_name: str, tensors, static_outputs=None, **kwargs)

* Finalizes the existing named node and builds its exportable graph.
* ``tensors`` uses the same naming forms as ``input_tensors()``.
* ``static_outputs`` declares constant outputs using the same naming forms.
  Values must be plain ``torch.Tensor`` objects, not traced tensors.
* ``export_with`` selects the backend: ``"jit"``, ``"onnx"``, ``"pt2"``
  (``"exported-program"``), or ``"executorch"``. Variants include
  ``"jit-script"``, ``"jit-trace"``, ``"onnx-dynamo"``, and
  ``"onnx-torchscript"``. See :doc:`/guides/export`.
* ``backend_params`` supplies backend-specific options.
* Prunes unused inputs, returns unwrapped output values, and ends tracing for
  that node's current capture.

``annotate.method()``
---------------------

.. code-block:: python

   @annotate.method(**params)
   def your_function(...):
       ...

* Traces a self-contained function or method as one ``TracedTensorNode``.
* ``node_name`` defaults to the function name. ``export_with`` and
  ``backend_params`` select the backend and its options.
* Uses the function signature for inputs and the return value for outputs.
  Unpassed tensor-valued defaults become node-local constants.
* Other annotations may declare additional inputs or states inside the function.
* Runs normally when interpretation is disabled.

``annotate.state_tensors()``
----------------------------

.. code-block:: python

   annotate.state_tensors(node_name: str, tensors)

* Creates or reuses the named node and registers state inputs.
* ``tensors`` accepts a dict of names to initial tensors or named
  ``TensorSemantics`` descriptors with tensor values. Descriptors apply to both
  state ports. Nested state structures are unsupported.
* Returns one traced value for one state, otherwise a tuple in declaration order.
* Only states passed to ``update_state()`` become feedback outputs; others
  remain regular inputs.

``annotate.update_state()``
---------------------------

.. code-block:: python

   annotate.update_state(node_name: str, tensors)

* Updates states declared by ``state_tensors()`` on an existing node.
* ``tensors`` accepts a dict of state names to updated traced values or named
  ``TensorSemantics`` descriptors. Descriptors replace output metadata only.
* Checks state shape and dtype. Updated states become feedback outputs.
* Returns the supplied values: one value for one state, otherwise a tuple in
  declaration order.

``annotate.register_buffer()``
------------------------------

.. code-block:: python

   annotate.register_buffer(node_name: str,
                            tensors: torch.Tensor | list | tuple | dict)

* Registers outside tensors written and read by the node, such as accumulators.
  Call after ``input_tensors()`` on every node execution.
* ``tensors`` accepts a Torch tensor, list/tuple, or named dict. Unnamed buffers
  receive generated names.
* Returns traced values backed by those tensors: one value for one buffer,
  otherwise a tuple in input order. Use the returned values inside the node.
* Captures in-place resets on every execution. Without registration, resets
  before the first use of traced data are not exported and replay can accumulate
  across calls.
* For Warp, register ``wp.to_torch(array)`` and use
  ``wp.from_torch(result, dtype=array.dtype)`` in its place.
* Returns tensors unchanged outside tracing.

``annotate.module()``
---------------------

.. code-block:: python

   annotate.module(node_name: str, model: torch.nn.Module,
                   buffer_names: list[str] | None = None)

* Tracks registered module buffers on an existing node. Call after
  ``input_tensors()`` and before the forward pass.
* ``buffer_names`` selects a subset; omitted, all registered buffers are tracked.
* Temporarily injects traced buffers, emits reassigned buffers as feedback,
  and preserves untouched buffers as constants.
* Tracks reassignment, not in-place updates such as ``self.h.copy_(...)``.

``annotate.mirror_leapp_tags()``
--------------------------------

.. code-block:: python

   target = annotate.mirror_leapp_tags(source, target)

* Transfers tracing metadata between equivalent copies made outside standard
  PyTorch operations. ``source`` must be a registered node output.
* Requires exactly matching values; a mismatch logs an error and raises.
* Copies the producing node, output port, and tracing proxy to ``target``.
* Returns the target. Torch and Warp targets are upgraded in place; raw NumPy
  targets require assigning the return value.
* Returns the target unchanged outside graph interpretation.

See :doc:`/guides/buffers` for between-node copy examples.

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
     - Run the full pipeline. ``inputs`` is a dict that maps
       ``"node_name/input_name"`` keys to ``torch.Tensor`` values. Returns a dict of
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
* ``InferenceManager`` currently runs exported or referenced ``jit``,
  ``pt2``, and ``onnx`` models, and exported ``pte`` models.
* For ``pt2`` nodes, runtime inputs should be on the same device as the
  loaded exported program. ``get_mock_input()`` already allocates on the
  node device.
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
   |-- complete_pipeline.png     # Graph visualization
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
* :doc:`/guides/debugging` --- validation and debugging
* :doc:`/leapp_runtime` --- LEAPP Python runtime
* :doc:`/semantics/usage` --- semantic data annotation
