=============
LEAPP Runtime
=============

LEAPP provides a simple re-entry Python runtime called
:class:`~leapp.InferenceManager`. Use it to load an exported YAML bundle, run
the graph directly from Python, and inspect graph inputs, outputs, and feedback
state before connecting the bundle to a larger deployment runtime.

.. code-block:: python

   from leapp import InferenceManager

   manager = InferenceManager("my_graph/my_graph.yaml")

   print(manager.inputs)
   print(manager.outputs)

   mock_inputs = manager.get_mock_input()
   outputs = manager.run_policy(mock_inputs)

Use ``InferenceManager`` to:

* load the generated YAML and referenced model artifacts
* inspect graph-level inputs and outputs
* create mock inputs for a quick smoke run
* execute the exported graph from Python
* inspect or override feedback inputs

.. note::

   For ONNX models, ``InferenceManager`` uses CPU-safe ``onnxruntime`` by
   default. Install ``onnxruntime-gpu`` to enable CUDA execution; when the CUDA
   provider is available, LEAPP prefers it automatically for ONNX-backed nodes.

Running With Your Own Values
============================

``manager.inputs`` lists the external input keys that ``run_policy()`` expects,
using ``node_name/input_name`` format. Build an input dictionary with those keys
and ``torch.Tensor`` values that match the generated config shapes and dtypes:

.. code-block:: python

   from leapp import InferenceManager

   manager = InferenceManager("my_graph/my_graph.yaml")

   inputs = manager.get_mock_input()
   inputs["obs_processor/joint_pos"] = live_joint_pos
   inputs["obs_processor/joint_vel"] = live_joint_vel

   outputs = manager.run_policy(inputs)

``get_mock_input()`` is a convenient starting point because it creates tensors
with the expected shape, dtype, and device. Deployment code can also build the
dictionary from scratch; the keys should match ``manager.inputs``.

Reading Outputs
===============

``run_policy()`` returns a dictionary of final graph outputs. Keys use
``node_name/output_name`` format and match ``manager.outputs``:

.. code-block:: python

   outputs = manager.run_policy(inputs)
   action = outputs["policy/joint_targets"]

The same values are also cached under ``manager.value_dict["==out=="]`` after a
run.

Inspecting and Overriding Node Values
=====================================

``InferenceManager`` stores node-boundary values in ``manager.value_dict``. The
first level is a node name, and the second level is an input or routed output
port name. Final graph outputs are stored under the special ``"==out=="`` key.
Use this to inspect values moving between exported nodes:

.. code-block:: python

   print(manager.value_dict["obs_processor"]["joint_pos"])
   print(manager.value_dict["policy"]["obs_features"])
   print(manager.value_dict["==out=="]["policy/joint_targets"])

You can overwrite any node input buffer before a run. This is useful for
seeding or replacing feedback state after ``InferenceManager`` has initialized
feedback inputs from ``pipeline.initial_values``:

.. code-block:: python

   manager.set_input_value("policy", "hidden", hidden_override)

``value_dict`` exposes values at LEAPP node boundaries; it does not expose
intermediate operations inside a compiled TorchScript, ONNX, or ``pt2`` model.

Feedback State
==============

For graphs with feedback:

* feedback inputs are auto-initialized from the exported safetensors file
  when available
* you can inspect feedback targets via ``manager.feedback_inputs``
* you can manually override any feedback input with
  ``set_input_value(...)``

.. code-block:: python

   manager = InferenceManager("my_graph/my_graph.yaml")
   manager.set_input_value("stateful_node", "h", torch.zeros(1, 32))

See :doc:`generated_configs` for the YAML contract that deployment runtimes consume.