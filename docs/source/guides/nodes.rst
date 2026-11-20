=============
Node patterns
=============

This guide covers advanced node patterns. The primary interface is
:func:`~leapp.annotate.input_tensors` and
:func:`~leapp.annotate.output_tensors`; use :func:`~leapp.annotate.method`
when a node fits cleanly inside one function and you want a shorthand.

One model for the whole pipeline
================================

A node is an export boundary, not a restriction on how you organize Python
code. Use the same node name from the first input annotation through the final
output annotation to export preprocessing, model inference, and action
processing together as one model. The captured computation can cross helper
functions and class methods, including supported NumPy-to-PyTorch conversions.
There is no requirement to wrap the pipeline in a single ``nn.Module`` or an
RL environment adapter.

Introduce additional node boundaries when you want separate model artifacts,
different export backends, or stage interfaces you can inspect independently.
LEAPP exports one model per node; splitting the pipeline is optional.

Distributed inputs: one node, many call sites
=============================================

Each node should have one finalizing ``output_tensors()`` call, but it may
have multiple ``input_tensors()`` calls with the same ``node_name``. This is
useful when a node collects data from multiple helpers or files before
producing an output.

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate

   def get_lidar_data(env):
       lidar_data = env.get("lidar_data")
       return annotate.input_tensors("sensor_fusion",
                                     {"lidar_data": lidar_data})

   def get_camera_features(env):
       camera_features = env.get("camera_features")
       return annotate.input_tensors("sensor_fusion",
                                     {"camera_features": camera_features})

   def run_pipeline(env, model):
       leapp.start(name="distributed_inputs_example")

       lidar = get_lidar_data(env)
       camera = get_camera_features(env)

       fused = torch.cat([lidar, camera], dim=-1)
       annotate.output_tensors("sensor_fusion",
                               {"model_input": fused},
                               export_with="jit")

       leapp.stop()
       leapp.compile_graph()

Both ``input_tensors()`` calls reference the same node name, so LEAPP
treats them as one node with multiple inputs.

``annotate.method()`` as a shorthand
====================================

:func:`~leapp.annotate.method` is a convenience wrapper over the
traced-tensor path.

.. code-block:: python

   import torch
   from leapp import annotate

   @annotate.method(export_with="jit", node_name="preprocess")
   def preprocess(obs):
       return (obs - obs.mean()) / (obs.std() + 1e-6)

Use it when:

* one function naturally defines the node input and output boundary
* parameter names and return values already match the API you want to export

Prefer ``input_tensors()`` / ``output_tensors()`` when:

* node logic spans multiple helpers
* you want explicit tensor names
* you need semantic metadata on tensors

.. _variable-dimensions:

Variable dimensions
===================

By default an exported node accepts only the tensor shapes seen during
capture. To accept a range of sizes along a dimension, declare that dimension
with ``Axis(variable=Variable(...))`` on the node input:

.. code-block:: python

   import torch

   import leapp
   from leapp import Axis, TensorSemantics, Variable, annotate

   length = Variable("length", min=4, max=32, multiple_of=2)

   leapp.start("tokens")
   tokens = annotate.input_tensors(
       "pairs",
       TensorSemantics(
           "tokens", torch.zeros(2, 8), axes=[None, Axis(variable=length)]
       ),
   )
   pairs = tokens.reshape(tokens.shape[0], tokens.shape[1] // 2, 2).sum(dim=-1)
   annotate.output_tensors("pairs", {"pairs": pairs}, export_with="pt2")
   leapp.stop()
   leapp.compile_graph()

The exported ``pairs`` node accepts any even ``length`` from 4 to 32. The first
dimension has no declaration and stays fixed at 2.

Declaring a variable axis
-------------------------

* ``Variable(name, min=1, max=2**31 - 1, multiple_of=1)`` names a dimension and
  its inclusive bounds. The range must admit at least two sizes.
* Axes that use the same ``Variable`` name must have the same size at run time,
  including across tensors and nodes. Reusing a name with different bounds is an
  error.
* ``Axis(variable=True)`` declares an independent dimension with the default
  bounds. ``Axis(variable=False)`` asserts that the dimension is fixed; export
  fails if the producer or the inferred output makes it variable.
* A variable axis cannot carry ``names``, because a fixed list of labels cannot
  describe a changing size.
* The tensor rank stays fixed. Dimensions without a declaration keep their
  captured size.

For a decorated function, pass the ``TensorSemantics`` as the argument. The
port takes the ``TensorSemantics`` name, and the function receives the tensor:

.. code-block:: python

   @annotate.method(export_with="pt2")
   def encode(image):
       features = image.mean(dim=-1)
       return features

   height = Variable("height", min=4, max=16)
   encode(
       TensorSemantics(
           "image", image, axes=[None, None, Axis(variable=height), None]
       )
   )

What stays variable during capture
----------------------------------

Inside a node with a variable input, ``tensor.shape[i]``, ``tensor.size(i)``,
and ``tensor.numel()`` return traced sizes. The exported graph records a traced
size when it is:

* combined with ``+``, ``-``, ``*``, ``//`` or ``%``;
* passed to an operation that also takes a traced tensor, such as
  ``x.reshape(...)``, ``x[:, :n]`` or ``torch.split(x, n)``;
* passed to ``torch.reshape``, ``torch.zeros``, ``torch.ones``, ``torch.full``,
  ``torch.arange``, ``torch.eye``, ``torch.linspace`` or ``torch.logspace``.

.. warning::

   Everywhere else a traced size is used as the size of the captured sample:

   * Python conversions and control flow: ``int(...)``, ``len(tensor)``,
     comparisons such as ``if tensor.shape[1] == 8``, and
     ``range(tensor.shape[1])``. The exported graph follows the captured branch
     or loop count for every input.
   * Indexing a constant tensor, such as ``table[:, :n]``. Register the constant
     first with ``annotate.register_buffer()`` and index the returned tensor.
   * Other factories that take no traced tensor, such as ``torch.rand`` and
     ``torch.empty``.

   ``torch.linspace`` and ``torch.logspace`` are recorded, but the exporter
   fixes their step count, so a variable count is rejected by
   ``compile_graph()``. Build the range from ``torch.arange`` instead.

Indexed assignment (``y[:, 0] = value``, ``y[mask] = value``) and ``copy_()``
into a tensor with a variable dimension are rejected by ``compile_graph()``,
which names the assignment. Build the result with ``torch.where`` instead.
Assignment into a tensor whose dimensions are all fixed is unaffected.

What ``compile_graph()`` checks
-------------------------------

LEAPP exports each node with the declared ranges and reports an error when an
operation fixes a declared dimension to one size or cannot accept the whole
range. Output dimensions are inferred from the graph and written to YAML, with
derived sizes such as ``length // 2`` expressed in terms of the declared
variables. Downstream nodes inherit the dimensions of the outputs they
consume. See :ref:`config-variable-dimensions` for the YAML fields.

Variable dimensions require an export backend (``export_with``) other than
``executorch``, which exports the captured shapes only, and Torch tensor ports:
a node with variable dimensions cannot have NumPy or Warp inputs or outputs.
NumPy operations inside such a node use the captured sizes. A tensor with a
variable dimension cannot be fed back to an earlier node.

``InferenceManager`` checks every input against the contract and raises
``ValueError`` for a wrong rank, a changed fixed dimension, a size outside the
bounds or not a multiple of ``multiple_of``, or tensors that disagree on a
shared ``Variable``.

.. _generated-shape-validation:

Generated shape validation
==========================

For a decorated function with variable axes, ``compile_graph(validate=True)``
automatically generates additional input samples. No callback is required:

.. code-block:: python

   @annotate.method(export_with="pt2")
   def encode(image):
       features = image.mean(dim=-1)
       return features

The default generator preserves dtype and device. It uses reproducible
uniform values in [0, 1) for floating-point inputs (real and imaginary parts for
complex inputs), and 0/1 values for integer and Boolean inputs. Inputs whose shapes
are unchanged retain copies of their captured values.

Use the optional ``validation_inputs`` callback when your inputs need specific
values or relationships, such as valid token IDs and matching attention masks:

.. code-block:: python

   def make_inputs(samples, shapes, seed):
       image = samples["image"]
       generator = torch.Generator(device=image.device).manual_seed(seed)
       return {"image": torch.rand(
           shapes["image"], dtype=image.dtype, device=image.device,
           generator=generator,
       )}

   @annotate.method(export_with="pt2", validation_inputs=make_inputs)
   def encode(image):
       features = image.mean(dim=-1)
       return features

Pass the usual ``TensorSemantics(..., axes=[...])`` when calling ``encode`` to
specify variable dimensions. The callback receives copies of explicit tensor
arguments, requested shapes keyed by parameter name, and a reproducible seed.
Tensors passed through ``*args`` use the keys ``arg_0``, ``arg_1``, and so on;
tensors passed through ``**kwargs`` use their keyword.
It must construct business-valid data, including relationships such as token
IDs and attention masks. Semantic port names do not change these callback keys.

LEAPP samples legal boundaries and interior sizes, changing independent symbols
one at a time while respecting shared dimensions and divisibility. It generates
at most five additional samples, each with at most 64 MiB of logical input data.
This is not a bound on model, intermediate, output, or total GPU memory. Duplicate
shapes are removed; omitted candidates are logged. If no additional shape fits
the budget, requested generated validation fails rather than silently passing.

Validation runs the original function with tracing disabled and compares it to
the saved, reloaded artifact using the existing tolerances. Inputs are copied
independently; Python, NumPy and Torch random states on input devices are restored.
A mismatch, invalid generated input, or reference execution error fails validation;
``strict=True`` raises. If generation or original-function execution fails,
validation reports that it could not complete and suggests a custom callback;
this does not establish that the export is incorrect. Static annotated functions
and manually annotated nodes retain cached-sample validation.
``validate=False`` disables both cached and generated validation, including callbacks
and reference replay.

This first version supports top-level Torch tensor arguments, including
those passed through ``*args`` and ``**kwargs``. Outputs may be a tensor or a
tuple, list or dict of tensors.
Other arguments may be scalars or lists, tuples and dicts that contain only
scalars; each replay receives its own copy. A call cannot be replayed when an
argument holds a tensor inside a container, is any other object, or when the
function creates additional node inputs. In that case automatic generation logs
a warning and only the captured samples are compared; with a
``validation_inputs`` callback, validation fails. Pass ``validation_inputs=False``
to turn generated validation off for one function, for example when it has side
effects and must not run again during ``compile_graph()``. Tensor defaults remain
constants. Functions and bound methods must
be deterministic and replayable with stable configuration/model state and no
external side effects; LEAPP does not snapshot arbitrary objects or model weights.
Reference callables and sample copies are held only in memory and released after
validation or node reset; they are not written into YAML or model artifacts.
Passing these samples does not prove correctness for all admitted inputs.

Nested data connections
=======================

LEAPP can track connections through complex nested structures. Each
individual tensor within nested dicts, lists, or custom objects is tracked
separately.

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate

   @annotate.method(export_with="jit", node_name="process_robot_state")
   def process_robot_state(state_dict):
       processed_state = {
           'position': state_dict['position'] * 2.0,
           'velocity': state_dict['velocity'] + 1.0,
           'sensors': {
               'lidar': state_dict['sensors']['lidar'].mean(dim=1),
               'camera': state_dict['sensors']['camera'].flatten(),
           },
       }
       return processed_state

   def main():
       leapp.start(name="nested_data_example")

       robot_state = {
           'position': torch.tensor([1.0, 2.0, 3.0]),
           'velocity': torch.tensor([0.5, 0.5, 0.0]),
           'sensors': {
               'lidar': torch.randn(360, 3),
               'camera': torch.randn(3, 224, 224),
           },
       }

       processed = process_robot_state(robot_state)

       # LEAPP creates connections per individual tensor path:
       #   robot_state['position']            -> processed['position']
       #   robot_state['velocity']            -> processed['velocity']
       #   robot_state['sensors']['lidar']    -> processed['sensors']['lidar']
       #   robot_state['sensors']['camera']   -> processed['sensors']['camera']

       p = annotate.input_tensors("decision_maker", {"processed": processed})

       position_factor = p['position'].norm()
       velocity_factor = p['velocity'].sum()
       sensor_confidence = p['sensors']['lidar'].std()

       action = torch.stack(
           [position_factor, velocity_factor, sensor_confidence])
       annotate.output_tensors("decision_maker",
                               {"action": action},
                               export_with="jit")

       leapp.stop()
       leapp.compile_graph()

How LEAPP handles nested structures
-----------------------------------

When LEAPP sees a complex nested data structure (dicts, lists, tuples) as
an input or output, it automatically:

#. **Flattens the structure.** Each individual tensor within the nested
   structure is extracted and tracked separately. For example,
   ``state_dict['sensors']['lidar']`` becomes a distinct input named
   ``state_dict_sensors_lidar``.
#. **Generates an auto-interface.** LEAPP generates wrapper code that
   accepts flat tensors and reconstructs them into the nested structure
   your original code expects on input, then unpacks the returned nested
   structure into flat tensors on output.
#. **Tracks connections at tensor level.** This flattening enables LEAPP
   to track data flow between nodes at the individual tensor level, not at
   the parameter level.

The result: all exported nodes have simple, flat tensor interfaces, while
your original code continues to use nested structures naturally. This
guarantees:

* consistent tensor-level connection tracking across all nodes
* compatibility with deployment frameworks that expect flat tensor I/O
* clear visibility into exactly which tensors flow between which nodes

In the code above, the ``process_robot_state`` node's exported model has
four separate tensor inputs (``state_dict_position``,
``state_dict_velocity``, ``state_dict_sensors_lidar``,
``state_dict_sensors_camera``) rather than a single complex dictionary
input.

Summary
=======

* Distributed traced inputs let one node collect data from multiple call
  sites.
* ``annotate.method()`` is a shorthand for self-contained functions.
* Nested data structures are automatically tracked.

LEAPP's goal is to capture your computational graph accurately. Being
explicit about data dependencies and naming will help avoid most issues.
