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

* combined with another integer by ``+``, ``-``, ``*``, ``//`` or ``%``;
* passed to an operation that also takes a traced tensor, such as
  ``x.reshape(...)``, ``x[:, :n]`` or ``torch.split(x, n)``;
* passed to ``torch.tensor``, ``torch.zeros``, ``torch.ones``, ``torch.full``,
  ``torch.arange`` or ``torch.eye``.

.. warning::

   Everywhere else a traced size is used as the size of the captured sample:

   * Python conversions and control flow: ``int(...)``, ``len(tensor)``,
     comparisons such as ``if tensor.shape[1] == 8``, and
     ``range(tensor.shape[1])``. The exported graph follows the captured branch
     or loop count for every input.
   * Other arithmetic on a size, such as ``/``, ``**``, ``abs()`` and
     ``round()``, and arithmetic between a size and a constant tensor.
   * Indexing a constant tensor, such as ``table[:, :n]``. Register the constant
     first with ``annotate.register_buffer()`` and index the returned tensor.
   * Other factories that take no traced tensor, such as ``torch.rand`` and
     ``torch.empty``.
   * NumPy operations inside the node.

   LEAPP does not detect these uses. The export fails when the captured size
   conflicts with a variable dimension. Otherwise the node is exported and
   returns wrong results at other sizes. :ref:`generated-shape-validation`
   reports this for a decorated function, but not for a manually annotated
   node.

Variable dimensions between nodes
---------------------------------

Declare a variable dimension once, on the input where the tensor enters the
pipeline. A node input that is connected to another node's output inherits
that output's variable dimensions, with their name, bounds and
``multiple_of``, and needs no declaration. A size the producer derives, such
as ``2 * length + 1``, is inherited as that expression.

Nothing else passes between nodes. Code between two nodes is not exported and
runs once, on the captured sample:

* A tensor computed between nodes is a new pipeline input. Declare its variable
  dimensions again on that input; otherwise they are fixed.
* A size read between nodes, such as ``n = y.shape[1]``, is the captured size.
  A later node that uses it may return wrong results at other sizes, and
  validation cannot detect this. To avoid it, do not carry a size from one
  node into another. Read it inside the node that uses it.

What ``compile_graph()`` checks
-------------------------------

LEAPP exports each node with the declared ranges and reports an error when the
exporter fixes a declared dimension to one size or reports a narrower range.
Output dimensions are inferred from the graph and written to YAML, with
derived sizes such as ``length // 2`` expressed in terms of the declared
variables. Downstream nodes inherit the dimensions of the outputs they
consume. See :ref:`config-variable-dimensions` for the YAML fields.

``InferenceManager`` checks every input against the contract and raises
``ValueError`` for a wrong rank, a changed fixed dimension, a size outside the
bounds or not a multiple of ``multiple_of``, or tensors that disagree on a
shared ``Variable``.

Limits
------

``compile_graph()`` rejects a node with variable dimensions that has:

* no export backend (``export_with``), or ``executorch``, which exports the
  captured shapes only;
* a NumPy or Warp input or output;
* an indexed assignment (``y[:, 0] = value``, ``y[mask] = value``) or
  ``copy_()`` into a tensor with a variable dimension, even when no output uses
  the assigned tensor. The error names the assignment. Build the result with
  ``torch.where`` instead; assignment into a tensor whose dimensions are all
  fixed is unaffected;
* ``torch.linspace`` or ``torch.logspace`` with a variable count. Build the
  range from ``torch.arange`` instead;
* a tensor with a variable dimension fed back to an earlier node. A
  fixed-shape tensor can be fed back;
* an input whose dimension its producer computes with ``//`` or ``%``, or from
  more than one variable. Such a dimension can still be a graph output. A node
  input accepts a declared dimension or a linear size such as
  ``2 * length + 1``.

.. _generated-shape-validation:

Generated shape validation
==========================

For a decorated function with variable axes, ``compile_graph(validate=True)``
generates additional input samples. It runs the original function on them with
tracing disabled and compares the results with the saved, reloaded artifact,
using the ``rtol`` and ``atol`` of ``compile_graph()``.

LEAPP changes one variable at a time, to its bounds, to sizes near the captured
one and to sizes in between, and keeps shared variables equal and
``multiple_of`` satisfied. It generates at most five additional samples, each
with at most 64 MiB of input tensors. Memory used by the model itself is not
limited. Duplicate shapes are removed and omitted candidates are logged. If no
additional shape fits the limit, validation fails. Passing these samples does
not prove every size in the range.

The default generator preserves dtype and device. It uses reproducible
uniform values in [0, 1) for floating-point inputs (real and imaginary parts for
complex inputs), and 0/1 values for integer and Boolean inputs. Inputs whose shapes
are unchanged retain copies of their captured values.

Pass a ``validation_inputs`` callback when the inputs need specific values or
relationships, such as valid token IDs and matching attention masks:

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

The callback receives copies of the explicit tensor arguments, the requested
shapes and a reproducible seed, and returns the generated tensors. Tensors and
shapes are keyed by function parameter name, not port name. Tensors passed
through ``*args`` use the keys ``arg_0``, ``arg_1``, and so on; tensors passed
through ``**kwargs`` use their keyword.

Validation fails on a mismatch or an invalid generated input; ``strict=True``
raises. It also fails when generating inputs or running the original function
raises. The message then says that validation could not complete and suggests
a callback, since the export itself may be correct.

Generated validation has these requirements:

* Tensor arguments are top-level Torch tensors, including those passed through
  ``*args`` and ``**kwargs``. Other arguments may be scalars, or lists, tuples
  and dicts that contain only scalars; each replay receives its own copy.
  Tensor defaults remain constants.
* Outputs are a tensor or a tuple, list or dict of tensors.
* The function runs again during ``compile_graph()``, so it must return the
  same result for the same inputs and have no side effects. LEAPP does not copy
  model weights or other objects. Pass ``validation_inputs=False`` to turn
  generated validation off for a function that must not run again.

A call cannot be replayed when an argument holds a tensor inside a container
or is any other object, or when the function creates additional node inputs.
Automatic generation then logs a warning and compares only the captured
samples; with a ``validation_inputs`` callback, validation fails.

Inputs are copied for each replay, and the Python, NumPy and Torch random
states on the input devices are restored afterwards. LEAPP keeps the function
and the copies in memory until validation or a node reset, and writes neither
to the bundle.

Static annotated functions and manually annotated nodes keep cached-sample
validation. ``validate=False`` disables both cached and generated validation.

.. note::

   A node built with ``annotate.input_tensors()`` and
   ``annotate.output_tensors()`` has no function to run again, so its saved
   model is compared at the captured sizes only. A backend can still convert an
   operation in a way that fails or returns different values at other sizes.
   Run the saved model at the sizes you need, or use ``annotate.method``.

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
