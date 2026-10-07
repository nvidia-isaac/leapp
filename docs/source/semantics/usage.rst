======================
Tensor Semantics Usage
======================

Semantic annotations add runtime-facing meaning to tensors using
:class:`~leapp.TensorSemantics`. Semantic annotations describe **what**
a tensor represents (e.g. joint positions, target torques) and provide
element-level naming, making the generated YAML specifications
self-documenting and enabling downstream consumers to interpret the data
correctly. They are useful when an exported graph must be connected to a
robot runtime, simulator, message bus, or controller that needs to know not
just tensor shapes, but what those tensors mean.

.. note::

   Semantic annotation is only available for
   :func:`~leapp.annotate.input_tensors` and
   :func:`~leapp.annotate.output_tensors`. It is **not** supported for
   :func:`~leapp.annotate.method` nodes.

General Usage
=============

The helpers return ``TensorSemantics`` objects, which can be passed individually
or as a list. This example exports a small joint controller for one two-joint
robot: measured positions and velocities, a position setpoint, and an effort
output. Each tensor has shape ``[robot, joint]`` and shares the same axes.

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate, Axis, AxisKind
   from leapp import joint_position, joint_velocity, joint_effort

   q = torch.tensor([[0.1, -0.2]])
   dq = torch.zeros(1, 2)
   q_target = torch.tensor([[0.4, 0.3]])
   joint_axes = [
       Axis(AxisKind.ROBOT, names=["robot0"]),
       Axis(AxisKind.ELEMENT, names=["shoulder", "elbow"]),
   ]

   leapp.start("joint_controller")
   q_in, dq_in, target_in = annotate.input_tensors("controller", [
       joint_position("q", q, axes=joint_axes),
       joint_velocity("dq", dq, axes=joint_axes),
       joint_position("q_target", q_target, axes=joint_axes, is_setpoint=True),
   ])

   # A simple proportional/derivative controller.
   effort = 20.0 * (target_in - q_in) - 2.0 * dq_in
   annotate.output_tensors(
       "controller",
       joint_effort("effort", effort, axes=joint_axes),
       export_with="jit",
   )
   leapp.stop()
   leapp.compile_graph(visualize=False)

The input/output lists determine direction. Only ``q_target`` is marked as a
setpoint; measured inputs use the default ``is_setpoint=False``. The same Kind
can describe measured and requested positions, and outputs need no extra flag.

The generated YAML includes these tensor descriptions (model parameters and
pipeline wiring are omitted here):

.. code-block:: yaml

   models:
     controller:
       inputs:
       - name: q
         dtype: float32
         shape: [1, 2]
         type: tensor
         kind: joint/position
         axes:
         - {kind: robot, names: [robot0]}
         - {kind: element, names: [shoulder, elbow]}
       - name: dq
         dtype: float32
         shape: [1, 2]
         type: tensor
         kind: joint/velocity
         axes:
         - {kind: robot, names: [robot0]}
         - {kind: element, names: [shoulder, elbow]}
       - name: q_target
         dtype: float32
         shape: [1, 2]
         type: tensor
         kind: joint/position
         is_setpoint: true
         axes:
         - {kind: robot, names: [robot0]}
         - {kind: element, names: [shoulder, elbow]}
       outputs:
       - name: effort
         dtype: float32
         shape: [1, 2]
         type: tensor
         kind: joint/effort
         axes:
         - {kind: robot, names: [robot0]}
         - {kind: element, names: [shoulder, elbow]}

Helpers are optional. For example, the first input can also be written as:

.. code-block:: python

   from leapp import TensorSemantics, Kind

   TensorSemantics("q", q, kind=Kind.JOINT_POSITION, axes=joint_axes)

Use ``axes=None`` or individual ``None`` entries when the consumer already knows
part of the layout. They are not needed for this fully labelled joint interface.

Notes
=====

``extra``
---------

The ``extra`` field accepts a dictionary of additional semantic metadata.
Keys in ``extra`` are flattened into the generated YAML tensor entry rather
than nested under an ``extra`` key. Use this for downstream-specific fields
that LEAPP does not model directly, such as external IDs, units, or
application-specific labels. Use the standard spatial fields for frame metadata.

.. code-block:: python

   TensorSemantics(
       "joint_pos",
       tensor,
       kind=Kind.JOINT_POSITION,
       extra={"sensor_id": "encoders", "units": "rad"},
   )

The generated YAML includes the extra fields at the same level as ``kind`` and
``axes``:

.. code-block:: yaml

   - name: joint_pos
     dtype: float32
     shape: [1, 12]
     type: tensor
     kind: joint/position
     sensor_id: encoders
     units: rad

Unknown semantic keys applied internally through ``update_semantics()`` are
also stored in ``extra`` and serialized this way.

.. warning::

   ``extra`` fields are non-standard, project-defined metadata. Downstream
   deployment frameworks should not be expected to understand or support them.
   Use ``extra`` only for project-specific integration data, and prefer
   standard LEAPP semantic fields whenever possible.

Passing conventions
===================

``TensorSemantics`` are passed as a **single object** or a **list**. They
cannot be placed inside a dict --- use the standard dict format for raw
tensors and the list format for ``TensorSemantics``. The only supported
top-level formats are:

* a dict of named raw tensors
* a single ``TensorSemantics``
* a list of ``TensorSemantics``

Bare top-level tensors and other unnamed top-level collections are not
supported.

.. code-block:: python

   # OK: single TensorSemantics
   annotate.input_tensors(
       "node",
       TensorSemantics("pos", tensor, kind=Kind.JOINT_POSITION),
   )

   # OK: list of TensorSemantics
   annotate.input_tensors("node", [
       TensorSemantics("pos", pos_tensor,
                       kind=Kind.JOINT_POSITION),
       TensorSemantics("vel", vel_tensor,
                       kind=Kind.JOINT_VELOCITY),
   ])

   # OK: regular dict (no semantic metadata)
   annotate.input_tensors("node", {"pos": pos_tensor, "vel": vel_tensor})

   # NOT supported: TensorSemantics inside a dict
   annotate.input_tensors("node", {
       "pos": TensorSemantics("pos", pos_tensor,
                              kind=Kind.JOINT_POSITION),
   })

   # NOT supported: mixing TensorSemantics and raw tensors
   # (use multiple calls to input_tensors in that case)
   annotate.input_tensors("node", [
       TensorSemantics("pos", pos_tensor,
                       kind=Kind.JOINT_POSITION),
       vel_tensor,
   ])

Limitations
===========

#. ``input_tensors`` and ``output_tensors`` only --- semantic annotations
   are not available for :func:`~leapp.annotate.method` nodes. These
   nodes derive their I/O descriptions automatically from function
   signatures and traced values.
#. **No mixing** --- when using ``TensorSemantics``, all items must be
   ``TensorSemantics``. You cannot mix raw tensors and ``TensorSemantics``
   in the same list.
#. **No dict wrapping** --- ``TensorSemantics`` must be passed directly
   or in a list. Each carries its own name, so the dict key is
   unnecessary.
#. **Name uniqueness** --- each ``TensorSemantics`` name must be unique
   within the same node's inputs (or outputs). Duplicate names raise an
   error.
#. **Semantic fields are optional** --- all semantic fields (``kind``,
   ``axes``, spatial metadata, and
   ``extra``) are optional. A ``TensorSemantics`` with no semantic fields
   behaves identically to passing a raw tensor with the same name.
#. **Extra fields are flattened** --- keys in ``extra`` become top-level YAML
   fields on the tensor entry. Avoid keys that collide with built-in fields
   such as ``name``, ``dtype``, ``shape``, ``type``, ``kind``,
   ``axes``, ``reference``, ``expressed_in``, or ``is_setpoint``.

Graph-level semantics
=====================

Use ``GraphConfigs`` for metadata that applies to the whole exported LEAPP
bundle rather than to a specific tensor. ``GraphConfigs`` is passed to
:func:`~leapp.compile_graph`, after tracing has stopped.

.. code-block:: python

   from leapp import GraphConfigs

   leapp.compile_graph(
       graph_configs=GraphConfigs(
           frequency=50,
           extra={"skip-first-run": True},
       ),
   )

``frequency`` describes the graph-level execution frequency in Hz. ``extra``
works like ``TensorSemantics.extra`` within ``configs``: keys are flattened into
the generated ``pipeline.configs`` entry rather than nested under an ``extra``
key.

.. code-block:: yaml

   pipeline:
     data_flow: {}
     feedback_flow: {}
     inputs: {}
     outputs: {}
     configs:
       frequency: 50
       skip-first-run: true

Graph-level semantics are independent of tensor-level temporal metadata. LEAPP
does not validate time-axis periods against ``GraphConfigs.frequency``.

See :doc:`kind_element_names` for standard ``kind`` values and element naming,
and :doc:`temporal` for temporal axes on chunked tensors.

