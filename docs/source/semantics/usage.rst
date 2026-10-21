======================
Tensor Semantics Usage
======================


.. warning::

   ``Kind``, ``AxisKind``, ``ExpressionFrame``, ``Axis``, the quantity helpers,
   and the ``axes``, ``is_setpoint``, ``reference``, and ``expressed_in`` fields
   are experimental. They are imported directly from ``leapp``. Their contract
   may evolve in future releases.

.. note::

   The interfaces presented here are intended to replace ``InputKindEnum``,
   ``OutputKindEnum``, ``element_names``, and ``TemporalAxis``. They are
   experimental; the legacy APIs remain fully supported and are not deprecated.
   See :doc:`legacy` for their reference and optional migration guidance.

A tensor's shape tells a runtime how many values to expect, but not what those
values mean. For example, a tensor shaped ``[1, 2]`` might contain:

* measured or desired joint positions;
* joint velocities or efforts;
* proportional or derivative gains;
* many other quantities.

A controller also needs to know which column belongs to which joint before
it can interpret those values correctly.

Semantic annotations record this meaning alongside the tensor in the exported
YAML. They help a robot runtime, simulator, or controller connect model ports
to physical quantities. They do not change the computation or transform values.
Use them when a downstream consumer needs this information; ordinary named
raw tensors are sufficient when names, shapes, and dtypes already define the
interface.

Semantic annotations describe four parts of that contract:

* **Quantity:** ``Kind`` identifies what the tensor represents; see
  :ref:`semantics-quantities` for built-in, custom, and unspecified kinds.
* **Purpose:** a setpoint is a desired value that a controller aims to reach
  or maintain. For example, a controller may receive both a joint's measured
  position and the position it should move to. Both have the same ``Kind``,
  but the desired-position input uses ``is_setpoint=True`` to distinguish it
  from the measurement. Only inputs may use ``is_setpoint=True``.
* **Layout:** ``Axis`` identifies the elements, components, or time samples
  along each tensor dimension. A joint-position tensor shaped ``[2, 3]`` could
  describe two robots with three joints each (axes ordered as system, joint),
  or three robots with two joints each (axes ordered as joint, system). The
  order of the ``Axis`` entries tells the consumer how to interpret the
  dimensions; it must match the tensor's actual layout.
* **Frames:** spatial metadata identifies the reference frame and coordinate
  basis for quantities such as position and velocity. For example, a tool's
  position relative to the robot base differs from its position relative to
  the room. Likewise, moving along the tool's x-axis may mean a different
  direction from moving along the room's x-axis. ``reference`` identifies
  what a position or motion is relative to; ``expressed_in`` identifies whose
  coordinate axes describe the components. This tells the consumer how to
  interpret the values without guessing the frame convention.

The :doc:`convenience` helpers provide a concise way to construct these
annotations. Start with the controller example below, then see
:doc:`kind_element_names` for the meaning of individual fields.

.. note::

   Semantic annotation is only available for
   :func:`~leapp.annotate.input_tensors` and
   :func:`~leapp.annotate.output_tensors`. It is **not** supported for
   :func:`~leapp.annotate.method` nodes.

General Usage
=============

This example exports a simplified joint controller:

* The robot has two joints: shoulder and elbow.
* Inputs are measured joint positions, measured joint velocities, and desired
  joint positions (the position setpoint).
* The output is a joint effort command.
* Each tensor has shape ``[1, 2]``, representing ``[system, joint]``, and shares
  the same axis labels.
* Quantity helpers return ``TensorSemantics`` objects. We pass the inputs as
  a list and the output as a single object.

For each joint, the controller computes:

.. math::

   \tau = K_p (q_{\mathrm{target}} - q) - K_d \dot{q}

Here, :math:`q` is measured position, :math:`\dot{q}` is measured velocity,
and :math:`q_{\mathrm{target}}` is desired position. The proportional term
pulls the joint toward its target; the velocity term adds damping, with zero
desired velocity. We use :math:`K_p = 20` and :math:`K_d = 2` for both joints.
For these rotational joints, the effort :math:`\tau` represents torque.

This is a simplified example for demonstrating annotation and export. The gains
are illustrative; the computation omits robot dynamics, actuator limits, and
other considerations needed for a deployed controller.

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import torch
         import leapp
         from leapp import annotate, Axis, AxisKind
         from leapp import joint_position, joint_velocity, joint_effort

         # Sample values for tracing: one robot, with shoulder then elbow values.
         q = torch.tensor([[0.1, -0.2]])
         dq = torch.zeros(1, 2)
         q_target = torch.tensor([[0.4, 0.3]])
         # Describe what each dimension indexes, using the same order as the tensors.
         joint_axes = [
             Axis(AxisKind.SYSTEM, names=["robot0"]),
             Axis(AxisKind.ELEMENT, names=["shoulder", "elbow"]),
         ]

         # Begin capturing the computation in a graph named joint_controller.
         leapp.start("joint_controller")
         # Declare the controller inputs and their meaning. Only q_target is desired.
         # Use the returned traced tensors in the computation below.
         q_in, dq_in, target_in = annotate.input_tensors("controller", [
             joint_position("q", q, axes=joint_axes),
             joint_velocity("dq", dq, axes=joint_axes),
             joint_position("q_target", q_target, axes=joint_axes, is_setpoint=True),
         ])

         # Compute effort from position error and velocity damping, per joint.
         effort = 20.0 * (target_in - q_in) - 2.0 * dq_in
         # Publish the effort output and select TorchScript (jit) for this node.
         annotate.output_tensors(
             "controller",
             joint_effort("effort", effort, axes=joint_axes),
             export_with="jit",
         )
         # Finish tracing, then write the model and YAML metadata to disk.
         leapp.stop()
         leapp.compile_graph(visualize=False)

   .. tab:: YAML

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
               - {kind: system, names: [robot0]}
               - {kind: element, names: [shoulder, elbow]}
             - name: dq
               dtype: float32
               shape: [1, 2]
               type: tensor
               kind: joint/velocity
               axes:
               - {kind: system, names: [robot0]}
               - {kind: element, names: [shoulder, elbow]}
             - name: q_target
               dtype: float32
               shape: [1, 2]
               type: tensor
               kind: joint/position
               is_setpoint: true
               axes:
               - {kind: system, names: [robot0]}
               - {kind: element, names: [shoulder, elbow]}
             outputs:
             - name: effort
               dtype: float32
               shape: [1, 2]
               type: tensor
               kind: joint/effort
               axes:
               - {kind: system, names: [robot0]}
               - {kind: element, names: [shoulder, elbow]}

The input/output lists determine direction. Only ``q_target`` is marked as a
setpoint. Other annotations leave ``is_setpoint=None``, the default, so the
field is omitted. Set ``is_setpoint=False`` to explicitly record that an input
is not a setpoint; YAML then includes ``is_setpoint: false``. The same Kind can
describe measured and requested positions. Outputs accept ``False`` or ``None``
but reject ``True``.

Helpers are optional. For example, the first input can also be written as:

.. code-block:: python

   from leapp import TensorSemantics, Kind

   TensorSemantics("q", q, kind=Kind.JOINT_POSITION, axes=joint_axes)

Use ``axes=None`` or individual ``None`` entries when the consumer already knows
part of the layout. They are not needed for this fully labeled joint interface.

Custom metadata (``extra``)
===========================

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

``TensorSemantics`` are passed as a **single object**, a **list**, or a **tuple**. They
cannot be placed inside a dict --- use the standard dict format for raw
tensors and a list or tuple for ``TensorSemantics``. The only supported
top-level formats are:

* a dict of named raw tensors
* a single ``TensorSemantics``
* a list or tuple of ``TensorSemantics``

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
   in the same list or tuple.
#. **No dict wrapping** --- ``TensorSemantics`` must be passed directly
   or in a list or tuple. Each carries its own name, so the dict key is
   unnecessary.
#. **Name uniqueness** --- each ``TensorSemantics`` name must be unique
   within each list or tuple of annotations. Duplicate names in that
   collection raise an error.
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

Graph-level semantics are independent of tensor-level temporal metadata.

See :doc:`kind_element_names` for quantities and axes, :doc:`spatial` for
spatial metadata, then :doc:`convenience` for helper selection.
