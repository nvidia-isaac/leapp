=====
Usage
=====

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

Instead of passing raw tensors, wrap them in ``TensorSemantics`` objects.
Pass them as a single object or as a list:

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate, TensorSemantics
   from leapp.utils.enums import InputKindEnum, OutputKindEnum

   joint_pos = torch.randn(1, 12)
   joint_vel = torch.randn(1, 12)

   leapp.start("my_robot")

   traced_pos, traced_vel = annotate.input_tensors("policy", [
       TensorSemantics("joint_pos", joint_pos,
                       kind=InputKindEnum.JOINT_POSITION,
                       element_names=["hip_l", "knee_l", "ankle_l",
                                      "hip_r", "knee_r", "ankle_r",
                                      "shoulder_l", "elbow_l", "wrist_l",
                                      "shoulder_r", "elbow_r", "wrist_r"]),
       TensorSemantics("joint_vel", joint_vel,
                       kind=InputKindEnum.JOINT_VELOCITY),
   ])

   command = traced_pos + traced_vel

   annotate.output_tensors("policy", [
       TensorSemantics("command", command,
                       kind=OutputKindEnum.JOINT_TORQUES,
                       element_names=["hip_l", "knee_l", "ankle_l",
                                      "hip_r", "knee_r", "ankle_r",
                                      "shoulder_l", "elbow_l", "wrist_l",
                                      "shoulder_r", "elbow_r", "wrist_r"]),
   ])

   leapp.stop()
   leapp.compile_graph()

The generated YAML includes the semantic metadata:

.. code-block:: yaml

   models:
     policy:
       inputs:
       - name: joint_pos
         dtype: float32
         shape: [1, 12]
         type: tensor
         kind: state/joint/position
         element_names: [[hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r,
             shoulder_l, elbow_l, wrist_l, shoulder_r, elbow_r, wrist_r]]
       - name: joint_vel
         dtype: float32
         shape: [1, 12]
         type: tensor
         kind: state/joint/velocity
       outputs:
       - name: command
         dtype: float32
         shape: [1, 12]
         type: tensor
         kind: target/joint/torques
         element_names: [[hip_l, knee_l, ankle_l, hip_r, knee_r, ankle_r,
             shoulder_l, elbow_l, wrist_l, shoulder_r, elbow_r, wrist_r]]

Notes
=====

``extra``
---------

The ``extra`` field accepts a dictionary of additional semantic metadata.
Keys in ``extra`` are flattened into the generated YAML tensor entry rather
than nested under an ``extra`` key. Use this for downstream-specific fields
that LEAPP does not model directly, such as coordinate frames, external IDs,
units, or application-specific labels.

.. code-block:: python

   TensorSemantics(
       "joint_pos",
       tensor,
       kind=InputKindEnum.JOINT_POSITION,
       extra={"frame": "base", "units": "rad"},
   )

The generated YAML includes the extra fields at the same level as ``kind`` and
``element_names``:

.. code-block:: yaml

   - name: joint_pos
     dtype: float32
     shape: [1, 12]
     type: tensor
     kind: state/joint/position
     frame: base
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
       TensorSemantics("pos", tensor, kind=InputKindEnum.JOINT_POSITION),
   )

   # OK: list of TensorSemantics
   annotate.input_tensors("node", [
       TensorSemantics("pos", pos_tensor,
                       kind=InputKindEnum.JOINT_POSITION),
       TensorSemantics("vel", vel_tensor,
                       kind=InputKindEnum.JOINT_VELOCITY),
   ])

   # OK: regular dict (no semantic metadata)
   annotate.input_tensors("node", {"pos": pos_tensor, "vel": vel_tensor})

   # NOT supported: TensorSemantics inside a dict
   annotate.input_tensors("node", {
       "pos": TensorSemantics("pos", pos_tensor,
                              kind=InputKindEnum.JOINT_POSITION),
   })

   # NOT supported: mixing TensorSemantics and raw tensors
   # (use multiple calls to input_tensors in that case)
   annotate.input_tensors("node", [
       TensorSemantics("pos", pos_tensor,
                       kind=InputKindEnum.JOINT_POSITION),
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
   ``element_names``, temporal metadata from ``TemporalAxis``, and
   ``extra``) are optional. A ``TensorSemantics`` with no semantic fields
   behaves identically to passing a raw tensor with the same name.
#. **Extra fields are flattened** --- keys in ``extra`` become top-level YAML
   fields on the tensor entry. Avoid keys that collide with built-in fields
   such as ``name``, ``dtype``, ``shape``, ``type``, ``kind``,
   ``element_names``, ``temporal_period_ms``, or ``__temporal_axis__``.

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
does not validate ``TemporalAxis`` values against ``GraphConfigs.frequency``.

See :doc:`kind_element_names` for standard ``kind`` values and element naming,
and :doc:`temporal` for temporal axes on chunked tensors.

