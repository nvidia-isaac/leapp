==========================
Semantic data annotation
==========================

This guide covers how to add semantic metadata to tensors using
:class:`~leapp.TensorSemantics`. Semantic annotations describe **what**
a tensor represents (e.g. joint positions, target torques) and provide
element-level naming, making the generated YAML specifications
self-documenting and enabling downstream consumers to interpret the data
correctly.

.. note::

   Semantic annotation is only available for
   :func:`~leapp.annotate.input_tensors` and
   :func:`~leapp.annotate.output_tensors`. It is **not** supported for
   :func:`~leapp.annotate.method` nodes.

Basic usage
===========

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

Semantic fields
===============

``kind``
--------

The ``kind`` field describes the **semantic role** of a tensor --- what
physical quantity or command it represents. LEAPP provides two separate
enums for inputs and outputs to clearly distinguish between observed
state and commanded targets. ``kind`` may also be any plain string when
you need a custom semantic label.

``InputKindEnum``
~~~~~~~~~~~~~~~~~

Used with :func:`~leapp.annotate.input_tensors`. These represent
**observed state** or **commanded references** flowing into a node.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Enum value
     - YAML string
     - Description
   * - ``JOINT_POSITION``
     - ``state/joint/position``
     - Observed joint positions (e.g. encoder readings)
   * - ``JOINT_VELOCITY``
     - ``state/joint/velocity``
     - Observed joint velocities
   * - ``JOINT_EFFORT``
     - ``state/joint/effort``
     - Observed joint effort
   * - ``BODY_POSE``
     - ``state/body/pose``
     - Observed body pose
   * - ``BODY_POSITION``
     - ``state/body/position``
     - Observed body position
   * - ``BODY_VEL``
     - ``state/body/velocity``
     - Observed body velocity
   * - ``BODY_ACC``
     - ``state/body/acceleration``
     - Observed body acceleration
   * - ``BODY_LINEAR_ACCELERATION``
     - ``state/body/linear_acceleration``
     - Body linear acceleration (e.g. from IMU)
   * - ``BODY_LINEAR_VELOCITY``
     - ``state/body/linear_velocity``
     - Body linear velocity
   * - ``BODY_ANGULAR_ACCELERATION``
     - ``state/body/angular_acceleration``
     - Body angular acceleration
   * - ``BODY_ANGULAR_VELOCITY``
     - ``state/body/angular_velocity``
     - Body angular velocity (e.g. gyroscope)
   * - ``BODY_ROTATION``
     - ``state/body/rotation``
     - Body rotation / orientation
   * - ``WRENCH``
     - ``state/wrench``
     - Observed wrench values
   * - ``VECTOR3D``
     - ``state/vector3d``
     - Generic 3D vector state
   * - ``COMMAND_JOINT_POSITION``
     - ``command/joint/position``
     - Commanded joint position reference
   * - ``COMMAND_JOINT_VELOCITY``
     - ``command/joint/velocity``
     - Commanded joint velocity reference
   * - ``COMMAND_BODY_ROTATION``
     - ``command/body/rotation``
     - Commanded body rotation reference
   * - ``COMMAND_BODY_VELOCITY``
     - ``command/body/velocity``
     - Commanded body velocity reference
   * - ``COMMAND_POSE``
     - ``command/body/pose``
     - Commanded body pose reference
   * - ``COMMAND_JOINT_TORQUES``
     - ``command/joint/torques``
     - Commanded joint torques reference

.. code-block:: python

   from leapp.utils.enums import InputKindEnum

   TensorSemantics("joint_pos", tensor, kind=InputKindEnum.JOINT_POSITION)
   TensorSemantics("imu_gyro", tensor,
                   kind=InputKindEnum.BODY_ANGULAR_VELOCITY)
   TensorSemantics("target_pos", tensor,
                   kind=InputKindEnum.COMMAND_JOINT_POSITION)

   # Custom string kinds are also allowed.
   TensorSemantics("terrain_latent", tensor,
                   kind="state/environment/terrain_embedding")

``OutputKindEnum``
~~~~~~~~~~~~~~~~~~

Used with :func:`~leapp.annotate.output_tensors`. These represent
**target commands** or **control outputs** produced by a node.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Enum value
     - YAML string
     - Description
   * - ``KP``
     - ``kp``
     - Proportional gain
   * - ``KD``
     - ``kd``
     - Derivative gain
   * - ``JOINT_POSITION``
     - ``target/joint/position``
     - Target joint position
   * - ``JOINT_VELOCITY``
     - ``target/joint/velocity``
     - Target joint velocity
   * - ``JOINT_TORQUES``
     - ``target/joint/torques``
     - Target joint torques
   * - ``JOINT_EFFORT``
     - ``target/joint/effort``
     - Target joint effort
   * - ``BODY_POSITION``
     - ``target/body/position``
     - Target body position
   * - ``BODY_LINEAR_ACCELERATION``
     - ``target/body/linear_acceleration``
     - Target body linear acceleration
   * - ``BODY_ORIENTATION``
     - ``target/body/orientation``
     - Target body orientation
   * - ``BODY_LINEAR_VELOCITY``
     - ``target/body/linear_velocity``
     - Target body linear velocity
   * - ``BODY_ANGULAR_ACCELERATION``
     - ``target/body/angular_acceleration``
     - Target body angular acceleration

.. code-block:: python

   from leapp.utils.enums import OutputKindEnum

   TensorSemantics("torques", action, kind=OutputKindEnum.JOINT_TORQUES)
   TensorSemantics("kp_gains", kp, kind=OutputKindEnum.KP)

.. note::

   LEAPP does not enforce using ``InputKindEnum`` only for inputs or
   ``OutputKindEnum`` only for outputs, but it is strongly recommended to
   follow this convention. The ``kind`` field accepts enum values and
   plain strings.

``element_names``
-----------------

The ``element_names`` field provides human-readable names for the
elements along each dimension of a tensor. The canonical format is
``list[list[str]]``, where the outer list corresponds to tensor
dimensions and each inner list names the elements in that dimension.
LEAPP also accepts several shorthand formats and normalizes them
automatically:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Input format
     - Normalized to
     - Use case
   * - ``"hip"``
     - ``[["hip"]]``
     - Single named element
   * - ``["hip", "knee", "ankle"]``
     - ``[["hip", "knee", "ankle"]]``
     - Flat list --- names for one dimension
   * - ``[["batch"], ["x", "y", "z"]]``
     - ``[["batch"], ["x", "y", "z"]]``
     - Already canonical --- per-dimension names
   * - ``[None, None, ["r", "g", "b"]]``
     - ``[None, None, ["r", "g", "b"]]``
     - Partial --- only name specific dimensions

.. code-block:: python

   # Name elements along the last dimension.
   TensorSemantics(
       "joint_pos", tensor,
       element_names=["hip", "knee", "ankle",
                      "shoulder", "elbow", "wrist"])

   # Name elements per dimension (e.g. for a [batch, 3] tensor).
   TensorSemantics(
       "position", tensor,
       element_names=[None, ["x", "y", "z"]])

   # Name a single element.
   TensorSemantics("gravity", tensor, element_names="z")


``TemporalAxis``
--------------------

Use ``TemporalAxis`` inside ``element_names`` to mark one tensor axis as
temporal and record the period between samples on that axis. This is useful for
chunked outputs such as an action tensor shaped ``[num_chunks, num_joints]``.

Place ``TemporalAxis`` directly in the outer ``element_names`` list. Do not
wrap it in a list. LEAPP serializes the temporal axis as the reserved
``__temporal_axis__`` sentinel and emits ``temporal_period_ms`` as a sibling
field on the tensor entry.

.. code-block:: python

   from leapp import TemporalAxis

   TensorSemantics(
       "actions",
       actions,
       kind=OutputKindEnum.JOINT_TORQUES,
       element_names=[
           TemporalAxis(period_ms=100),
           ["hip", "knee", "ankle"],
       ],
   )

.. code-block:: yaml

   - name: actions
     dtype: float32
     shape: [4, 3]
     type: tensor
     kind: target/joint/torques
     element_names:
     - __temporal_axis__
     - [hip, knee, ankle]
     temporal_period_ms: 100

Downstream consumers can find the temporal axis by locating
``__temporal_axis__`` in ``element_names``. LEAPP does not validate
``temporal_period_ms`` against ``GraphConfigs.frequency``.

.. warning::

   ``__temporal_axis__`` is reserved for LEAPP output. Use
   ``TemporalAxis`` in Python annotations rather than writing the
   sentinel directly. A tensor may contain at most one temporal axis marker.


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

