=============
Kind and Axes
=============

``Kind`` describes the quantity, independently of input/output direction.
Input annotations default to measured state; set ``is_setpoint=True`` for a
setpoint input. Output annotations reject ``is_setpoint=True``. Custom string
kinds and ``kind=None`` are also accepted.

.. list-table::
   :header-rows: 1

   * - Kind
     - YAML value
     - Optional helper
   * - ``JOINT_POSITION``
     - ``joint/position``
     - ``joint_position(...)``
   * - ``JOINT_VELOCITY``
     - ``joint/velocity``
     - ``joint_velocity(...)``
   * - ``JOINT_EFFORT``
     - ``joint/effort``
     - ``joint_effort(...)``
   * - ``FRAME_POSE``
     - ``frame/pose``
     - ``frame_pose(...)``
   * - ``FRAME_POSITION``
     - ``frame/position``
     - ``frame_position(...)``
   * - ``FRAME_ORIENTATION``
     - ``frame/orientation``
     - ``frame_orientation(...)``
   * - ``FRAME_TWIST``
     - ``frame/twist``
     - ``frame_twist(...)``
   * - ``FRAME_LINEAR_VELOCITY``
     - ``frame/linear_velocity``
     - ``frame_linear_velocity(...)``
   * - ``FRAME_ANGULAR_VELOCITY``
     - ``frame/angular_velocity``
     - ``frame_angular_velocity(...)``
   * - ``FRAME_ACCELERATION``
     - ``frame/acceleration``
     - ``frame_acceleration(...)``
   * - ``FRAME_LINEAR_ACCELERATION``
     - ``frame/linear_acceleration``
     - ``frame_linear_acceleration(...)``
   * - ``FRAME_ANGULAR_ACCELERATION``
     - ``frame/angular_acceleration``
     - ``frame_angular_acceleration(...)``
   * - ``FRAME_WRENCH``
     - ``frame/wrench``
     - ``frame_wrench(...)``
   * - ``VECTOR3D``
     - ``vector3d``
     - ``vector3d(...)``
   * - ``KP``
     - ``kp``
     - ``kp(...)``
   * - ``KD``
     - ``kd``
     - ``kd(...)``
   * - ``IMAGE``
     - ``image``
     - ``image(...)``

``Axis``
--------

``axes`` is either ``None`` or a list with one entry per tensor dimension.
Entries can be ``Axis`` objects or ``None``. ``Axis.kind`` and ``Axis.names``
are independently optional. Supplied names must match the dimension size.

.. code-block:: python

   from leapp import Axis, AxisKind, Kind, TensorSemantics

   annotation = TensorSemantics(
       "q", q, kind=Kind.JOINT_POSITION,
       axes=[Axis(AxisKind.ROBOT), Axis(AxisKind.ELEMENT, names=["hip", "knee"])],
   )

Standard kinds are ``ROBOT``, ``ELEMENT``, ``COMPONENT``, and ``TIME``.
They serialize as ``robot``, ``element``, ``component``, and ``time``.
Custom strings such as ``Axis(kind="token")`` remain available. Strings
matching standard values receive the same validation as their enum equivalents.
``Axis(names=["a", "b"])`` supplies labels without declaring a kind.
See :doc:`temporal` for time axes.

Spatial metadata
----------------

``reference`` names the frame against which pose or motion is measured.
``expressed_in`` identifies the coordinate basis: a literal frame name,
``ExpressionFrame.SELF`` (the element's own frame), or
``ExpressionFrame.REFERENCE`` (the frame named by ``reference``).
Twists and wrenches default to ``SELF``. A literal frame named ``self`` is
not the same as the enum selector.

.. code-block:: yaml

   kind: frame/twist
   reference: base
   expressed_in: {selector: self}
   axes:
   - {kind: element, names: [tool]}
   - {kind: component, names: [vx, vy, vz, wx, wy, wz]}

A literal expression frame serializes as ``expressed_in: {frame: world}``.
Poses use ``reference`` only; wrenches use ``expressed_in`` only. Twist linear
velocity is evaluated at the element frame's origin; wrench moments are taken
about that origin. These annotations do not transform tensor values.

Convenience helpers
-------------------

Every built-in Kind has a helper exported from ``leapp``. Helpers return
ordinary ``TensorSemantics`` and accept ``axes=None`` and
``is_setpoint=False``. Spatial helpers require ``reference`` except for
``frame_wrench``. Motion and wrench helpers accept ``expressed_in`` with
``SELF`` as the default; pose, position, and orientation helpers do not.

.. code-block:: python

   from leapp import frame_pose, frame_twist, image, joint_position

   target = joint_position("target", q_target, is_setpoint=True)
   pose = frame_pose("pose", tool_pose, reference="world")
   twist = frame_twist("twist", tool_twist, reference="base")
   camera = image("camera", pixels)

Helpers additionally validate supplied pose, orientation, position, and velocity
component labels; permutations are allowed. The generic constructor performs
structural metadata validation, allowing custom conventions. Missing axes are
not inferred, and importers decide what metadata they require. Images do not
assume channel order, colour encoding, or normalization. Acceleration and wrench
component conventions remain consumer-defined in this POC.

Migration
---------

``Kind`` replaces ``InputKindEnum`` and ``OutputKindEnum``. Remove direction
prefixes, use ``FRAME_*`` instead of ``BODY_*``, and use joint effort for torque.
``axes`` replaces ``element_names``; include ``None`` for unspecified dimensions.
Temporal metadata now lives on ``Axis(AxisKind.TIME, period_ms=...)``.
YAML schema version 1.4 identifies this change.
