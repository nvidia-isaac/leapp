=============
Kind and Axes
=============

A deployment consumer needs to distinguish both the quantity in a tensor and
its layout. A ``[2, 3]`` tensor, for example, could hold three joint values for
two robots or three position components for two tools. ``Kind`` describes the
quantity; ``Axis`` describes what each dimension indexes. Spatial fields add
the coordinate-frame information needed to interpret poses and motion; see
:doc:`spatial`.

Use :doc:`convenience` to construct common annotations with named helpers, or
supply these fields directly to ``TensorSemantics`` as shown below.

.. warning::

   ``Kind``, ``AxisKind``, ``ExpressionFrame``, ``Axis``, the quantity helpers,
   and the ``axes``, ``is_setpoint``, ``reference``, and ``expressed_in`` fields
   are experimental. They are imported directly from ``leapp``. Their contract
   may evolve in future releases.

.. _semantics-quantities:

``Kind``: what your tensor represents
-------------------------------------

``Kind`` identifies the physical quantity, such as joint position, joint
velocity, or effort. Choose it from the values the tensor actually contains. For example,
an ``actions`` tensor might contain desired positions, velocities, or efforts;
its name alone does not tell you which.

You can describe the quantity in three ways:

* **A built-in quantity:** use a ``Kind`` member such as
  ``Kind.JOINT_POSITION`` for a standard physical quantity. The table below
  lists the available members and their YAML values.
* **A custom string:** use a non-empty string such as
  ``kind="terrain_embedding"`` when your consumer understands an
  application-specific quantity. Agree its meaning with the consumer;
  LEAPP does not define the meaning of custom labels.

  .. note::

     Use a ``Kind`` member or convenience helper for a built-in quantity.
     A plain string alone is treated as a custom label; matching a built-in
     kind's spelling does not activate its defaults or validation.
* **No quantity:** omit ``kind`` or pass ``kind=None`` to leave it unspecified.
  This is useful for normalized observations or vectors containing several
  different measurements that do not share one physical kind. You can still
  supply axis labels and other metadata without declaring a quantity.

``Kind`` describes the quantity, independently of input/output direction.
Both measured and desired joint positions use ``Kind.JOINT_POSITION``.

.. list-table::
   :header-rows: 1

   * - Kind
     - YAML value
   * - ``JOINT_POSITION``
     - ``joint/position``
   * - ``JOINT_VELOCITY``
     - ``joint/velocity``
   * - ``JOINT_EFFORT``
     - ``joint/effort``
   * - ``FRAME_POSE``
     - ``frame/pose``
   * - ``FRAME_POSITION``
     - ``frame/position``
   * - ``FRAME_ORIENTATION``
     - ``frame/orientation``
   * - ``FRAME_TWIST``
     - ``frame/twist``
   * - ``FRAME_LINEAR_VELOCITY``
     - ``frame/linear_velocity``
   * - ``FRAME_ANGULAR_VELOCITY``
     - ``frame/angular_velocity``
   * - ``FRAME_ACCELERATION``
     - ``frame/acceleration``
   * - ``FRAME_LINEAR_ACCELERATION``
     - ``frame/linear_acceleration``
   * - ``FRAME_ANGULAR_ACCELERATION``
     - ``frame/angular_acceleration``
   * - ``FRAME_WRENCH``
     - ``frame/wrench``
   * - ``FORCE_APPLICATION_POSITION``
     - ``frame/force_application_position``
   * - ``VECTOR3D``
     - ``vector3d``
   * - ``KP``
     - ``kp``
   * - ``KD``
     - ``kd``
   * - ``IMAGE``
     - ``image``

.. _semantics-axes:
.. _semantics-time-axes:

``Axis``: what each dimension of your tensor represents
-------------------------------------------------------

An ``Axis`` describes one dimension of a tensor. It has two parts that tell
a consumer what the entries along that dimension represent:

* ``kind`` identifies the category, using an ``AxisKind`` such as ``SYSTEM``
  or ``ELEMENT``.
* ``names`` labels the individual entries in their tensor order. For example,
  joint names tell the consumer which value belongs to which joint.

Here are two small examples:

.. code-block:: python

   from leapp import Axis, AxisKind

   # A dimension of size 2: entry 0 is robot0, entry 1 is robot1.
   systems = Axis(kind=AxisKind.SYSTEM, names=["robot0", "robot1"])

   # A dimension of size 3: entries are shoulder, elbow, then wrist.
   joints = Axis(kind=AxisKind.ELEMENT, names=["shoulder", "elbow", "wrist"])

For a joint-position tensor shaped ``[2, 3]``, ``axes=[systems, joints]`` means
that rows select robots and columns select joints. The names describe the
order already present in the tensor; they do not reorder values.

Choose the ``AxisKind`` that matches what the dimension indexes:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Axis kind
     - Meaning and example
   * - ``SYSTEM``
     - Complete systems, such as robots, vehicles, or machines. In a
       ``[2, 3]`` joint-position tensor with layout ``[system, joint]``, the
       first dimension selects one of two robots. ``ELEMENT`` identifies
       parts within each system. An ordinary batch dimension need not
       identify systems; leave its kind unspecified if it does not.
   * - ``ELEMENT``
     - Physical elements such as joints or tools. In that same tensor, the
       second dimension selects one of three joints, such as shoulder,
       elbow, and wrist.
   * - ``COMPONENT``
     - Parts of a quantity. A tool-position tensor shaped ``[2, 3]`` can use
       ``[element, component]``: two tools, each with ``x``, ``y``, and ``z``
       position components.
   * - ``TIME``
     - Samples along a sequence. An action tensor shaped ``[4, 3]`` can use
       ``[time, element]``: four samples for three joints. A time axis also
       requires ``period_ms``; for example, ``Axis(AxisKind.TIME, period_ms=100)``
       records 100 milliseconds between samples. The period must be finite
       and positive, and each tensor may have at most one time axis.

The order of ``axes`` must match the tensor's dimensions. Labels describe the
order of values within a dimension:

* Supply one ``Axis`` or ``None`` per dimension, or use ``axes=None`` to leave
  the whole layout unspecified.
* ``Axis.kind`` and ``Axis.names`` are independently optional. For example,
  ``Axis(names=["a", "b"])`` supplies labels without declaring an axis kind.
* Supplied names must match the dimension size. Omitted labels are not inferred.
* Custom axis kinds such as ``Axis(kind="token")`` are also accepted. Strings
  matching standard values (i.e. ``"system"``) receive the same validation as their enum equivalents (``AxisKind.SYSTEM``).
* Use either ``axes`` or ``element_names``; they cannot be supplied together.

For one robot with two joints, the annotation can look like this:

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import torch
         from leapp import Axis, AxisKind, joint_position

         q = torch.zeros(1, 2)
         annotation = joint_position(
             "q", q,
             axes=[
                 Axis(AxisKind.SYSTEM),
                 Axis(AxisKind.ELEMENT, names=["hip", "knee"]),
             ],
         )

   .. tab:: YAML

      Tensor entry after registration and export.

      .. code-block:: yaml

         name: q
         dtype: float32
         shape: [1, 2]
         type: tensor
         kind: joint/position
         axes:
         - {kind: system}
         - {kind: element, names: [hip, knee]}

For a sequence of effort commands, use a time axis.
Here, each row contains one sample for the same three joints:

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import torch
         from leapp import Axis, AxisKind, joint_effort

         actions = torch.zeros(4, 3)
         annotation = joint_effort(
             "actions", actions,
             axes=[
                 Axis(AxisKind.TIME, period_ms=100),
                 Axis(AxisKind.ELEMENT, names=["hip", "knee", "ankle"]),
             ],
         )

   .. tab:: YAML

      When registered with ``annotate.output_tensors``, this produces:

      .. code-block:: yaml

         - name: actions
           dtype: float32
           shape: [4, 3]
           type: tensor
           kind: joint/effort
           axes:
           - {kind: time, period_ms: 100}
           - {kind: element, names: [hip, knee, ankle]}

As a more complex example, here positions for two tools on two robots over four time samples are recorded:

* **System:** two robots, ``robot0`` and ``robot1``.
* **Time:** four samples, spaced 100 milliseconds apart.
* **Element:** two tools per robot, ``left_tool`` and ``right_tool``.
* **Component:** the ``x``, ``y``, and ``z`` coordinates of each tool.

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import torch
         from leapp import Axis, AxisKind, frame_position

         # Shape: [system, time, element, component].
         positions = torch.zeros(2, 4, 2, 3)
         annotation = frame_position(
             "tool_positions", positions,
             reference="world",
             axes=[
                 Axis(AxisKind.SYSTEM, names=["robot0", "robot1"]),  # 2 robots
                 Axis(AxisKind.TIME, period_ms=100),                 # 4 samples
                 Axis(AxisKind.ELEMENT, names=["left_tool", "right_tool"]),  # 2 tools
                 Axis(AxisKind.COMPONENT, names=["x", "y", "z"]),   # 3 coordinates
             ],
         )

   .. tab:: YAML

      Tensor entry after registration and export.

      .. code-block:: yaml

         name: tool_positions
         dtype: float32
         shape: [2, 4, 2, 3]
         type: tensor
         kind: frame/position
         axes:
         - {kind: system, names: [robot0, robot1]}
         - {kind: time, period_ms: 100}
         - {kind: element, names: [left_tool, right_tool]}
         - {kind: component, names: [x, y, z]}
         spatial:
           reference: world

For example, ``positions[1, 2, 0, :]`` holds robot1's left-tool XYZ position at
the third sample, relative to the frame named ``world``.

The axis order follows your data layout; LEAPP does not require systems or time
to come first. A tensor stored as ``[time, system, element, component]`` would
have shape ``[4, 2, 2, 3]`` and put the ``TIME`` axis before the ``SYSTEM`` axis.
Reordering the axis descriptions alone does not move the tensor's values.

Axis kinds serialize as ``system``, ``element``, ``component``, and ``time``.
The time period and graph frequency describe different things:

* ``Axis.period_ms`` describes spacing between samples inside a tensor.
* ``GraphConfigs.frequency`` describes graph execution frequency in Hz.

LEAPP does not validate one against the other or schedule runtime calls from
this metadata.
