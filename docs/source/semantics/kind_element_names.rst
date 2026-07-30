====================
Kind & Element Names
====================

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

See :doc:`usage` for general ``TensorSemantics`` usage patterns and
:doc:`temporal` for temporal axis metadata.

