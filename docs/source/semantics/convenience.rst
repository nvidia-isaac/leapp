=======================
Convenience API
=======================

Quantity helpers return ``TensorSemantics`` objects for common quantities.
Pass them to ``annotate.input_tensors`` or ``annotate.output_tensors``.

.. warning::

   The convenience helpers and their quantity, axis, and spatial metadata are
   experimental. They are imported directly from ``leapp`` and may evolve.

Choosing a helper
=================

Choose the helper for :ref:`the quantity your tensor represents
<semantics-quantities>`:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Helpers
     - Use for
   * - ``joint_position``, ``joint_velocity``, ``joint_acceleration``, ``joint_effort``
     - Joint quantities, including acceleration in radians/s² or meters/s²
       according to joint type, and torque represented as joint effort.
   * - ``frame_pose``, ``frame_position``, ``frame_orientation``
     - A frame's pose, position, or orientation relative to a reference frame.
   * - ``frame_twist``, ``frame_linear_velocity``, ``frame_angular_velocity``
     - Combined or separate linear and angular velocities.
   * - ``frame_acceleration``, ``frame_linear_acceleration``, ``frame_angular_acceleration``
     - Combined or separate linear and angular accelerations.
   * - ``frame_wrench``, ``force_application_position``
     - Force and torque, optionally paired with the position where force acts.
   * - ``vector3d``
     - A generic three-dimensional vector when no more specific quantity fits.
   * - ``kp``, ``kd``
     - Proportional and derivative gains.
   * - ``image``
     - Image data with layout and encoding agreed with the consumer.
   * - ``point_cloud``
     - Cartesian XYZ point coordinates in meters. ``reference`` names their
       coordinate frame; use axes to describe point and component dimensions.

For example, construct annotations for measured joint positions and a desired
position input using the same helper:

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import torch
         from leapp import Axis, AxisKind, joint_position

         q = torch.tensor([0.1, -0.2])
         target = torch.tensor([0.4, 0.3])
         joint_axes = [Axis(AxisKind.ELEMENT, names=["shoulder", "elbow"])]

         # Describe the measured positions, then mark the desired positions as a setpoint.
         measured = joint_position("q", q, axes=joint_axes)
         desired = joint_position("q_target", target, axes=joint_axes, is_setpoint=True)

   .. tab:: YAML

      Input metadata after registration and export; dtype, shape, and type are omitted.

      .. code-block:: yaml

         inputs:
         - name: q
           kind: joint/position
           axes:
           - {kind: element, names: [shoulder, elbow]}
         - name: q_target
           kind: joint/position
           is_setpoint: true
           axes:
           - {kind: element, names: [shoulder, elbow]}

Helpers default to ``is_setpoint=None``, which omits the field from YAML.
Explicit ``True`` or ``False`` values are preserved; outputs reject ``True``.

These objects describe inputs; registration connects them to a node. See
:doc:`usage` for an explanation of setpoints and a complete controller export
using both annotations.

.. _semantics-helper-spatial-arguments:

Spatial choices
===============

Spatial arguments depend on the quantity:

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Helpers
     - ``reference``
     - ``expressed_in``
   * - Pose, position, orientation, point cloud
     - Frame name or ``None``
     - Not accepted, always expressed in the same frame as ``reference``
   * - Twist, linear/angular velocity, and acceleration
     - Frame name or ``None``
     - Frame name, ``None``, ``SELF``, or ``REFERENCE``
   * - Wrench, force-application position
     - Not accepted, does not describe a quantity relative to another frame
     - Frame name, ``None``, or ``SELF``

``None`` means global. See :doc:`spatial` for frame selectors and the optional
pairing of wrenches with force-application positions.

Validation and the generic constructor
======================================

For the helpers below, ``COMPONENT`` names must be a unique subset of the
allowed names and match the dimension size. Order the labels to match your values.
Omitted components remain unspecified; they do not imply zero.

.. list-table::
   :header-rows: 1

   * - Helper
     - Component names
   * - ``frame_pose``
     - ``x, y, z, qx, qy, qz, qw``
   * - ``frame_orientation``
     - ``qx, qy, qz, qw``
   * - ``frame_position``, ``force_application_position``, ``point_cloud``
     - ``x, y, z``
   * - ``frame_twist``
     - ``x, y, z, angular_x, angular_y, angular_z``
   * - ``frame_linear_velocity``
     - ``x, y, z``
   * - ``frame_angular_velocity``
     - ``angular_x, angular_y, angular_z``

For example, a planar position uses two components:

.. code-block:: python

   import torch
   from leapp import Axis, AxisKind, frame_position

   position = frame_position(
       "position", torch.tensor([1.0, 2.0]),
       axes=[Axis(AxisKind.COMPONENT, names=["x", "y"])],
   )

A planar twist similarly uses ``["x", "y", "angular_z"]``. Duplicate or
unrecognized component names raise ``ValueError``.

Use ``TensorSemantics`` directly for custom component conventions or ``extra``
metadata. It validates metadata structure without the helper-specific name
checks. Helpers describe existing values; they do not reshape, reorder, or
transform the tensor.
