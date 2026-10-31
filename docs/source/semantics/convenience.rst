=======================
Convenience API
=======================

Quantity helpers make common annotations easier to read at a model boundary.
Instead of repeating ``TensorSemantics`` and a ``Kind`` member for every port,
you can name the quantity directly: ``joint_position``, ``frame_twist``, or
``image``. Each helper returns an ordinary ``TensorSemantics`` object that you
pass to ``annotate.input_tensors`` or ``annotate.output_tensors``.

.. warning::

   The convenience helpers and their quantity, axis, and spatial metadata are
   experimental. They are imported directly from ``leapp`` and may evolve.

Choosing a helper
=================

Once you have identified :ref:`the quantity your tensor represents
<semantics-quantities>`, use the matching helper below. Each name describes the
annotation it creates.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Helpers
     - Use for
   * - ``joint_position``, ``joint_velocity``, ``joint_effort``
     - Joint quantities, including torque represented as joint effort.
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

Spatial helpers only accept the spatial arguments appropriate for their
quantity. The table distinguishes which arguments are available from which
values they accept:

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Helpers
     - ``reference``
     - ``expressed_in``
   * - Pose, position, orientation
     - Frame name or ``None``
     - Not accepted, always expressed in the same frame as ``reference``
   * - Twist, linear/angular velocity, and acceleration
     - Frame name or ``None``
     - Frame name, ``None``, ``SELF``, or ``REFERENCE``
   * - Wrench, force-application position
     - Not accepted, does not describe a quantity relative to another frame
     - Frame name, ``None``, or ``SELF``

Both arguments default to ``None`` wherever they are accepted. 
Wrench and force-application-position helpers reject ``REFERENCE`` because
they do not accept a ``reference`` argument.

Validation and the generic constructor
======================================

Helpers enforce standard component names for the common quantities listed
below. If you provide names on a ``COMPONENT`` axis, they must contain exactly
the names listed for that helper, with no missing, repeated, or additional
names. Other names raise ``ValueError``. This gives consumers a consistent
vocabulary for these quantities.

Any ordering of the required names is accepted. You should make sure that order
matches what you actually have in your tensor: LEAPP cannot check the physical
meaning of the values, so incorrect ordering can still pass validation and
export successfully.

.. list-table::
   :header-rows: 1

   * - Helper
     - Component names
   * - ``frame_pose``
     - ``x, y, z, qx, qy, qz, qw``
   * - ``frame_orientation``
     - ``qx, qy, qz, qw``
   * - ``frame_position``, ``force_application_position``
     - ``x, y, z``
   * - ``frame_twist``
     - ``vx, vy, vz, wx, wy, wz``
   * - ``frame_linear_velocity``
     - ``vx, vy, vz``
   * - ``frame_angular_velocity``
     - ``wx, wy, wz``

For a position, the required names are ``x``, ``y``, and ``z``. These examples
show accepted permutations and rejected name selections:

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import torch
         from leapp import Axis, AxisKind, frame_position

         # Accepted: labels match the position components in the tensor.
         xyz = torch.tensor([1.0, 2.0, 3.0])
         position = frame_position(
             "position", xyz,
             axes=[Axis(AxisKind.COMPONENT, names=["x", "y", "z"])],
         )

         # Accepted: a different order, with values and labels both ordered z, x, y.
         zxy = torch.tensor([3.0, 1.0, 2.0])
         reordered_position = frame_position(
             "position", zxy,
             axes=[Axis(AxisKind.COMPONENT, names=["z", "x", "y"])],
         )

         # Rejected (ValueError): velocity labels cannot describe a position.
         frame_position(
             "position", xyz,
             axes=[Axis(AxisKind.COMPONENT, names=["vx", "vy", "vz"])],
         )

         # Rejected (ValueError): z is missing and y appears twice.
         frame_position(
             "position", xyz,
             axes=[Axis(AxisKind.COMPONENT, names=["x", "y", "y"])],
         )

   .. tab:: YAML and errors

      The accepted calls produce these semantic fields when exported separately.
      Both rejected calls raise ``ValueError`` during construction, so neither
      produces YAML. Run rejected calls individually to observe each error.

      .. code-block:: yaml

         # Accepted: values ordered x, y, z
         name: position
         kind: frame/position
         axes:
         - {kind: component, names: [x, y, z]}
         spatial: {reference: null}

         ---
         # Accepted: values ordered z, x, y
         name: position
         kind: frame/position
         axes:
         - {kind: component, names: [z, x, y]}
         spatial: {reference: null}

These checks apply when names are supplied on a ``COMPONENT`` axis. Helpers do
not infer missing labels or change the underlying representation.

Use ``TensorSemantics`` directly for custom conventions or additional ``extra``
metadata. For joint positions, the helper is equivalent to:

.. code-block:: python

   from leapp import Kind, TensorSemantics

   measured = TensorSemantics("q", q, kind=Kind.JOINT_POSITION, axes=joint_axes)

The generic constructor validates metadata structure without the helper-specific
component-name checks. 
