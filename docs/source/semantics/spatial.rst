.. _semantics-spatial:

================
Spatial Metadata
================

.. warning::

   Spatial metadata and its quantity helpers are experimental and may evolve.

To interpret a spatial quantity, a consumer needs to know both what it is
relative to and which coordinate axes were used to write its values:

``reference``: relative to which frame?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Suppose we describe the twist (linear and angular velocity) of Frame A.
* With ``reference="world"``, we describe A's motion relative to the world.
* With ``reference="frame_b"``, we describe A's motion relative to Frame B.
* If B is moving, these are different relative motions and generally have
  different values.
* For example, if A is fixed to a moving B, A's twist relative to B is zero,
  even though A can have a nonzero twist relative to the world.

``expressed_in``: which frame's axes do we use to write the numbers?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Keep ``reference="frame_b"``: we are still describing A's motion relative
  to B.
* Use ``expressed_in="frame_a"`` to write that twist using A's axes, or
  ``expressed_in="frame_b"`` to use B's axes.
* When those axes have different orientations, the numerical values generally
  differ, even though the relative motion is the same.
* For example, a linear velocity along A's x-axis may be along B's y-axis,
  so the value appears in a different velocity component.

Both fields accept non-empty frame names. For ``expressed_in``, you can also
select axes without repeating a frame name:

* ``expressed_in=ExpressionFrame.SELF`` uses the element's own coordinate axes.
* ``expressed_in=ExpressionFrame.REFERENCE`` uses the coordinate axes of the
  frame specified by the ``reference`` argument. For example, with
  ``reference="base"``, this selector uses the base frame's axes. 

``None`` means global for either field. A literal string ``"self"`` names a
frame and differs from ``ExpressionFrame.SELF``. These annotations describe
the values you already have; they do not rotate or transform them.

For example, the following snippet describes a tool's velocity relative to its base, with components
expressed along the tool's own axes:

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import torch
         from leapp import Axis, AxisKind, ExpressionFrame, frame_twist

         tool_velocity = torch.zeros(1, 6)
         motion = frame_twist(
             "tool_velocity", tool_velocity,
             reference="base",
             expressed_in=ExpressionFrame.SELF,
             axes=[
                 Axis(AxisKind.ELEMENT, names=["tool"]),
                 Axis(AxisKind.COMPONENT, names=["x", "y", "z", "angular_x", "angular_y", "angular_z"]),
             ],
         )

   .. tab:: YAML

      The corresponding semantic fields in the exported YAML are:

      .. code-block:: yaml

         kind: frame/twist
         spatial:
           reference: base
           expressed_in: {selector: self}
         axes:
         - {kind: element, names: [tool]}
         - {kind: component, names: [x, y, z, angular_x, angular_y, angular_z]}

.. _spatial-yaml-fields:

Spatial fields in YAML
~~~~~~~~~~~~~~~~~~~~~~~~

Spatial fields are grouped under ``spatial``. Convenience helpers emit the
following fields even when their values use the global default:

.. list-table::
   :header-rows: 1
   :widths: 45 35 20

   * - Helper family
     - Fields emitted under ``spatial``
     - Default values
   * - Pose, position, orientation, point cloud
     - ``reference``
     - ``null``
   * - Twist, linear/angular velocity, acceleration
     - ``reference``, ``expressed_in``
     - Both ``null``
   * - Wrench, force-application position
     - ``expressed_in``
     - ``null``

Here, ``null`` means global. A non-null ``expressed_in`` uses either
``{frame: <name>}`` or ``{selector: self/reference}``, subject to the helper's
accepted selectors. The selector value is either ``self`` or ``reference``.
This table describes convenience-helper output; custom string kinds alone do
not activate those defaults.

Other annotations may omit ``spatial`` or its fields. Missing spatial fields
also mean global for applicable spatial quantities. See
:ref:`yaml-minimum-contract` for the full required/optional field contract.

Wrenches and the force application position
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without a separate application position, a wrench's force acts at the element
origin and its torque is measured about that origin. An optional
``force_application_position`` pairs with it through the same element names,
input/output direction, and ``expressed_in`` value. Its offset force adds a
lever-arm moment to the supplied torque.

The application position is an offset from the element origin with
``ExpressionFrame.SELF``, an absolute global position with ``None``, or a
position in the named frame's origin and axes.

.. dropdown:: Example: force applied at a local offset

   Both annotations describe the tool in its own axes. Register them together
   as inputs or together as outputs.

   .. code-block:: python

      import torch
      from leapp import Axis, AxisKind, ExpressionFrame
      from leapp import frame_wrench, force_application_position

      tool_axis = Axis(AxisKind.ELEMENT, names=["tool"])
      wrench = frame_wrench(
          "tool_wrench", torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0, 0.0]]),
          expressed_in=ExpressionFrame.SELF,
          axes=[
              tool_axis,
              Axis(AxisKind.COMPONENT, names=["fx", "fy", "fz", "tx", "ty", "tz"]),
          ],
      )
      application = force_application_position(
          "force_position", torch.tensor([[0.1, 0.0, 0.0]]),
          expressed_in=ExpressionFrame.SELF,
          axes=[tool_axis, Axis(AxisKind.COMPONENT, names=["x", "y", "z"])],
      )
