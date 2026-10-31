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
                 Axis(AxisKind.COMPONENT, names=["vx", "vy", "vz", "wx", "wy", "wz"]),
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
         - {kind: component, names: [vx, vy, vz, wx, wy, wz]}

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
   * - Pose, position, orientation
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
* A wrench contains force and torque. Without a separate application position,
  the force acts at the element origin and moments are taken about that origin.
* ``force_application_position`` describes an optional XYZ position. It is
  paired with a wrench by element name and input/output purpose.
* The wrench and application position use the same ``expressed_in`` value.
* With an application position, the wrench's moment is the supplied torque;
  applying the offset force adds its lever-arm moment.

For the application position, ``expressed_in`` determines how to interpret XYZ:

* ``ExpressionFrame.SELF`` means an offset from the element origin.
* ``None`` means an absolute global position.
* A named frame supplies the position's origin and coordinate axes.


For example, apply a force along a tool's local y-axis at a point 0.1 meters
along its local x-axis:

* Both annotations identify the same frame through
  ``Axis(AxisKind.ELEMENT, names=["tool"])``. This shared element name connects
  the wrench to its application position.
* Register both in the same input/output direction. Both use
  ``ExpressionFrame.SELF``, so the force and offset use the tool's own axes.

.. tabs::

   .. tab:: Python

      .. code-block:: python

         import torch
         from leapp import Axis, AxisKind, ExpressionFrame
         from leapp import frame_wrench, force_application_position

         # The shared element name pairs these two annotations.
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
             axes=[
                 tool_axis,  # Same frame name as the wrench: "tool".
                 Axis(AxisKind.COMPONENT, names=["x", "y", "z"]),
             ],
         )
         # Pass [wrench, application] together to annotate.output_tensors.

   .. tab:: YAML

      The two entries after registration as outputs and export. Their matching
      ``element`` names connect them. Numeric force and offset values travel
      in the tensors; the YAML describes how to interpret them.

      .. code-block:: yaml

         outputs:
         - name: tool_wrench
           dtype: float32
           shape: [1, 6]
           type: tensor
           kind: frame/wrench
           axes:
           - {kind: element, names: [tool]}
           - {kind: component, names: [fx, fy, fz, tx, ty, tz]}
           spatial:
             expressed_in: {selector: self}
         - name: force_position
           dtype: float32
           shape: [1, 3]
           type: tensor
           kind: frame/force_application_position
           axes:
           - {kind: element, names: [tool]}
           - {kind: component, names: [x, y, z]}
           spatial:
             expressed_in: {selector: self}
