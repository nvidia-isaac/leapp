.. _semantics-spatial:

================
Spatial Metadata
================

.. warning::

   Spatial metadata and its quantity helpers are experimental and may evolve.

Spatial metadata identifies a quantity's reference frame and coordinate axes.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Field
     - Meaning
     - Effect of changing it
   * - ``reference``
     - The frame relative to which the quantity is measured.
     - Changes the relative quantity. A tool fixed to a moving base has zero
       velocity relative to the base, but may move relative to the world.
   * - ``expressed_in``
     - The frame whose coordinate axes describe the values.
     - Changes the components used to describe the same relative quantity.

Both fields accept non-empty frame names; ``None`` means global.
``expressed_in`` also accepts ``ExpressionFrame.SELF`` for the element's axes
or ``ExpressionFrame.REFERENCE`` for the reference frame's axes, where supported.
See :ref:`semantics-helper-spatial-arguments` for each helper's accepted arguments.
A literal string ``"self"`` names a frame and differs from the ``SELF`` selector.
Annotations describe existing values; they do not rotate or transform them.

For example, this describes a tool's velocity relative to its base, with
components expressed along the tool's own axes:

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

.. _spatial-yaml-fields:

Spatial fields in YAML
~~~~~~~~~~~~~~~~~~~~~~~~

Helpers export their accepted spatial arguments under ``spatial``, including
global defaults as ``null``. A non-null ``expressed_in`` uses ``{frame: <name>}``
or ``{selector: self/reference}``. Custom string kinds alone do not activate
these defaults. Missing spatial fields also mean global for applicable
quantities; see :ref:`yaml-minimum-contract` for the full field contract.

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
