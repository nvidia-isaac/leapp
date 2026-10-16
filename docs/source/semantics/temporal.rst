==================
Temporal Semantics
==================

Use ``Axis(AxisKind.TIME, period_ms=...)`` to mark one tensor dimension as
temporal and record the period between samples. For example, an action tensor
of shape ``[4, 3]`` has four steps for three joints:

.. code-block:: python

   from leapp import Axis, AxisKind, joint_effort

   joint_effort("actions", actions, axes=[
       Axis(AxisKind.TIME, period_ms=100),
       Axis(AxisKind.ELEMENT, names=["hip", "knee", "ankle"]),
   ])

.. code-block:: yaml

   name: actions
   dtype: float32
   shape: [4, 3]
   type: tensor
   kind: joint/effort
   axes:
   - {kind: time, period_ms: 100}
   - {kind: element, names: [hip, knee, ankle]}

A tensor may contain at most one time axis. Its period must be finite and
positive. The period is not validated against ``GraphConfigs.frequency``;
graph execution frequency and temporal sample spacing are separate concepts.
