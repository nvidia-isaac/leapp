==================
Temporal Semantics
==================

``TemporalAxis``
----------------

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

See :doc:`kind_element_names` for ``kind`` and ``element_names`` metadata.

