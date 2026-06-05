=============
Node patterns
=============

This guide covers advanced node patterns. The primary interface is
:func:`~leapp.annotate.input_tensors` and
:func:`~leapp.annotate.output_tensors`; use :func:`~leapp.annotate.method`
when a node fits cleanly inside one function and you want a shorthand.

Distributed inputs: one node, many call sites
=============================================

Each node should have one finalizing ``output_tensors()`` call, but it may
have multiple ``input_tensors()`` calls with the same ``node_name``. This is
useful when a node collects data from multiple helpers or files before
producing an output.

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate

   def get_lidar_data(env):
       lidar_data = env.get("lidar_data")
       return annotate.input_tensors("sensor_fusion",
                                     {"lidar_data": lidar_data})

   def get_camera_features(env):
       camera_features = env.get("camera_features")
       return annotate.input_tensors("sensor_fusion",
                                     {"camera_features": camera_features})

   def run_pipeline(env, model):
       leapp.start(name="distributed_inputs_example")

       lidar = get_lidar_data(env)
       camera = get_camera_features(env)

       fused = torch.cat([lidar, camera], dim=-1)
       annotate.output_tensors("sensor_fusion",
                               {"model_input": fused},
                               export_with="jit")

       leapp.stop()
       leapp.compile_graph()

Both ``input_tensors()`` calls reference the same node name, so LEAPP
treats them as one node with multiple inputs.

``annotate.method()`` as a shorthand
====================================

:func:`~leapp.annotate.method` is a convenience wrapper over the
traced-tensor path.

.. code-block:: python

   import torch
   from leapp import annotate

   @annotate.method(export_with="jit", node_name="preprocess")
   def preprocess(obs):
       return (obs - obs.mean()) / (obs.std() + 1e-6)

Use it when:

* one function naturally defines the node input and output boundary
* parameter names and return values already match the API you want to export

Prefer ``input_tensors()`` / ``output_tensors()`` when:

* node logic spans multiple helpers
* you want explicit tensor names
* you need semantic metadata on tensors

Nested data connections
=======================

LEAPP can track connections through complex nested structures. Each
individual tensor within nested dicts, lists, or custom objects is tracked
separately.

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate

   @annotate.method(export_with="jit", node_name="process_robot_state")
   def process_robot_state(state_dict):
       processed_state = {
           'position': state_dict['position'] * 2.0,
           'velocity': state_dict['velocity'] + 1.0,
           'sensors': {
               'lidar': state_dict['sensors']['lidar'].mean(dim=1),
               'camera': state_dict['sensors']['camera'].flatten(),
           },
       }
       return processed_state

   def main():
       leapp.start(name="nested_data_example")

       robot_state = {
           'position': torch.tensor([1.0, 2.0, 3.0]),
           'velocity': torch.tensor([0.5, 0.5, 0.0]),
           'sensors': {
               'lidar': torch.randn(360, 3),
               'camera': torch.randn(3, 224, 224),
           },
       }

       processed = process_robot_state(robot_state)

       # LEAPP creates connections per individual tensor path:
       #   robot_state['position']            -> processed['position']
       #   robot_state['velocity']            -> processed['velocity']
       #   robot_state['sensors']['lidar']    -> processed['sensors']['lidar']
       #   robot_state['sensors']['camera']   -> processed['sensors']['camera']

       p = annotate.input_tensors("decision_maker", {"processed": processed})

       position_factor = p['position'].norm()
       velocity_factor = p['velocity'].sum()
       sensor_confidence = p['sensors']['lidar'].std()

       action = torch.stack(
           [position_factor, velocity_factor, sensor_confidence])
       annotate.output_tensors("decision_maker",
                               {"action": action},
                               export_with="jit")

       leapp.stop()
       leapp.compile_graph()

How LEAPP handles nested structures
-----------------------------------

When LEAPP sees a complex nested data structure (dicts, lists, tuples) as
an input or output, it automatically:

#. **Flattens the structure.** Each individual tensor within the nested
   structure is extracted and tracked separately. For example,
   ``state_dict['sensors']['lidar']`` becomes a distinct input named
   ``state_dict_sensors_lidar``.
#. **Generates an auto-interface.** LEAPP generates wrapper code that
   accepts flat tensors and reconstructs them into the nested structure
   your original code expects on input, then unpacks the returned nested
   structure into flat tensors on output.
#. **Tracks connections at tensor level.** This flattening enables LEAPP
   to track data flow between nodes at the individual tensor level, not at
   the parameter level.

The result: all exported nodes have simple, flat tensor interfaces, while
your original code continues to use nested structures naturally. This
guarantees:

* consistent tensor-level connection tracking across all nodes
* compatibility with deployment frameworks that expect flat tensor I/O
* clear visibility into exactly which tensors flow between which nodes

In the code above, the ``process_robot_state`` node's exported model has
four separate tensor inputs (``state_dict_position``,
``state_dict_velocity``, ``state_dict_sensors_lidar``,
``state_dict_sensors_camera``) rather than a single complex dictionary
input.

Summary
=======

* Distributed traced inputs let one node collect data from multiple call
  sites.
* ``annotate.method()`` is a shorthand for self-contained functions.
* Nested data structures are automatically tracked.

LEAPP's goal is to capture your computational graph accurately. Being
explicit about data dependencies and naming will help avoid most issues.
