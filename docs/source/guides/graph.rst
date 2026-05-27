=================
Graph operations
=================

This guide explores graph-level operations: cycles, feedback connections,
and tracing continuity across graph edges.

Cycle detection and feedback connections
========================================

LEAPP automatically detects cycles through **feedback connections**. A
feedback connection occurs when data flows from a later node back to an
earlier node, creating a loop in the graph.

How LEAPP detects cycles
------------------------

LEAPP assigns each node an index when it completes its initial trace.
In practice, this is the node completion order, not the order a node
name first appeared. When analyzing connections:

* **Normal connections** flow from a lower-indexed node to a higher-indexed
  node (forward flow).
* **Feedback connections** flow from a higher-indexed node back to a
  lower-indexed node (backward flow).

Feedback connections are visualized in red, while normal connections
appear in black.

Capturing feedback behavior
---------------------------

For graph-level feedback that is inferred from re-entry across nodes, you
need to **run your graph multiple times** within the same tracing session.
This lets LEAPP observe data flowing from a later node back into an
earlier node on a later iteration.

This is different from explicit state APIs such as
:func:`~leapp.annotate.state_tensors` / :func:`~leapp.annotate.update_state`
and :func:`~leapp.annotate.module`, which can produce feedback metadata in
a single trace.

A complete runnable version of this example lives in
``examples/feedback_example.py``:

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate

   def mix_with_feedback(data, feedback):
       centered = data - 0.5
       return torch.tanh(centered + 0.25 * feedback)

   def blend_feedback(hidden, previous_feedback):
       return 0.8 * previous_feedback + 0.2 * hidden

   def main():
       leapp.start(name="sample_feedback_graph")

       policy_memory = torch.tensor([0.0])

       for _ in range(2):  # needed for inferred cross-node feedback
           policy_inputs = annotate.input_tensors("policy_step", {
               "observation_scalar": torch.tensor([1.0]),
               "policy_memory_in": policy_memory,
           })
           policy_context = mix_with_feedback(policy_inputs[0],
                                              policy_inputs[1])
           control_action = torch.clamp(policy_context * 2.0,
                                        min=-1.0, max=1.0)
           annotate.output_tensors(
               "policy_step",
               {"policy_context": policy_context,
                "control_action": control_action},
               export_with="jit",
           )

           feedback_inputs = annotate.input_tensors("feedback_update", {
               "policy_context": policy_context,
               "policy_memory_prev": policy_memory,
           })
           policy_memory = blend_feedback(feedback_inputs[0],
                                          feedback_inputs[1])
           annotate.output_tensors(
               "feedback_update",
               {"policy_memory_out": policy_memory},
               export_with="jit",
           )

       leapp.stop()
       leapp.compile_graph()

In this example, ``policy_memory_out`` flows from ``feedback_update``
(completed later) back into the ``policy_memory_in`` input of
``policy_step`` on the next iteration, creating a feedback connection.

.. image:: /_static/images/feedback_example_graph.png
   :alt: Feedback example graph
   :align: center

Inspect detected feedback details under the ``feedback_flow`` field:

.. code-block:: yaml

   feedback_flow:
       feedback_update/policy_memory_out:
         - policy_step/policy_memory_in
         - feedback_update/policy_memory_prev

Important considerations
------------------------

.. warning::

   **Minimum two iterations required.** You must run the loop at least
   twice for LEAPP to detect inferred cross-node feedback:

   * **First iteration** --- LEAPP traces all nodes and establishes direct
     connections.
   * **Second iteration** --- LEAPP observes data flowing back to earlier
     nodes, confirming the feedback connection.

   Explicit feedback declared with ``state_tensors()`` / ``update_state()``
   or detected via ``annotate.module()`` does not require a second
   iteration.

.. note::

   **Port names are preserved.** LEAPP emits source and target port names
   as annotated. Downstream frameworks should read the explicit
   ``data_flow`` / ``feedback_flow`` mappings rather than assuming
   connected port names match.

Maintaining tracing with ``mirror_leapp_tags``
==============================================

When LEAPP traces tensor data, it relies on internal tags to track data
provenance. Some patterns duplicate data without using standard PyTorch
operations like ``clone()`` or ``detach()`` --- for example, an in-place
assignment like ``tensor[:] = other_tensor``, or a round-trip through
``numpy.array(tagged_tensor)``. In these cases the tags do not transfer
automatically.

:func:`~leapp.annotate.mirror_leapp_tags` solves this by explicitly
transferring tracing tags from a source tensor to a target tensor.

When to use it
--------------

Use ``mirror_leapp_tags`` when you:

* Copy tensor data using in-place operations
  (``self._prev_action[:] = self._action``)
* Need to maintain tracing continuity across manual data duplication
* Want LEAPP to recognize that two tensors contain the same logical data

How it works
------------

The function performs two operations:

#. **Verifies data equivalence** --- first checks that source and target
   contain exactly the same values.
#. **Transfers tags** --- if verification passes, copies all LEAPP
   internal tracking tags from source to target.

If the data does not match, LEAPP logs an error and raises instead of
copying incorrect tracing metadata.

Example: pre-allocated buffer
-----------------------------

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate

   class DataProcessor:
       def __init__(self):
           self._buffer = torch.zeros(10)

       @annotate.method(export_with="jit")
       def process(self, input_data: torch.Tensor):
           # input_data is tagged so graph connections can be tracked.
           self._buffer[:] = input_data
           annotate.mirror_leapp_tags(input_data, self._buffer)
           return self._buffer * 2.0

API signature
-------------

.. code-block:: python

   annotate.mirror_leapp_tags(source, target)

Parameters:

* ``source`` --- tensor containing the original data and LEAPP tags
* ``target`` --- tensor that should receive the tags (must have identical
  values to ``source``)

Important considerations
------------------------

.. warning::

   **Data must match exactly.** The function raises if the values in
   source and target differ:

   .. code-block:: python

      source = torch.tensor([1.0, 2.0, 3.0])
      target = torch.tensor([1.0, 2.0, 4.0])  # Different value
      annotate.mirror_leapp_tags(source, target)  # Raises

.. note::

   **Only works during tracing.** Outside ``leapp.start()`` /
   ``leapp.stop()`` blocks, ``mirror_leapp_tags`` safely no-ops.

When *not* to use it
--------------------

You don't need this function for standard PyTorch operations:

.. code-block:: python

   new_tensor = old_tensor.clone()    # tracked automatically
   detached = old_tensor.detach()     # tracked automatically
   copied = old_tensor                # reference, no duplication
   new_tensor = [old_tensor]          # container change, same tensor

LEAPP tags are automatically preserved through:

* ``.clone()`` --- creates a copy with the same tags
* ``.detach()`` --- preserves tags
* ``.contiguous()`` --- preserves tags
* ``.cpu()`` / ``.cuda()`` --- device transfers preserve tags

Also avoid it when the source and target values are intentionally
**different**:

.. code-block:: python

   new_tensor = old_tensor + 10        # computation -- track as a node
   new_tensor[:5] = old_tensor         # partial overlap -- not equivalent

Summary
=======

* **Feedback connections** capture cyclic behavior by running your graph
  multiple times during tracing.
* :func:`~leapp.annotate.mirror_leapp_tags` maintains proper tracing when
  duplicating tensor data with in-place operations.

These features give you fine-grained control over how LEAPP interprets
and optimizes your computational graphs for deployment.
