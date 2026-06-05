============================
Buffers and constant tensors
============================

Buffers are useful for two different jobs, and LEAPP treats them differently
depending on whether they change during tracing:

* **State buffers** are reassigned during execution and become feedback state.
* **Constant buffers** are read but not reassigned and are baked into the
  exported model.

For modules, :func:`~leapp.annotate.module` handles both cases automatically.

Static outputs
==============

Sometimes a node needs to output a **constant tensor** that is not derived
from any input. The ``static_outputs`` parameter on ``output_tensors()``
handles this case:

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate

   leapp.start(name="static_example")

   input_tensor = torch.tensor([1.0, 2.0, 3.0])
   traced_input = annotate.input_tensors('my_node',
                                         {'input': input_tensor})

   # Computed output -- derived from the traced input.
   computed_output = traced_input + 1.0

   # Static output -- a constant, NOT derived from any input.
   static_tensor = torch.tensor([4.0, 5.0, 6.0])

   annotate.output_tensors(
       'my_node',
       {'computed': computed_output},
       static_outputs={'static': static_tensor},
       export_with="jit",
   )

   leapp.stop()
   leapp.compile_graph()

The exported model returns both outputs: ``computed`` (input-dependent) and
``static`` (always ``[4, 5, 6]``).

``static_outputs`` follows the same top-level naming contract as
``output_tensors()``:

* pass a dict of named raw tensors for plain static outputs, or
* pass ``TensorSemantics(...)`` / a list of ``TensorSemantics(...)`` if the
  static outputs should carry semantic metadata in the exported YAML.

.. warning::

   * Static outputs must be **raw** ``torch.Tensor`` values. Using a
     ``TracedTensor`` will raise an error.
   * Bare top-level tensors are not accepted. Pass a dict of named raw
     tensors or ``TensorSemantics(...)`` / a list of ``TensorSemantics(...)``.
   * Static outputs are merged with the regular outputs in the compiled
     model --- downstream nodes can consume them like any other output.

Preserving tags after manual copies
===================================

When LEAPP traces tensor data, it relies on internal tags to track data
provenance. Some patterns duplicate data without standard PyTorch operations
like ``clone()`` or ``detach()``. For example, an in-place assignment like
``tensor[:] = other_tensor`` may copy values without carrying LEAPP tags to the
new tensor.

:func:`~leapp.annotate.mirror_leapp_tags` explicitly transfers tracing tags
from a source tensor to a target tensor after verifying that their values match.

Use ``mirror_leapp_tags`` when you:

* Copy tensor data using in-place operations.
* Need to maintain tracing continuity across manual data duplication.
* Want LEAPP to recognize that two tensors contain the same logical data.

.. code-block:: python

   import torch
   from leapp import annotate

   class DataProcessor:
       def __init__(self):
           self._buffer = torch.zeros(10)

       @annotate.method(export_with="jit")
       def process(self, input_data: torch.Tensor):
           self._buffer[:] = input_data
           annotate.mirror_leapp_tags(input_data, self._buffer)
           return self._buffer * 2.0

.. warning::

   ``mirror_leapp_tags`` requires source and target values to match exactly. It
   is for preserving trace continuity after equivalent copies, not for marking a
   newly computed tensor as if it came directly from another tensor.

You do not need ``mirror_leapp_tags`` for standard PyTorch operations such as
``clone()``, ``detach()``, ``contiguous()``, ``cpu()``, or ``cuda()``; LEAPP tags
are preserved through those operations automatically.
