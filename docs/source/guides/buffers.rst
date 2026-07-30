================
Constant tensors
================

Use this page when a node must emit a tensor that is not derived from any
traced input. For module buffer state versus constants, see :doc:`graph`.

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

Preserving LEAPP tags across nonstandard copies
===============================================

Between nodes, LEAPP uses internal tags on finished outputs to wire
``pipeline.data_flow``. If you copy those values into another tensor with a
nonstandard pattern such as ``buffer[:] = upstream_output``, the values move
but the tags often do not, so the next node may look disconnected.

:func:`~leapp.annotate.mirror_leapp_tags` copies tags from the source output to
the destination tensor after verifying that their values match.

Use it only for copies **between** finished nodes. Inside a traced node,
full-slice assignment and ``copy_()`` from a traced tensor are handled
automatically.

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate

   @annotate.method(export_with="jit")
   def upstream(x: torch.Tensor):
       return x + 1.0

   @annotate.method(export_with="jit")
   def downstream(x: torch.Tensor):
       return x * 2.0

   leapp.start(name="tag_copy_example")
   out = upstream(torch.tensor([1.0, 2.0, 3.0]))

   buffer = torch.zeros_like(out)
   buffer[:] = out
   annotate.mirror_leapp_tags(out, buffer)

   result = downstream(buffer)
   leapp.stop()
   leapp.compile_graph()

.. warning::

   ``mirror_leapp_tags`` requires source and target values to match exactly. It
   is for preserving graph wiring after equivalent copies, not for marking a
   newly computed tensor as if it came directly from another tensor.

You do not need ``mirror_leapp_tags`` for standard PyTorch operations such as
``clone()``, ``detach()``, ``contiguous()``, ``cpu()``, or ``cuda()``; LEAPP tags
are preserved through those operations automatically.
