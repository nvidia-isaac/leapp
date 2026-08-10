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

Preserving traced state across nonstandard copies
=================================================

Between nodes, LEAPP wires ``pipeline.data_flow`` from the producing node and
output port that a finished output value carries. If you copy such a value into
another tensor, the data moves but that state does not, so the next node looks
disconnected.

:func:`~leapp.annotate.mirror_leapp_tags` copies the traced state from the
source output to the destination value after verifying that their values match.

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

   leapp.start(name="state_copy_example")
   out = upstream(torch.tensor([1.0, 2.0, 3.0]))

   buffer = torch.zeros_like(out)
   buffer[:] = out
   annotate.mirror_leapp_tags(out, buffer)

   result = downstream(buffer)
   leapp.stop()
   leapp.compile_graph()

Torch and Warp destinations are upgraded in place, so the return value can be
ignored. A raw ``np.ndarray`` cannot be upgraded in place, so NumPy callers must
assign the return value instead:

.. code-block:: python

   buffer = annotate.mirror_leapp_tags(out, buffer)

.. warning::

   ``mirror_leapp_tags`` requires source and target values to match exactly. It
   is for preserving graph wiring after equivalent copies, not for marking a
   newly computed tensor as if it came directly from another tensor.

Copies made with ``clone()``, ``detach()``, ``contiguous()``, ``cpu()``, or
``cuda()`` currently need ``mirror_leapp_tags`` as well. If a node looks
disconnected after such a copy, either pass the original output to the next node
or mirror the state onto the copy.
