=============
How it works
=============

``TracedTensor`` subclasses ``torch.Tensor`` and shares its storage, so it
holds real data and behaves like a tensor everywhere LEAPP is not involved. It
implements ``__torch_function__``, torch's dispatch hook, which lets it see
every torch call made on it before torch executes it.

Each intercepted call does two things:

#. Runs the original torch operation eagerly on the real tensors, so the value
   you get back is exactly what torch would have produced.
#. Records that same operation into the FX graph and wraps the result in a new
   ``TracedTensor`` pointing at the new graph node.

Distilled into an FX graph
==========================

Every recorded operation lands in one place: a ``torch.fx.Graph`` owned by the
node being traced. That graph is the only thing an export backend ever sees.

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: leapp-fit-grid

   .. grid-item-card:: What it is
      :class-card: leapp-fit-card sd-rounded-3

      Torch's own way of describing a computation as data instead of code. An
      FX graph is an ordered list of nodes, and each node records one step: its
      kind, what it targets, and which earlier nodes its arguments came from.

   .. grid-item-card:: Why LEAPP records into it
      :class-card: leapp-fit-card sd-rounded-3

      It is the one form every torch export path already accepts, so a single
      recording can leave as TorchScript, ``.pt2``, or ONNX with no second
      trace. It is also plain Python, so LEAPP can edit it after the fact.

   .. grid-item-card:: How LEAPP fills it
      :class-card: leapp-fit-card sd-rounded-3

      :func:`~leapp.annotate.input_tensors` opens an empty graph and adds one
      node per declared input. Each intercepted call appends one more.
      :func:`~leapp.annotate.output_tensors` closes the graph and hands it to
      that node's backend.

This is also where the other tensor libraries meet. :doc:`NumPy <numpy>` calls
and :doc:`Warp <warp>` segments become nodes in this same graph, so a node that
mixes libraries still exports as one model.

Nothing is renamed on the way in: a node's target is the torch function you
called. Translation into a backend's own vocabulary happens later, at export,
which is why one recording can go two ways at once:

.. raw:: html

   <div class="leapp-diagram" aria-label="One torch call recorded once and exported to two backends">
     <div class="leapp-diagram-stack">
       <div class="leapp-diagram-box leapp-box-code">torch.clamp(x, -5.0, 5.0)</div>
       <div class="leapp-diagram-chevron">&darr; recorded as</div>
       <div class="leapp-diagram-box">%clamp = call_function[target=torch.clamp]</div>
       <div class="leapp-artifact-grid">
         <div class="leapp-diagram-chevron">&darr; onnx</div>
         <div class="leapp-diagram-chevron">&darr; pt2</div>
       </div>
       <div class="leapp-artifact-grid">
         <span>Clip</span>
         <span>torch.ops.aten.clamp.default</span>
       </div>
     </div>
   </div>

Only the middle row belongs to LEAPP. The two below it are produced by torch's
own exporters from that single node.

LEAPP also edits the finished graph before handing it over. A faithful
recording is not always a portable one, so a few passes normalize the forms
that no IR would accept, such as method calls recorded as functions,
decomposed ``aten`` operations, and constants left as transposed views. They
change how the computation is written down, never what it computes, and they
are what keeps one trace usable across TorchScript, PT2, and ONNX.

Tracing spreads by data, not by scope
=====================================

:func:`~leapp.annotate.input_tensors` does not copy your tensor. It changes the
class of the object you handed in, so that object *is* the traced one, backed
by the same storage as before. Every operation on it then returns a new
``TracedTensor`` carrying the graph node that produced it.

Tracing therefore follows values rather than code. LEAPP has no list of
functions to watch and needs no annotation on the code doing the work: a
helper in your project, a method on your model, or a routine from another
package all get traced as long as a traced tensor flows through them and they
do their work in torch.

The same rule sets the boundary. Tracing continues exactly as far as the
traced type does, and stops wherever a value stops being a ``TracedTensor``.
:doc:`torch_limitations` covers the ways that happens.

Dead code never reaches the export
==================================

Recording happens as your code runs, so the graph initially holds everything
the tracer saw. Declaring the outputs is what decides which of it matters.

At :func:`~leapp.annotate.output_tensors`, LEAPP walks backwards from the
declared outputs, marks every node reachable from them, and erases the rest.
What survives is the smallest computation connecting the declared inputs to
the declared outputs, so intermediate values that no output depends on cost
nothing in the exported model.

Declared inputs are pruned the same way, which is usually worth knowing about,
so LEAPP names them:

.. code-block:: text

   detected the following inputs are not used in the computation or directly
   returned as output

An input reported there either feeds nothing, or is returned unchanged. Both
mean the node's declared interface is wider than the work it does.
