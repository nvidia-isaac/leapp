===========
Limitations
===========

Torch tracing records the operations that dispatch through ``TracedTensor``.
The cases below are where the recorded graph cannot follow what the Python code
does.

View and mutation support
=========================

View updates rely on PyTorch's functionalization support for the recorded
operations. Unsupported mutation or view combinations fail during graph
construction. Aliases must be represented in the captured graph: a slice
created before tracing does not automatically acquire a relationship to a
separately declared base tensor.

Exported artifacts expose value updates through declared outputs or feedback
state; callers must consume those values rather than expect their input
storage to change. Backend restrictions still apply, including restrictions on
tensor-derived slice bounds.

Data-dependent control flow
===========================

An exported graph is a static structure and cannot hold a branch that depends
on tensor values. Using a traced tensor in a boolean context records only the
branch taken during tracing and logs an error:

.. code-block:: text

   Attempted to use TracedTensor 'x' from node 'node' in a boolean context
   (if/while/and/or/not) during tracing.
   ...
   The traced graph is a static DAG and cannot represent dynamic branches.

Use ``torch.where()`` for element-wise choices, fixed iteration counts instead
of value-driven loops, or split the node around the decision so the branch
becomes the caller's problem rather than the graph's.

Extracting Python values
========================

``.item()``, ``.tolist()``, ``float()``, and ``int()`` return ordinary Python
values captured at trace time. They leave the graph, and anything computed
from them afterwards is frozen at its traced value.

Torch gives no warning when this happens, unlike the NumPy path. A scalar
pulled out mid-node and multiplied back in looks completely normal and exports
a wrong model, so treat scalar extraction as a deliberate trace break.
