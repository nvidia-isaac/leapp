==========
Validation
==========

This guide focuses on how to validate that a LEAPP export is correct.
There are two layers of validation during export:

#. Automatic per-node validation during
   ``leapp.compile_graph(validate=True)``.
#. Replay validation across cached re-entry examples using
   ``max_cached_io``.

Automatic validation during ``compile_graph()``
===============================================

The main validation entry point is:

.. code-block:: python

   leapp.compile_graph(
       validate=True,
       rtol=1e-3,
       atol=1e-5,
       strict=True,
   )

When ``validate=True``, LEAPP:

* runs each exported node model
* feeds it the captured traced inputs
* compares exported outputs against the outputs seen during tracing
* logs any deviation it detects
* returns a dict mapping node names to validation results

``strict=True`` raises if any node fails validation. ``strict=False``
still runs validation, but lets you inspect the result dict even when
some nodes fail.

What is compared
----------------

Validation happens per node, not just at the whole-graph level. For each
node, LEAPP compares:

* the number of outputs
* output tensor values via ``torch.allclose(..., rtol=..., atol=...)``
* NaN / Inf presence in exported and traced outputs

If a node has no compiled model, validation is skipped and treated as
successful. This is expected for metadata-only nodes such as ``non_traced``
or dry-run cases.

Multi-example validation with cached inputs
===========================================

If a node is re-entered multiple times within the same trace session,
LEAPP can cache multiple I/O examples and validate the exported model
against all of them.

.. code-block:: python

   leapp.start(name="my_graph", max_cached_io=5)

How cached validation works
---------------------------

* The first execution becomes the main traced example.
* Later re-entries are validated for structural consistency and stored as
  cached examples.
* During ``compile_graph(validate=True)``, LEAPP validates the exported
  model against the original traced example **and** every cached example.

This is especially useful for:

* looped pipelines
* stateful nodes
* cases where one example is not enough to trust the export

What gets cached
----------------

For each node, LEAPP caches:

* input values
* output values
* updated tags needed for feedback detection across re-entry

The validation log labels examples using numeric sample indices:

* ``sample 0`` for the original traced example
* ``sample 1``, ``sample 2``, ... for cached re-entry examples

If a later example fails while the first one passes, you can see exactly
which replayed example exposed the mismatch.

What LEAPP reports on validation failure
========================================

When validation fails, LEAPP prints more than just "allclose failed".
Depending on the failure mode, it reports different diagnostics.

Output count mismatch
---------------------

If the exported model returns a different number of outputs than the
traced node, LEAPP logs:

* node name
* example label (``sample 0``, ``sample 1``, ...)
* actual output count
* expected output count

NaN / Inf analytics
-------------------

If either the exported output or the traced output contains NaN or Inf
values, LEAPP logs:

* which node/output failed
* whether NaNs/Infs came from the exported output or the traced source
  output
* counts and percentages of NaN / Inf values

This is useful for quickly distinguishing export corruption from already
unstable traced outputs.

Numeric mismatch analytics
--------------------------

When values differ outside ``rtol`` / ``atol``, LEAPP logs:

* node name and output name
* example label (``sample N``)
* active ``rtol`` and ``atol``
* shape and dtype of source and exported outputs
* source and exported value ranges
* absolute-difference statistics: max, mean, and the ``p50``, ``p75``,
  ``p90``, ``p99``, ``p995`` percentiles
* path to the LEAPP log file for deeper inspection

This is the main numeric-debugging signal when an export is "close but
wrong".

Validation summary
------------------

At the end, LEAPP prints a summary:

* passed node count
* failed node count
* node count that errored during execution

If ``strict=True``, LEAPP then raises with the list of failed node names.

Re-entry validation before export
=================================

Before final model validation runs, LEAPP also checks that repeated
executions of a node are structurally consistent. On re-entry LEAPP
validates:

* input/output names
* shape and dtype descriptions
* tags used for graph connectivity

These checks catch cases where later executions no longer match the
original trace shape, dtype, or connection structure.

Recommended validation workflow
===============================

For high-confidence exports:

#. Trace representative executions, not just one trivial example.
#. Increase ``max_cached_io`` when nodes are re-entered or stateful.
#. Run ``leapp.compile_graph(validate=True, strict=True)``.
#. Relax ``rtol`` / ``atol`` only when the mismatch is expected numerical
   drift.

When validation fails
=====================

Use the failure signal to decide what to inspect next:

* If only a ``cached[i]`` sample fails, your export may not generalize
  across re-entry. If you missed some variable inputs, LEAPP treats those
  as constants.
* If NaN / Inf appears only in exported outputs, the export backend
  likely introduced instability.
* If NaN / Inf already appears in traced source outputs, the original
  computation is unstable too.
* If output counts differ, inspect the export backend and output
  declarations first.
* If ranges and diff percentiles look systematically shifted, suspect
  backend conversion semantics or tolerance settings.
