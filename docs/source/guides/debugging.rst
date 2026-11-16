=========
Debugging
=========

Validate exports with representative inputs, then use the logs to locate any
differences between the captured pipeline and the exported model.

Confirm the export
==================

``compile_graph(validate=True)`` compares each exported node with its cached
Python input/output samples. Shapes must match exactly; values are compared
with ``rtol`` and ``atol``.

.. code-block:: python

   results = leapp.compile_graph(
       validate=True, strict=True, rtol=1e-3, atol=1e-5,
   )

``strict=True`` raises on failure. Use ``strict=False`` to inspect the result
dict instead. Nodes without compiled models are skipped.

Run more than one step
----------------------

Use different observations and commands so validation can catch live values
that were captured as constants. ``max_cached_io`` limits the samples per node.

.. code-block:: python

   leapp.start("policy_graph", max_cached_io=5)
   for _ in range(5):
       obs = annotate.input_tensors("policy", {"obs": next_obs()})
       annotate.output_tensors(
           "policy", {"action": policy(obs)}, export_with="jit",
       )
   leapp.stop()
   leapp.compile_graph(validate=True, strict=True)

Re-entry checks names, shapes, dtypes, and connectivity during tracing.
Export validation checks the recorded values after compilation.

When validation fails
---------------------

The mismatch log identifies the node, output, and sample, with shapes,
tolerances, and error statistics.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Symptom
     - Check
   * - Sample 0 passes, later samples fail
     - Declare changing values with ``annotate.input_tensors()`` rather than
       leaving them as constants.
   * - Replaying sample 0 gives a different result
     - Check undeclared persistent buffers and nondeterministic execution.
       Use ``annotate.register_buffer()`` for traced buffers, or
       ``state_tensors()`` / ``update_state()`` for state carried between calls.
   * - Small numerical differences
     - Compare with ``float32`` and without ``torch.autocast``. On NVIDIA GPUs,
       try ``NVIDIA_TF32_OVERRIDE=0`` before starting Python, then recapture.

.. dropdown:: Numerical differences and replay

   Validation repeats sample 0 after the other samples, restoring declared
   buffers first. A missed reset of an undeclared buffer can make the export
   accumulate values across calls.

   ONNX Runtime and PyTorch can round differently. TorchScript scripting can
   also change rounding after its first call when it fuses operations. These
   effects are more visible with ``float16`` or ``bfloat16``.

   Diagnostic hints suggest possible causes; they do not establish that a
   difference is harmless.

Data-dependent shapes
---------------------

For results of boolean masks, ``nonzero``, or ``unique``, use ``.shape``,
``.size()``, or ``.numel()`` to keep size-dependent computation in the graph.
``len(x)`` and Python ``float`` arithmetic on sizes produce constants. Include
samples with different result sizes to catch a frozen count.

.. _stateful-validation:

Stateful transitions and sequential replay
------------------------------------------

Cached validation checks each recorded ``(observation, state) -> (action,
next_state)`` transition independently, using its recorded input state.
Declare state in the exported interface as described in :doc:`graph`.

Also compare a sequence through :doc:`../leapp_runtime` with the original
pipeline, starting from the same state. This feeds exported state into later
calls and checks whether small accepted errors accumulate. Include episode
resets and observations beyond the capture set.

Inspect what was captured
=========================

``leapp.start("policy", save_path="exports")`` writes detailed diagnostics to
``exports/policy/log.txt``. Use ``verbose=True`` on ``start()`` for trace-time
console output, and on ``compile_graph()`` for export and validation output.

FX graph
--------

Search the log for ``Compiled graph module for <node name>``. The FX dump shows
the graph passed to the backend: input placeholders, recorded operations, and
outputs. Check it for missing inputs or unexpected constants.

Graph image
-----------

On Python 3.11+, ``compile_graph(visualize=True)`` also writes a PNG showing
node connections, boundary tensors, and dtypes.

Isolate the failure
===================

Use these options to narrow down which stage causes the problem:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Option
     - Effect
   * - ``start(..., dry_run=True)``
     - Trace and write graph YAML without compiling models.
   * - ``start(..., non_traced=["node"])``
     - Skip export for selected nodes while preserving their connections.
   * - ``export_with=None`` on ``output_tensors()``
     - Skip export for that node.
   * - ``start(..., global_patching=False)``
     - Disable session-wide conversion and function patches.

These options still execute your Python pipeline. Warp dry runs also retain
the second pass that checks whether a segment's execution diverges.
