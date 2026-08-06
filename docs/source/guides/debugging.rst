=========
Debugging
=========

This guide covers the LEAPP features that are most useful while bringing up
a graph: logging, full FX graph inspection, dry-run capture, and selective
non-traced nodes.

Logging
=======

Every ``leapp.start()`` call configures a LEAPP log file in the graph output
directory. For example:

.. code-block:: python

   leapp.start("debug_policy", save_path="exports")

creates:

.. code-block:: text

   exports/debug_policy/log.txt

The log file captures all LEAPP log levels, including DEBUG output. The
console is quieter by default and shows warnings and errors. Use
``verbose=True`` when you want LEAPP to stream the detailed trace and compile
log to the console while still writing the same information to ``log.txt``:

.. code-block:: python

   leapp.start("debug_policy", save_path="exports", verbose=True)
   # ... run your traced code ...
   leapp.stop()
   leapp.compile_graph(validate=True, verbose=True)

Use ``leapp.start(..., verbose=True)`` to see trace-time diagnostics. Use
``leapp.compile_graph(..., verbose=True)`` to turn verbose console output on
for graph compilation, export, and validation.

Full FX graph inspection
------------------------

For each traced node, LEAPP writes the full ``torch.fx.Graph`` to the log
after building the node's ``fx.GraphModule``. Search for
``Compiled graph module for <node name>`` in ``log.txt``:

.. code-block:: text

   [DEBUG]: Compiled graph module for policy:
   graph():
       %obs : [num_users=1] = placeholder[target=obs]
       %linear : [num_users=1] = call_module[target=linear](args = (%obs,), kwargs = {})
       %tanh : [num_users=1] = call_function[target=torch.tanh](args = (%linear,), kwargs = {})
       return tanh
   [DEBUG]: Graph module inputs: [...]
   [DEBUG]: Graph module outputs: [...]

This is the exact FX graph LEAPP hands to the export backend. It is often
the fastest way to answer questions like:

* Did my annotated inputs become FX placeholders?
* Did an operation trace as the operator I expected?
* Was an input trimmed because it was not used by the output?
* Are the graph outputs the values I passed to ``annotate.output_tensors()``?

Dry run and selective non-traced nodes
======================================

LEAPP provides two related options for building the graph without
exporting every node, both declared on ``leapp.start()``:

* ``leapp.start(..., dry_run=True)`` traces normally but skips export for
  every node.
* ``leapp.start(..., non_traced=[...])`` skips export for only selected
  nodes.

Both still trace. Tracing is what produces the traced values that carry
graph connectivity, so it cannot be skipped while still producing a graph.

.. note::

   Earlier versions of ``dry_run`` and ``non_traced`` also disabled
   tracing, which let a node whose internals could not be traced still
   appear in the graph. That is no longer supported, and a tracing failure
   is now reported instead of being silently skipped.

``start(dry_run=True)``: skip export for the whole graph
--------------------------------------------------------

Use this when you want to explore graph boundaries, graph I/O, and
connectivity without paying export cost. Export is normally the slow step
for large models, while tracing runs the annotated code once and is
comparatively cheap.

In this mode:

* ``input_tensors()`` and related APIs return ``TracedTensor`` values,
  exactly as in a normal session
* FX graphs are still built, so connectivity is detected the same way
* YAML and graph structure are still produced
* model files are not exported

Warp graphs still require you to run the annotated path a second time, and
still report a segment that diverges between the two runs. Only the APIC
capture itself is skipped, because nothing consumes the captured bundle when
the node is not exported. Keeping the second run means a dry run tells you
whether your code would satisfy Warp export requirements. Nodes that reach
``export_with=None`` any other way behave the same.

.. code-block:: python

   leapp.start("debug_graph", dry_run=True)
   # ... run your traced code ...
   leapp.stop()
   leapp.compile_graph()

Useful for:

* debugging node boundaries
* checking graph I/O quickly
* validating connectivity before expensive export

``non_traced=[...]``: selective non-exported nodes
--------------------------------------------------

Use this when only some nodes should stay in the graph but should not be
exported, for example while iterating on one node and not wanting to pay
export cost for its neighbours.

With ``non_traced=[...]``, LEAPP still traces the listed node and still
connects it to its neighbours; it simply produces no model artifact.

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate

   leapp.start("mixed_graph", non_traced=["raw_node"])

   x = annotate.input_tensors("raw_node",
                              {"x": torch.tensor([1.0, 2.0, 3.0])})
   raw_y = x * 2.0
   annotate.output_tensors("raw_node",
                           {"y": raw_y},
                           export_with="jit")

   traced_y = annotate.input_tensors("traced_node", {"y": raw_y})
   traced_z = traced_y + 1.0
   annotate.output_tensors("traced_node",
                           {"z": traced_z},
                           export_with="jit")

   leapp.stop()
   leapp.compile_graph(validate=True)

Result:

* ``raw_node`` appears in the graph
* ``raw_node`` outputs still connect to ``traced_node``
* ``raw_node`` does not produce a compiled model artifact
* ``traced_node`` is still traced and exported normally

Choosing the right option
-------------------------

* Use ``start(dry_run=True)`` when no node should be exported.
* Use ``non_traced=[...]`` when only specific nodes should skip export.
* Use ``export_with=None`` on a single ``output_tensors()`` call when only
  that node should skip export.

Related debugging tools
=======================

* On Python 3.11+, ``leapp.compile_graph(visualize=True)`` writes a PNG
  graph image that is useful for checking node connectivity. Python 3.10
  emits a warning and skips this artifact.
* ``leapp.compile_graph(validate=True, strict=True, rtol=..., atol=...)``
  compares exported model outputs against captured outputs. See
  :doc:`runtime`.
* ``leapp.start(..., max_cached_io=...)`` controls how many re-entry examples
  LEAPP keeps for validation. This is useful when nodes run repeatedly or
  carry state. See :doc:`runtime`.
* ``leapp.start(..., global_patching=False)`` can help isolate issues caused
  by LEAPP's global tracing patches. Disable it only when you specifically
  need to test whether those patches are involved.
