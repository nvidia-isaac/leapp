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

LEAPP provides three related options for keeping graph structure without
fully compiling every node:

* ``leapp.start(..., dry_run=True)`` makes the entire trace metadata-only
  from the start.
* ``leapp.start(..., non_traced=[...])`` disables tracing/export for only
  selected nodes.
* ``leapp.compile_graph(..., dry_run=True)`` keeps an already-captured
  trace but skips compile/save/validate.

``start(dry_run=True)``: whole-graph metadata-only
--------------------------------------------------

Use this when you want to explore graph boundaries, graph I/O, and
connectivity without paying export cost.

In this mode:

* ``input_tensors()`` and related APIs return normal tensors instead of
  ``TracedTensor``
* LEAPP still tags outputs so graph connectivity can be detected
* YAML and graph structure are still produced
* model files are not exported

.. code-block:: python

   leapp.start("debug_graph", dry_run=True)
   # ... run your traced code ...
   leapp.stop()
   leapp.compile_graph()

Useful for:

* debugging node boundaries
* checking graph I/O quickly
* validating connectivity before expensive export

``non_traced=[...]``: selective non-compiled nodes
--------------------------------------------------

Use this when only some nodes should stay in the graph but should not be
traced through or compiled. This is especially useful because traced-tensor
nodes normally try to trace through the computation inside the node, which
can be problematic when:

* the code calls into functionality that is not trace-friendly
* the node intentionally acts as a placeholder or opaque stage
* tracing through that node raises errors even though you still want it
  represented in the graph

With ``non_traced=[...]``, LEAPP lets that node run on normal tensors,
skips export for it, but still tags its outputs so downstream traced nodes
can connect to it.

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

``compile_graph(dry_run=True)``: skip export after tracing
----------------------------------------------------------

Use this when you want a normal trace session first but want to skip
compile/save/validate at the final export step.

.. code-block:: python

   leapp.start("captured_graph")
   # ... normal tracing ...
   leapp.stop()
   leapp.compile_graph(dry_run=True, validate=False)

This differs from ``start(dry_run=True)``:

* tracing still happens normally during the session
* FX graphs and node traces are still built
* compile/save/validate are skipped only at the end

Choosing the right option
-------------------------

* Use ``start(dry_run=True)`` when the whole graph should be metadata-only.
* Use ``non_traced=[...]`` when only specific nodes should stay uncompiled.
* Use ``compile_graph(dry_run=True)`` when you already did a real trace and
  only want to skip final export work.

Related debugging tools
=======================

* ``leapp.compile_graph(visualize=True)`` writes SVG and PNG graph images
  that are useful for checking node connectivity.
* ``leapp.compile_graph(validate=True, strict=True, rtol=..., atol=...)``
  compares exported model outputs against captured outputs. See
  :doc:`runtime`.
* ``leapp.start(..., max_cached_io=...)`` controls how many re-entry examples
  LEAPP keeps for validation. This is useful when nodes run repeatedly or
  carry state. See :doc:`runtime`.
* ``leapp.start(..., global_patching=False)`` can help isolate issues caused
  by LEAPP's global tracing patches. Disable it only when you specifically
  need to test whether those patches are involved.
