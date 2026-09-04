=========
Debugging
=========

LEAPP provides multiple lines of defense against an incorrectly traced
graph. The goal is confidence that the traced graph is numerically
correct.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - When
     - Use
   * - You want numeric confidence in an export
     - ``compile_graph(validate=True)``
   * - Validation failed and you need the recorded ops
     - ``log.txt`` and the FX dump
   * - You need to tell tracing from export from patching
     - ``dry_run``, ``non_traced``, ``global_patching=False``

Confirm the export
==================

``compile_graph(validate=True)`` checks that each exported node matches
the traced Python on the inputs captured during tracing. LEAPP runs the
exported model, compares outputs with
``torch.allclose(..., rtol=..., atol=...)``, logs any deviation, and
returns a dict mapping node names to results.

.. code-block:: python

   leapp.compile_graph(
       validate=True,
       rtol=1e-3,
       atol=1e-5,
       strict=True,
   )

``strict=True`` raises if any node fails. ``strict=False`` still runs
validation, but leaves the result dict available. Nodes with no compiled
model, such as ``non_traced`` or dry-run cases, are skipped and treated
as successful.

A mismatch log includes the node and output names, the sample index,
tolerances, shapes and dtypes, value ranges, and absolute-difference
percentiles. Several samples are what make those numbers meaningful:
one pair of tensors cannot characterize error, and it cannot show
whether a value was captured because it is computed or because it
happened to be constant on the first step.

Run more than one step
----------------------

.. warning::

   Loop the annotated policy. A graph that matches the first observation
   can still have inlined a live input as a constant, and a single
   sample says nothing about error on any other input.

.. code-block:: python

   leapp.start(name="policy_graph", max_cached_io=5)
   for _ in range(5):
       obs = annotate.input_tensors("policy", {"obs": next_obs()})
       action = policy(obs)
       annotate.output_tensors(
           "policy", {"action": action}, export_with="jit",
       )
   leapp.stop()
   leapp.compile_graph(validate=True, strict=True)

Each iteration is stored as a sample, up to ``max_cached_io``.
``compile_graph(validate=True)`` replays the export against every one.
Use inputs that look like deployment: different state, commands, and
timing, not copies of the first frame. If a later step fails while the
first passes, the log names that ``sample N``.

Re-entry also checks names, shapes, dtypes, and connectivity against
the first step. Those failures show up during tracing, not at
``compile_graph()``.

Inspect what was captured
=========================

Every ``leapp.start()`` call writes a LEAPP log file in the graph output
directory:

.. code-block:: python

   leapp.start("debug_policy", save_path="exports")

creates ``exports/debug_policy/log.txt``. The file includes DEBUG
output. The console is quieter and shows warnings and errors. Pass
``verbose=True`` to stream the same detail to the console:

.. code-block:: python

   leapp.start("debug_policy", save_path="exports", verbose=True)
   # ... run your traced code ...
   leapp.stop()
   leapp.compile_graph(validate=True, verbose=True)

``leapp.start(..., verbose=True)`` covers trace-time diagnostics.
``leapp.compile_graph(..., verbose=True)`` covers compilation, export,
and validation.

FX graph
--------

For each traced node, LEAPP writes the full ``torch.fx.Graph`` to the
log after building the node's ``fx.GraphModule``. Search for
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

This is the graph LEAPP hands to the export backend. Use it to check
that annotated inputs became placeholders, that operations recorded as
the operators you expected, that unused inputs were trimmed, and that
the graph outputs match what you passed to ``annotate.output_tensors()``.

Graph image
-----------

On Python 3.11+, ``leapp.compile_graph(visualize=True)`` writes a PNG
of the node graph, which is the fastest check that nodes connected the
way you intended and the dtypes are as you expect.

Isolate the failure
===================

These flags turn pieces of a LEAPP session off so you can tell whether
a bad result comes from tracing, from export, or from LEAPP's own
patches.

Skip export
-----------

Tracing is what produces the values that carry graph connectivity, so
it cannot be skipped while still producing a graph. Export can.

* ``leapp.start(..., dry_run=True)`` traces every node and writes YAML,
  but exports no models. Use it to inspect boundaries, I/O, and
  connectivity without paying export cost.
* ``leapp.start(..., non_traced=["some_node"])`` does the same for
  listed nodes only: they still appear in the graph and still connect
  to neighbors, but they produce no model artifact.
* ``export_with=None`` on a single ``output_tensors()`` call skips
  export for that node only.

Warp graphs still run the annotated path a second time and still report
a segment that diverges between the two runs. Only APIC capture is
skipped, because nothing consumes the captured bundle when the node is
not exported.

.. code-block:: python

   leapp.start("debug_graph", dry_run=True)
   # ... run your traced code ...
   leapp.stop()
   leapp.compile_graph()

Disable session patches
-----------------------

``leapp.start(..., global_patching=False)`` turns off the conversions
LEAPP patches for the tracing session (for example NumPy and torch
interop). Use this only to test whether those patches are involved in
a failure.
