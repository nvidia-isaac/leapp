=============
How it works
=============

LEAPP does not rewrite Warp kernels as torch operations. It hands the work to
Warp's own APIC capture and stores the result, so LEAPP exports a subset of
what APIC already supports: if APIC cannot capture and replay a piece of Warp
code, LEAPP cannot export it either. This is in contrast to
:doc:`NumPy tracing <how_it_works>`, where each call is looked up and recorded
as an equivalent torch operation.

The unit LEAPP captures is a **segment**: a consecutive run of Warp calls on
values belonging to one LEAPP node. A segment is neither a single launch nor
the whole node. Where one ends and the next begins is decided by the code
sitting between the launches, so LEAPP has to work the boundaries out before
it can capture anything.

That is why a Warp pipeline runs its annotated path twice before
``leapp.stop()``. The first execution finds the boundaries, the second
captures each segment, and what survives is an ordinary torch graph carrying
one node per segment. The tabs below follow the same example through all
three steps: one node taking two inputs, with a few torch operations and
three ``wp.launch()`` calls between them.

.. tab-set::

   .. tab-item:: Pass 1

      .. raw:: html
         :file: ../_static/images/warp_pass_one_discovery.svg

      The first execution only watches. Every public ``warp.*`` call runs
      normally and returns a real result, so the Python code behaves exactly
      as it would without LEAPP.

      What LEAPP does alongside that is bookkeeping. A Warp call that
      receives a traced Warp value opens a segment, and LEAPP notes the call
      sequence, the arrays crossing in and out, and the point where the
      segment closes. `What breaks a segment`_ below lists everything that
      closes one.

      The diagram shows two of them. The first ``wp.launch()`` opens a
      segment and the one after it joins the same segment, because nothing in
      between is a boundary. The torch function then closes it. The third
      launch opens a second segment, which the node output closes.

      Nothing is captured during this pass and nothing reaches the graph yet.
      At the end of it LEAPP knows only where each segment starts and stops.

   .. tab-item:: Pass 2

      .. raw:: html
         :file: ../_static/images/warp_pass_two_capture.svg

      The same code runs a second time along the same path. Because the
      boundaries are now known, LEAPP can open an APIC capture at the start of
      each segment and close it at the end, shown by the dashed purple regions
      in the diagram.

      This is the step that cannot be folded into the first pass. A capture
      has to be opened before the first call it records, so LEAPP would have
      to know where the segment ends before it has seen the code that ends it.
      Running once to look and once to record is what resolves that.

      While capturing, LEAPP checks each call against what the first pass
      recorded. If the second execution takes a different Warp control-flow
      path, adds a region, or calls different Warp operations, that is
      reported as an error rather than quietly recorded as something new. The
      captured program is therefore always the one that was discovered.

      The result of this pass is one APIC archive per segment, holding the
      Warp program in replayable form.

   .. tab-item:: Final result

      .. raw:: html
         :file: ../_static/images/warp_recorded_graph.svg

      Each captured segment collapses into a single ``leapp::warp_runner``
      operation, the purple nodes in the diagram. The torch work around them
      is recorded operation by operation, exactly as it would be in a pipeline
      with no Warp in it.

      The kernels themselves never appear. A segment that ran three launches
      and one that ran a single launch both arrive as one node, because the
      graph refers to the captured program rather than describing it.

      To make that node portable, LEAPP packs the segment's APIC archive
      together with the Warp modules it compiled into one binary blob and
      embeds the blob in the model as a constant input, alongside the shapes
      and dtypes of the arrays crossing the boundary. The exported artifact
      carries the Warp program with it and reads nothing back from the machine
      that traced it.

      At inference, LEAPP's native custom operator library unpacks the bundle
      and replays the capture: ``com.nvidia.warp::WrpRunner`` under ONNX
      Runtime, and a torch custom operator under PT2. Both libraries come from
      ``leapp-build-warp-runtime``; see :doc:`installation`.

What breaks a segment
=====================

A segment is a consecutive run of Warp calls. Anything LEAPP cannot record
inside the capture closes the open segment, and the next Warp call starts a
new one.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Cause
     - What happens
   * - A recorded torch operation
     - Adding a node to the traced graph closes the segment first. One
       ``torch.clamp()`` in the middle of a Warp block splits it in two.
   * - Reading data out of Warp
     - ``wp.to_torch()`` and ``wp.array.numpy()`` end the segment.
       ``wp.from_torch()`` and ``wp.from_numpy()`` do not.
   * - Device synchronization
     - ``wp.synchronize()``, ``wp.synchronize_device()``,
       ``wp.synchronize_event()``, and ``wp.synchronize_stream()`` close it.
   * - Unmanaged CUDA work
     - Kernel launches, allocations, copies, fills, recorded CUDA events, and
       stream waits from outside Warp close the segment. A capture can only
       replay the Warp program it recorded.
   * - Two LEAPP nodes collide
     - A segment belongs to one node. A Warp call on values owned by another
       node closes the open segment and starts a new one.
   * - Node outputs
     - ``annotate.output_tensors()`` closes whatever is still open for that
       node.
   * - An explicit ``warp_op()`` block
     - Entering the block closes any open segment. Leaving it closes the
       block's own segment.

``wp.copy()`` and further ``wp.launch()`` calls on the same node's values
close nothing. They extend the segment that is already open.

.. note::
   Group Warp calls together when you can. Each fragment becomes its own
   graph node with its own captured archive, so splitting one Warp block
   pays that cost more than once. Do torch work, conversions, and
   synchronization before or after the Warp calls, not between them.
   ``annotate.warp_op()`` declares a whole segment by hand when LEAPP
   cannot infer the grouping.

See :doc:`warp_limitations` for what this model rules out.
