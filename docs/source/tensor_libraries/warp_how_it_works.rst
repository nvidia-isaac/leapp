=============
How it works
=============

LEAPP does not rewrite Warp kernels as torch operations. It hands the work to
Warp's own APIC capture and stores the result, so LEAPP exports a subset of
what APIC already supports: if APIC cannot capture and replay a piece of Warp
code, LEAPP cannot export it either. This is in contrast to
:doc:`NumPy tracing <how_it_works>`, where each call is looked up and recorded
as an equivalent torch operation.

Segments
========

LEAPP captures runs of Warp calls rather than individual launches. A
**segment** is a consecutive run of Warp calls on values belonging to one
LEAPP node. It ends at a boundary:

* an explicit device synchronization, such as ``wp.synchronize_device()``
* a conversion that leaves Warp: ``wp.to_torch()``, ``wp.from_torch()``,
  ``wp.array.numpy()``, or ``wp.from_numpy()``
* a Warp call on values owned by a different LEAPP node
* ``annotate.output_tensors()`` for the node

``wp.copy()`` and further launches on the same node's values continue the open
segment. ``annotate.warp_op()`` declares a segment explicitly instead, and
every Warp call inside the block belongs to it.

The two passes
==============

Each node containing Warp work runs twice before ``leapp.stop()``.

**Discovery.** LEAPP observes public ``warp.*`` calls. A call that receives a
traced Warp value opens a segment, and LEAPP records the call sequence, the
segment boundaries, and the arrays crossing them.

**Capture.** The same code runs again. LEAPP checks each call against what
discovery recorded, then wraps each segment in an APIC capture. A divergence
is reported as an error rather than recorded as a new segment.

A capture has to know where the segment ends before it can begin, which is why
one execution is not enough.

What gets saved
===============

Each captured segment becomes one ``leapp::warp_runner`` operation in the
exported graph. LEAPP packs that segment's APIC archive together with the Warp
modules it compiled into a single binary blob, then embeds the blob in the
model as a constant input alongside the shapes and dtypes of the arrays
crossing the boundary. The exported artifact carries the Warp program with it
and reads nothing back from the machine that traced it.

At inference, LEAPP's native custom operator library unpacks the bundle and
replays the capture: ``com.nvidia.warp::WrpRunner`` under ONNX Runtime, and a
torch custom operator under PT2. Both libraries come from
``leapp-build-warp-runtime``; see :doc:`installation`.

See :doc:`warp_limitations` for what this model rules out.
