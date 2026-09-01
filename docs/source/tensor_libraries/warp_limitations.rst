===========
Limitations
===========

Warp support is built on APIC capture and replay, so the restrictions below
follow from what LEAPP has to guarantee for a capture to be replayable in an
exported model. Plain Warp restrictions that apply with or without LEAPP are
not repeated here.

The two passes must match
=========================

The discovery pass and the capture pass have to encounter the same Warp
regions, in the same order, with the same calls. Warp control flow that
depends on the data therefore cannot be exported, because the second pass
would take a different path.

Each way of breaking this is reported:

.. list-table::
   :header-rows: 1
   :widths: 38 24 38

   * - Cause
     - Raised at
     - Message
   * - Node executed only once
     - ``compile_graph()``
     - ``not executed a second time for APIC capture``
   * - Capture pass skips a discovered region
     - ``compile_graph()``
     - ``not executed a second time for APIC capture``
   * - Capture pass adds a region
     - the second execution
     - ``Warp capture encountered more regions than discovery``
   * - Capture pass calls different Warp operations
     - the second execution
     - ``Warp segment diverged between discovery and capture``

Linux and CUDA only
===================

Entering a Warp region on any other platform raises ``ImportError``:

.. code-block:: text

   LEAPP: Warp tracing is only available on Linux.

CPU Warp arrays are a quieter case. Tracing and capture succeed, but the
exported Warp runner requires a CUDA device, so the export fails at the
validation step that ``leapp.compile_graph()`` runs by default:

.. code-block:: text

   Model validation failed for 1 node(s): ['node_a']

Only two export backends
========================

A node containing a Warp segment must use ``export_with="onnx"`` or
``export_with="pt2"``. The bundle is embedded as a constant input, which the
other backends cannot represent, so they raise ``NotImplementedError`` naming
the two that work.

Boundary dtypes
===============

Arrays crossing a segment or node boundary must have a primitive scalar dtype
or a densely packed homogeneous compound dtype such as ``wp.vec3``, a
quaternion, or a matrix. Compound dtypes are described in expanded scalar
torch layout, so a logical ``(2, 4)`` array of ``wp.vec3`` is recorded as a
``(2, 4, 3)`` ``float32`` tensor.

Heterogeneous ``@wp.struct`` arrays have no such layout and are rejected with
``Unsupported dtype``.

One node per Warp call
======================

A single Warp call cannot consume traced arrays owned by different LEAPP
nodes, because the call would have to belong to two segments at once:

.. code-block:: text

   <call> received traced Warp arrays from different LEAPP trace contexts.

Reaching into another node's values from inside an explicit
``annotate.warp_op()`` block fails differently, since the open block cannot be
closed by anything but its own context manager:

.. code-block:: text

   Cannot begin WarpOp because the active WarpOp is protected by an owner token.

``warp_op()`` uses only the node name
=====================================

``annotate.warp_op()`` accepts ``inputs``, ``outputs``, and further keyword
arguments for symmetry with the other annotations, but Warp capture reads none
of them. The capture also always runs on ``cuda:0``, so an explicit block is
not a way to target a second GPU.

Inside the block, LEAPP protects the segment rather than policing it. A
synchronization or host readback that would close an automatic segment only
logs a warning and leaves the explicit block open, so the capture keeps
growing. Keep the block limited to Warp work that is meant to be replayed.

What is not intercepted
=======================

LEAPP patches public ``warp.*`` callables. Private ``warp._*`` entry points and
the kernel-language symbols used inside a ``@wp.kernel`` body are not patched.
A Warp call that receives no traced Warp value opens no segment and leaves
nothing in the graph, so Warp work on arrays that never passed through
:func:`~leapp.annotate.input_tensors` is simply absent from the export.

Warp patching is installed by ``leapp.start(..., global_patching=True)``, the
default. It also disables torch's CUDA caching allocator for the rest of the
process, which can slow down torch allocations after a Warp trace.

Running the export needs the native library
===========================================

An exported Warp graph is not self-executing: it needs the custom operator
library that matches the backend. The PT2 path fails at ``compile_graph()``
and the ONNX path fails at inference, both with a message pointing at the
build command:

.. code-block:: text

   LEAPP Warp pt2 runtime was not found at <path>. Run leapp-build-warp-runtime
   or set LEAPP_WARP_PT2_CUSTOM_OP_LIBRARY.

See :doc:`installation`.
