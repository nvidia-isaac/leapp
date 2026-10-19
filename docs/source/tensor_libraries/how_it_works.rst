============
How it works
============

``TracedNpArray`` subclasses ``np.ndarray``, so it holds real array data.
Its ``__array_ufunc__`` and ``__array_function__`` hooks intercept NumPy calls
that participate in those protocols. During active tracing, LEAPP captures
those calls through PyTorch's tensor-backed NumPy compatibility layer,
``torch._numpy``.

From a NumPy call to ATen
=========================

For each intercepted function or direct ufunc call, LEAPP:

#. Selects an implementation from a small override registry, or discovers the
   corresponding callable in ``torch._numpy``. Discovery covers the NumPy,
   linear algebra, and FFT namespaces; it excludes the random namespace.
#. Captures that implementation with ``make_fx`` using tensor versions of the
   inputs, then functionalizes the captured graph to remove temporary
   in-place operations.
#. Compares the captured result with real NumPy on copies of the capture
   inputs. Shapes and dtypes must match, and values must agree within numerical
   tolerances. The captured graph must contain supported functional operations.
#. Inserts the captured ATen operations into the current LEAPP FX graph and
   wraps the eager NumPy result with the corresponding graph references.

ATen is PyTorch's tensor operator layer. One NumPy call can contribute several
ATen operations. The exported graph contains tensor operations, with no
``torch._numpy`` arrays or calls needed during replay. LEAPP still executes
real NumPy for the eager result, including its view/storage behavior.

The old handwritten NumPy-to-torch translation tables are removed. There is
no legacy fallback when capture or validation fails: the call raises an error
during active tracing. After the owning node finishes tracing, its arrays use
native NumPy behavior again.

For example, this preprocessing node stays in NumPy at the call site:

.. code-block:: python

   state, velocity = annotate.input_tensors("preprocess", {
       "state": frame["observation.state"],
       "velocity": frame["observation.velocity"],
   })

   state_norm = np.clip((state - STATE_MEAN) / STATE_STD, -5.0, 5.0)
   obs = np.concatenate([state_norm, velocity])

Its recorded computation contains ATen subtraction, division, clipping, and
concatenation. Schematically, omitting constant-copy nodes:

.. code-block:: text

   state, velocity = placeholders
   mean, std, lower, upper = graph constants
   sub   = torch.ops.aten.sub(state, mean)
   div   = torch.ops.aten.div(sub, std)
   clamp = torch.ops.aten.clamp(div, lower, upper)
   cat   = torch.ops.aten.cat([clamp, velocity])

Plain arrays such as ``STATE_MEAN`` and ``STATE_STD`` become graph constants.
If they must vary at runtime, declare them as inputs. Exact ATen sequences can
vary with the installed PyTorch version.

Small override registry
=======================

Overrides use the same capture and validation path as upstream implementations.
They cover specific semantic or export differences:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Override behavior
   * - ``np.cbrt``
     - Preserves the sign for negative inputs and signed zero.
   * - ``np.pad``
     - Supports static padding widths, constant mode, and a scalar fill value.
   * - ``np.nanmean``
     - Supplies the missing reduction and promotes integer inputs to float64.
   * - ``np.sort``, ``np.argsort``
     - Uses a fixed-size TopK lowering for export. Stable and structured sorting
       are rejected.

What gets traced
================

Coverage depends on the callable, arguments, dtype, installed PyTorch version,
and export backend. Finding a compatibility implementation is only the first
step; that particular call must also pass capture and NumPy parity checks.
Examples exercised by the tests include:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Category
     - Examples
   * - Arithmetic and comparison
     - Arithmetic ufuncs, ``np.sin``, ``np.sqrt``, ``np.cbrt``, and
       ``np.isclose`` feeding ``np.where``
   * - Reductions
     - ``np.sum``, ``np.mean``, ``np.nanmean``, ``np.var``, and
       ``np.quantile`` with supported axis and dtype arguments
   * - Shape and selection
     - ``np.reshape``, ``np.transpose``, ``np.concatenate``, ``np.diff``,
       ``np.broadcast_to``, constant padding, and sorting
   * - Linear algebra
     - ``np.matmul``, ``np.linalg.norm``, ``np.linalg.inv``, and ``np.cross``

Array methods, indexing, and NumPy/Torch conversions retain their own tracing
paths. Static basic reverse slices use ``torch.flip`` for full reversal and
normalized indices with ``torch.index_select`` for other negative strides.
Calls that never reach a traced operand, such as standalone array creation,
are not made traceable by compatibility-layer discovery.

These examples are not a promise of every NumPy signature or backend. The
adapter uses private PyTorch APIs, and capture-sample parity alone cannot prove
correctness for other values. Validate exported results on changed inputs as
well as the capture input; see :doc:`/guides/debugging` and :doc:`limitations`.
