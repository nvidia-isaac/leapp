===========
Limitations
===========

NumPy function capture reuses ``torch._numpy`` implementations and checks their
captured ATen graphs against real NumPy. This expands function coverage, but
does not intercept every way that Python or a native extension can consume an
array. Calls that fail capture and values that escape the tracing hooks need
different handling.

Unsupported calls stop active capture
=====================================

An intercepted call with no implementation raises ``NotImplementedError``.
Capture and parity failures also raise an error instead of returning an eager
result that could silently become a graph constant. For example:

.. code-block:: python

   state = annotate.input_tensors("node", {"state": raw_state})
   spacing = np.spacing(state)  # no compatibility implementation

The diagnostic identifies the NumPy operation and node. Ordinary arrays and
inactive traced arrays continue to use native NumPy behavior.

Support is specific to the arguments and dtypes being captured:

* ``out=`` mutation and ``overwrite_input=True`` are rejected before modifying
  caller storage. General in-place NumPy functions are not supported by this
  adapter. Indexed assignment has a separate tracing path.
* Padding accepts static widths, constant mode, and a scalar fill value.
  Stable or structured sorting is rejected.
* Direct ufunc calls are supported through the adapter; methods such as
  ``np.add.reduce`` are not. Use the supported reduction function where
  appropriate, such as ``np.sum``.
* Shape, dtype, or numerical differences from NumPy cause a parity failure.
  For example, float32 FFT capture with NumPy 2.5 and PyTorch 2.14 is rejected
  because NumPy returns complex64 while the compatibility layer returns
  complex128. FFT dtype handling remains unresolved.

A successful capture does not establish support for every runtime value,
shape, device, or export backend. Compare exported results with NumPy on
changed inputs; validating only the capture sample can miss frozen values.

Untraced values can still become constants
==========================================

The error policy applies to calls that reach the dispatch hooks. A function
that accesses the raw buffer or otherwise bypasses those hooks can still
return an ordinary array with no graph reference. If that result is mixed
back into traced data, it can be recorded as a constant. Standalone array
constructors and writes into plain ``np.ndarray`` destinations are not made
traceable by the ATen adapter.

Declare runtime-dependent values as inputs, keep their computation in traced
operations, and check replay on different inputs. Intentional plain-array
constants, such as normalization statistics, remain valid graph operands.

Third-party functions
=====================

A helper written in Python on top of supported traced operations can be
captured even when it comes from another package:

.. code-block:: python

   def standardize(arr):
       centered = arr - np.mean(arr, axis=-1, keepdims=True)
       scale = np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True))
       return centered / (scale + 1e-6)

Code that accesses the raw buffer, including some SciPy, OpenCV, or numba
routines, may bypass tracing. Being implemented in native code is not itself
a reason a NumPy call fails: functions such as ``np.linalg.inv`` and supported
``np.pad`` calls are intercepted and captured through the compatibility layer.

When a faithful replacement can be expressed in supported operations, use
:doc:`../guides/patching` to register it for the unsupported module-level
function. Existing helpers can keep their calls to that function; the adapter's
operations become part of the exported model. Replacements must be pure and
match the behavior used by the application. Validate them against the original
function on fresh inputs.

If a faithful function patch is not practical, split the node:
call :func:`~leapp.annotate.output_tensors` before the untraceable step and
:func:`~leapp.annotate.input_tensors` after it, so the gap becomes an explicit
graph boundary that the deploying runtime has to fill.

Indexing down to a scalar breaks the chain
==========================================

NumPy returns a scalar when indexing removes every axis. LEAPP still returns
that plain scalar, losing its runtime dependency. This remains a separate
limitation from function capture:

.. code-block:: python

   row = x[0]             # 2-D input: TracedNpArray, one axis left
   corner = x[0, 0]       # plain NumPy scalar, no longer traced
   result = x + corner   # corner can become a capture-time constant

   corner = x[0:1, 0:1]   # retains an array and its graph reference

Keep at least one axis in the index when its value must remain dynamic.
In contrast, supported reductions such as ``np.mean(x)`` or
``np.linalg.norm(x)`` can return traced zero-dimensional arrays. Constant
NumPy scalar operands such as ``np.float32(2.0)`` are also accepted; accepting
a scalar operand does not restore a dependency already lost by indexing.

Reverse slices require static basic indices
===========================================

Negative-stride reads are supported for fixed input shapes and static basic
indices:

.. code-block:: python

   flipped = x[::-1]       # full reversal lowers to torch.flip
   selected = x[5:0:-2]   # normalized indices lower to torch.index_select

Basic tuples can combine these slices with integers, ``None``, and
``Ellipsis``. Runtime slice bounds and advanced indices combined with negative
strides are rejected during capture. Reverse assignment and replay of
mutations through reversed views are not supported by this read lowering.
The eager NumPy result still has NumPy view/storage behavior; that alone does
not establish alias-mutation behavior in the exported graph.

Extracting Python values
========================

``.item()``, ``.tolist()``, ``float()``, and ``int()`` return concrete values
captured at trace time and drop out of the graph. ``.item()`` and ``.tolist()``
log warnings; do not rely on warnings to detect every scalar conversion.

Using an array in a boolean context, such as ``if array.any():``, records only
the branch taken during tracing and logs an error. LEAPP's capture does not
preserve data-dependent Python branches; use ``np.where`` for element-wise
choices, or split the node around the decision.

Casts are limited to the shared dtypes
======================================

``astype`` is recorded only for dtypes that exist in both libraries:
``float16``, ``float32``, ``float64``, ``int8``, ``int16``, ``int32``,
``int64``, ``uint8``, and ``bool``. Any other target dtype casts eagerly, warns,
and leaves the graph at the original dtype. This method-level behavior is
separate from the function adapter's dtype parity checks.
