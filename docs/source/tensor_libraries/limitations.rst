===========
Limitations
===========

NumPy tracing records a torch graph from a lookup of equivalent operations.
Calls that are not in that lookup, and values that leave the NumPy array type,
are not represented in the exported model. This page covers the cases that
follow from that.

Untraced results become baked-in constants
==========================================

This is the failure to watch for, because nothing stops it.

When a NumPy call is not traced, its result is an ordinary ``np.ndarray`` with
no link to the graph. If that value is later combined with traced data, LEAPP
records it the way it records any other plain array: as a **constant**. The
exported model then returns the value captured during tracing, for every input.

Unmapped calls log a warning and then continue:

.. code-block:: text

   No torch equivalent for numpy function <name>. Operation will not be traced.

Treat that warning as an error.

.. code-block:: python

   state = annotate.input_tensors("node", {"state": raw_state})

   # np.interp has no torch equivalent, so this line is not recorded.
   calibrated = np.interp(state, CALIB_IN, CALIB_OUT)

   result = state * 0.1 + calibrated

   annotate.output_tensors("node", {"result": result}, export_with="jit")

That node exports without an error. Replay validation passes as well, because
validation replays the tracing inputs, which are the inputs the constant was
captured from. The graph is wrong for every other input.

LEAPP only raises when the untraced value is a node output itself:

.. code-block:: text

   output_tensors() for node 'node' received non-traced tensors: {'ndarray'}

.. warning::

   Read the tracing log. An untraced NumPy call in the middle of a node is
   reported once, as a warning, and never again.

Third-party functions
=====================

The rule is not "NumPy yes, everything else no". It is whether the function
decomposes into operations LEAPP can see. A helper written in Python on top of
traced NumPy calls traces fine, even from another package:

.. code-block:: python

   def standardize(arr):
       centered = arr - np.mean(arr, axis=-1, keepdims=True)
       scale = np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True))
       return centered / (scale + 1e-6)

Anything that reaches the raw buffer instead is invisible to the tracer, and
its result feeds the constant-folding case above. That covers compiled
extensions such as **SciPy**, **OpenCV**, and **numba**, NumPy's own compiled routines
including ``np.fft``, ``np.linalg.inv``, ``np.interp``, and ``np.pad``, and
plain Python iteration over array elements.

Where a rewrite in traced operations is not practical, split the node:
call :func:`~leapp.annotate.output_tensors` before the untraceable step and
:func:`~leapp.annotate.input_tensors` after it, so the gap becomes an explicit
graph boundary that the deploying runtime has to fill.

Indexing down to a scalar breaks the chain
==========================================

NumPy returns a scalar, not an array, when indexing removes every axis. That
scalar is a plain ``np.float32``, so tracing stops there. Torch does not have
this problem, because ``tensor[0]`` is still a tensor.

.. code-block:: python

   row = x[0]          # 2-D input: TracedNpArray, one axis left
   corner = x[0, 0]    # 2-D input: np.float32, no longer traced

   corner = x[0:1, 0:1]  # TracedNpArray, shape (1, 1)

Only the keys that reduce to a scalar are affected, so this bites hardest on
1-D arrays, where ``x[0]`` is already a scalar. Keep at least one axis in the
key. Slicing that leaves an array behind, including steps, boolean masks,
index arrays, ``None``, and ``Ellipsis``, is traced normally.

Reversed slicing disagrees with the graph
=========================================

``x[::-1]`` is ordinary NumPy but has no torch equivalent, since torch does not
support negative strides. The eager value is reversed, the recorded graph is
not, and nothing reports it at trace time. Use an index array instead:

.. code-block:: python

   flipped = x[::-1]                                  # graph is not reversed
   flipped = x[np.arange(x.shape[0] - 1, -1, -1)]     # traced correctly

``leapp.compile_graph(validate=True)``, the default, does catch this one:
the exported model disagrees with the traced Python output on the same input.

NumPy scalar types are not valid operands
=========================================

Python scalars and array constants are recorded as graph values; NumPy's own
scalar types are not.

.. code-block:: python

   y = x * 2.0                        # ok
   y = x * np.array(2.0, np.float32)  # ok
   y = x * np.float32(2.0)            # NotImplementedError during tracing

This also arrives indirectly, since many NumPy calls return scalar types. For
example ``np.linalg.norm(x)`` without an ``axis`` returns a ``np.float32``, so
``x / np.linalg.norm(x)`` fails on the division rather than on the norm.

Extracting Python values
========================

``.item()``, ``.tolist()``, ``float()``, and ``int()`` return concrete values
captured at trace time and drop out of the graph. Each logs a warning.

Using an array in a boolean context, such as ``if array.any():``, records only
the branch taken during tracing and logs an error. An exported graph is a
static structure and cannot hold a data-dependent branch; use ``np.where`` for
element-wise choices, or split the node around the decision.

Casts are limited to the shared dtypes
======================================

``astype`` is recorded only for dtypes that exist in both libraries:
``float16``, ``float32``, ``float64``, ``int8``, ``int16``, ``int32``,
``int64``, ``uint8``, and ``bool``. Any other target dtype casts eagerly, warns,
and leaves the graph at the original dtype.

Guidance
========

Keep the NumPy portion of a pipeline to the element-wise, reduction, and shape
work close to the data source, and cross into torch as early as the pipeline
allows. That is where NumPy support pays for itself, and it keeps the surface
that has to be checked small.
