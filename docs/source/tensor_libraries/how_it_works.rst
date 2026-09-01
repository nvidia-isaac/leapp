=============
How it works
=============

``TracedNpArray`` subclasses ``np.ndarray``, so it holds real array data and
behaves like an array everywhere LEAPP is not involved. It implements NumPy's
two dispatch hooks, ``__array_ufunc__`` and ``__array_function__``, which lets
it see every NumPy call made on it before NumPy executes it.

Each intercepted call does two things:

#. Runs the original NumPy operation eagerly on the underlying buffer, so the
   value you get back is exactly what NumPy would have produced.
#. Looks the NumPy function up in a translation table and records the
   **equivalent torch operation** into the FX graph.

So ``np.clip`` on a traced array returns a clipped NumPy array and records
``torch.clamp``. Given this node:

.. code-block:: python

   state, velocity = annotate.input_tensors("preprocess", {
       "state": frame["observation.state"],
       "velocity": frame["observation.velocity"],
   })

   state_norm = np.clip((state - STATE_MEAN) / STATE_STD, -5.0, 5.0)
   obs = np.concatenate([state_norm, velocity])

LEAPP records a graph that mentions no NumPy at all:

.. code-block:: text

   %state             = placeholder[target=state]
   %velocity          = placeholder[target=velocity]
   %_tensor_constant0 = get_attr[target=_tensor_constant0]
   %sub               = call_function[target=torch.sub](args = (%state, %_tensor_constant0))
   %_tensor_constant1 = get_attr[target=_tensor_constant1]
   %div               = call_function[target=torch.div](args = (%sub, %_tensor_constant1))
   %clamp             = call_function[target=torch.clamp](args = (%div, -5.0, 5.0))
   %cat               = call_function[target=torch.cat](args = ([%clamp, %velocity],))

Two details in that graph matter for :doc:`limitations`. Plain NumPy arrays
used as operands, here ``STATE_MEAN`` and ``STATE_STD``, become frozen
``get_attr`` constants. And each recorded node comes from a lookup, so a
NumPy call with no entry in the table records nothing.

What gets traced
================

Tracing is driven by an explicit NumPy-to-torch table, so support is a fixed
list rather than a general rule.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Category
     - Covered
   * - Arithmetic and math
     - ``+ - * / // % **``, ``np.add`` and the other arithmetic ufuncs,
       ``np.abs``, ``np.sqrt``, ``np.square``, ``np.exp``, ``np.log``,
       trigonometric and hyperbolic functions, ``np.floor``, ``np.ceil``,
       ``np.round``
   * - Comparison and logic
     - ``== != < <= > >=``, ``np.maximum``, ``np.minimum``, the logical and
       bitwise ufuncs, ``np.isnan``, ``np.isinf``, ``np.isfinite``
   * - Reductions
     - ``np.sum``, ``np.mean``, ``np.std``, ``np.var``, ``np.min``, ``np.max``,
       ``np.argmin``, ``np.argmax``, ``np.cumsum``, ``np.all``, ``np.any``,
       with ``axis`` and ``keepdims``
   * - Shape and layout
     - ``np.reshape``, ``np.transpose``, ``.T``, ``np.squeeze``,
       ``np.expand_dims``, ``np.concatenate``, ``np.stack``, ``np.split``,
       ``np.swapaxes``, ``np.moveaxis``, ``np.roll``
   * - Selection
     - ``np.where``, ``np.clip``, ``np.sort``, ``np.argsort``, ``np.nonzero``,
       slicing, boolean masks, index arrays, indexed assignment
   * - Linear algebra
     - ``np.matmul``, ``@``, ``np.dot``, ``np.einsum``, ``np.tensordot``,
       ``np.trace``, ``np.tril``, ``np.triu``
   * - Array creation
     - ``np.zeros_like``, ``np.ones_like``, ``np.full_like``, ``np.zeros``,
       ``np.ones``, ``np.arange``, ``np.linspace``, ``np.eye``
   * - Conversion
     - ``astype``, ``.copy()``, ``np.asarray``, ``torch.from_numpy``,
       ``torch.as_tensor``, ``TracedTensor.numpy()``

Anything outside the table runs normally and is skipped by the tracer, with a
warning naming the call:

.. code-block:: text

   No torch equivalent for numpy function <name>. Operation will not be traced.

Treat that warning as an error. See :doc:`limitations`. After tracing,
run ``compile_graph(validate=True)`` so a later sample can catch an
untraced result that was frozen as a constant. The workflow is on
:doc:`/guides/debugging`.
