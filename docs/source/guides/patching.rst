=================
Function patching
=================

Use a function patch when an existing workflow calls a function that LEAPP
cannot trace, and you can express the same computation with supported
operations.

Function patching is especially useful when you want to avoid modifying the
original processing code. Define a traceable adapter in your LEAPP integration
and register it once at the start of tracing; existing helpers can keep calling
the original library function.

When a patch is useful
======================

Typical cases include:

* OpenCV calls such as ``cv2.cvtColor`` that compute in native code outside
  LEAPP's traced operations.
* Unsupported NumPy functions such as ``np.interp``, used to interpolate
  sensor readings from a calibration table.
* Library calls used in several places or nested inside existing helpers,
  where replacing every call would be inconvenient.

Python helpers built entirely from supported operations generally trace
directly. Mixing NumPy and torch also works when the operations and conversions
are supported; see :doc:`../tensor_libraries/numpy`.

An unsupported operation can return an ordinary array and break the trace.
If that array is combined with traced data later, its value can become a
constant in the exported model. See :doc:`../tensor_libraries/limitations`
for the warning messages and other ways tracing can be lost.

Existing OpenCV workflow
========================

In ``examples/function_patch.py``, the stereo workflow calls
``image_features()`` for each camera. That helper calls ``prepare_image()``,
which converts the frame from BGR to RGB:

.. literalinclude:: ../../../examples/function_patch.py
   :language: python
   :pyobject: prepare_image

One patch covers both camera calls without changing these helpers.

Replace the OpenCV conversion
=============================

For a three-channel image with shape ``[height, width, channels]``,
``cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`` changes the channel order from
``[B, G, R]`` to ``[R, G, B]``. The replacement performs that same operation
with torch:

.. literalinclude:: ../../../examples/function_patch.py
   :language: python
   :pyobject: traced_cvt_color

``torch.from_numpy(src)`` brings the image into torch, ``flip((-1,))``
reverses the last axis, and ``.numpy()`` returns the array expected by
``prepare_image()``. LEAPP can trace all three operations. The image's shape
and dtype stay the same.

The adapter supports only this BGR-to-RGB conversion. Its checks reject other
conversion codes, destination buffers, and incompatible shapes. A replacement
must match the original function's behavior for the arguments your workflow
supplies.

Register the patch
==================

Pass the module, function name, and replacement to ``FunctionPatch`` when
starting the tracing session:

.. literalinclude:: ../../../examples/function_patch.py
   :language: python
   :start-at:     leapp.start(
   :end-before:     try:
   :dedent: 4

Calls to ``cv2.cvtColor`` containing actively traced data now use
``traced_cvt_color``. Calls containing only ordinary arrays still use OpenCV.
The exported model executes the recorded torch operations at runtime.

Run the complete example
========================

From the repository root:

.. code-block:: bash

   python -m pip install opencv-python-headless
   python examples/function_patch.py

The script exports the workflow and compares replay on fresh camera frames
against the original OpenCV computation. Using new inputs helps detect values
that were accidentally captured as constants.

Patch scope and requirements
============================

* User patches require ``global_patching=True``, which is the default.
* Use calls through the module, such as ``cv2.cvtColor(...)``. References
  captured with ``from cv2 import cvtColor`` before tracing are not updated.
* Replacements must use operations LEAPP can trace and return the expected
  structure, shape, and dtype.
* Replacements must be pure. Mutation, destination arguments such as
  ``dst=``, input/output aliasing, and calling the patched function from
  its own replacement are unsupported.
* Patches affect the whole Python process during the session, including other
  threads. Stop tracing before starting a session with a different patch list.

See :doc:`../api/index` for the full ``leapp.start(patching=...)`` contract.
