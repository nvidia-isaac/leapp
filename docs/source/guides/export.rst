=====================
Export configuration
=====================

This guide covers export backend selection and advanced export options.
For dry-run and non-traced debugging modes, see :doc:`debugging`.

Backend names and aliases
=========================

LEAPP supports these public backend names:

.. list-table::
   :header-rows: 1
   :widths: 28 28 28

   * - ``export_with`` value
     - Actual backend
     - Output
   * - ``"jit"``
     - ``jit-script``
     - TorchScript ``.pt``
   * - ``"jit-script"``
     - ``jit-script``
     - TorchScript ``.pt``
   * - ``"jit-trace"``
     - ``jit-trace``
     - TorchScript ``.pt``
   * - ``"onnx"``
     - ``onnx-dynamo``
     - ONNX ``.onnx``
   * - ``"onnx-dynamo"``
     - ``onnx-dynamo``
     - ONNX ``.onnx``
   * - ``"onnx-torchscript"``
     - ``onnx-torchscript``
     - ONNX ``.onnx``
   * - ``"exported-program"``
     - ``exported-program``
     - ``torch.export`` ``.pt2``
   * - ``"pt2"``
     - ``exported-program``
     - ``torch.export`` ``.pt2``
   * - ``None``
     - ``NoneExportBackend``
     - No compilation

Recommended defaults:

* start with ``"jit"`` for the fastest bring-up
* use ``"onnx"`` when you want the default ONNX exporter
* use ``"onnx-torchscript"`` for recurrent models such as ``nn.GRU`` and
  ``nn.LSTM``
* use ``"exported-program"`` (or the ``"pt2"`` alias) when you want a
  ``torch.export`` artifact instead of TorchScript or ONNX

TorchScript export
==================

``"jit"`` and ``"jit-script"`` select the TorchScript scripting backend.
``"jit-trace"`` selects the tracing backend.

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate

   leapp.start("torchscript_example")

   x = annotate.input_tensors("normalize", {"x": torch.randn(16)})
   y = torch.relu((x - x.mean()) / (x.std() + 1e-6))
   annotate.output_tensors("normalize", {"y": y}, export_with="jit")

   leapp.stop()
   leapp.compile_graph(validate=True)

ExportedProgram export
======================

``"exported-program"`` exports each node with :func:`torch.export.export`
and saves a ``.pt2`` artifact via :func:`torch.export.save`. The YAML
``backend`` field is written as ``pt2``.

``"pt2"`` is a shorthand alias for ``"exported-program"``, similar to how
``"onnx"`` aliases ``"onnx-dynamo"``.

Use this backend when you want the modern ``torch.export`` graph
representation rather than TorchScript or ONNX. It is a good fit for
feedforward PyTorch graphs that already compile cleanly through LEAPP's
FX capture path.

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate

   leapp.start("exported_program_example")

   x = annotate.input_tensors("policy", {"obs": torch.randn(1, 32)})
   action = torch.tanh(x[..., :12])
   annotate.output_tensors(
       "policy",
       {"action": action},
       export_with="exported-program",
   )

   leapp.stop()
   leapp.compile_graph(validate=True)

ExportedProgram backend parameters
----------------------------------

All ExportedProgram backend parameters are passed through
``backend_params``.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``strict``
     - PyTorch default
     - Passed through to :func:`torch.export.export`
   * - ``dynamic_shapes``
     - ``None``
     - Optional dynamic-shapes spec for :func:`torch.export.export`

Example:

.. code-block:: python

   annotate.output_tensors(
       "policy",
       {"action": action},
       export_with="pt2",
       backend_params={
           "strict": True,
       },
   )

Runtime notes
-------------

* ``InferenceManager`` can load and run ``pt2`` artifacts.
* When CUDA is available, LEAPP loads exported programs onto the GPU.
  Pass runtime inputs on the same device, or use
  ``InferenceManager.get_mock_input()`` which preallocates tensors on the
  node device.
* Lifted constants baked into the exported graph during trace may remain on
  CPU even after ``module.to(device)``. Prefer ``annotate.register_buffer``
  for values that must travel with the model, or keep runtime inputs on
  the model device.
* The YAML records the exporting PyTorch version in ``torch_version``.

ONNX export
===========

LEAPP exposes two ONNX backends:

* ``onnx-dynamo`` is the default behind ``export_with="onnx"``
* ``onnx-torchscript`` is the TorchScript-based ONNX path

Use ``onnx-dynamo`` for typical feedforward models. Use
``onnx-torchscript`` when the dynamo path produces unstable graphs or when
exporting recurrent models (e.g. ``nn.GRU``, ``nn.LSTM``). See
``examples/stateful_gru_export.py`` for a complete example.

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate

   leapp.start("onnx_example")

   x = annotate.input_tensors("policy", {"obs": torch.randn(1, 32)})
   action = torch.tanh(x[..., :12])
   annotate.output_tensors("policy", {"action": action}, export_with="onnx")

   leapp.stop()
   leapp.compile_graph(validate=True)

ONNX backend parameters
-----------------------

All ONNX backend parameters are passed through ``backend_params``.

.. list-table::
   :header-rows: 1
   :widths: 20 15 25 40

   * - Parameter
     - Default
     - Used by
     - Description
   * - ``opset_version``
     - PyTorch default
     - both ONNX backends
     - Override the ONNX opset version
   * - ``report``
     - ``False``
     - both ONNX backends
     - Emit exporter diagnostics
   * - ``verify``
     - ``False``
     - ``onnx-dynamo``
     - Enable exporter-side verification
   * - ``optimize``
     - ``True``
     - ``onnx-dynamo``
     - Enable dynamo ONNX optimization
   * - ``prescript``
     - ``False``
     - ``onnx-torchscript``
     - Script the module before ONNX export
   * - ``skip_validation``
     - ``False``
     - both at save time
     - Skip ``onnx.checker.check_model()`` when saving

Example:

.. code-block:: python

   annotate.output_tensors(
       "policy",
       {"action": action},
       export_with="onnx-dynamo",
       backend_params={
           "opset_version": 17,
           "verify": False,
           "optimize": True,
           "report": True,
       },
   )

Validation guidance
-------------------

Exporter-side options like ``verify`` are optional and backend-specific.
The main LEAPP validation path is still:

.. code-block:: python

   leapp.compile_graph(validate=True, rtol=1e-3, atol=1e-5, strict=True)

That validation compares exported model outputs against the captured
traced outputs. See :doc:`runtime` for details.

Bringing your own model
=======================

Set ``export_with=None`` when you want a node to appear in the graph
without asking LEAPP to compile it. This is useful for:

* prebuilt ``.pt``, ``.pt2``, or ``.onnx`` artifacts
* placeholder nodes that will be filled in later
* flows where LEAPP should capture I/O and graph edges but not produce a
  model

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate

   leapp.start("precompiled_example")

   x = annotate.input_tensors("precompiled_inference",
                              {"input_data": torch.randn(1, 10)})
   predictions = x  # representative traced output shape

   annotate.output_tensors(
       "precompiled_inference",
       {"predictions": predictions},
       export_with=None,
       backend_params={
           "model_path": "/models/my_optimized_model.pt",
           "copy_original_model": True,
       },
   )

   leapp.stop()
   leapp.compile_graph()

``None`` backend parameters
---------------------------

* ``model_path`` (optional): Path to an existing model artifact.
* ``copy_original_model`` (optional, default ``False``): Copy the provided
  artifact into the LEAPP output directory.
* ``device`` (optional): Device hint used when loading the artifact.

Important behavior:

* LEAPP still records input/output shapes and graph connectivity from the
  traced example tensors.
* If ``model_path`` is provided, LEAPP verifies the file and stores
  checksums in the YAML.
* The model path written into the YAML is made relative to the YAML
  directory when possible.
* ``InferenceManager`` currently runs referenced ``.pt``, ``.pt2``, and
  ``.onnx`` artifacts.
