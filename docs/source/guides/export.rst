=====================
Export configuration
=====================

This guide covers export backend selection and advanced export options.

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
   * - ``None``
     - ``NoneExportBackend``
     - No compilation

Recommended defaults:

* start with ``"jit"`` for the fastest bring-up
* use ``"onnx"`` when you want the default ONNX exporter
* use ``"onnx-torchscript"`` for recurrent models such as ``nn.GRU`` and
  ``nn.LSTM``

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

* prebuilt ``.pt`` or ``.onnx`` artifacts
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
* ``InferenceManager`` currently only runs referenced ``.pt`` and
  ``.onnx`` artifacts.

Dry run and selective non-traced nodes
======================================

LEAPP provides three related options for keeping graph structure without
fully compiling every node:

* ``leapp.start(..., dry_run=True)`` makes the entire trace metadata-only
  from the start.
* ``leapp.start(..., non_traced=[...])`` disables tracing/export for only
  selected nodes.
* ``leapp.compile_graph(..., dry_run=True)`` keeps an already-captured
  trace but skips compile/save/validate.

``start(dry_run=True)``: whole-graph metadata-only
--------------------------------------------------

Use this when you want to explore graph boundaries, graph I/O, and
connectivity without paying export cost.

In this mode:

* ``input_tensors()`` and related APIs return normal tensors instead of
  ``TracedTensor``
* LEAPP still tags outputs so graph connectivity can be detected
* YAML and graph structure are still produced
* model files are not exported

.. code-block:: python

   leapp.start("debug_graph", dry_run=True)
   # ... run your traced code ...
   leapp.stop()
   leapp.compile_graph()

Useful for:

* debugging node boundaries
* checking graph I/O quickly
* validating connectivity before expensive export

``non_traced=[...]``: selective non-compiled nodes
--------------------------------------------------

Use this when only some nodes should stay in the graph but should not be
traced through or compiled. This is especially useful because
traced-tensor nodes normally try to trace through the computation inside
the node, which can be problematic when:

* the code calls into functionality that is not trace-friendly
* the node intentionally acts as a placeholder or opaque stage
* tracing through that node raises errors even though you still want it
  represented in the graph

With ``non_traced=[...]``, LEAPP lets that node run on normal tensors,
skips export for it, but still tags its outputs so downstream traced
nodes can connect to it.

.. code-block:: python

   import torch
   import leapp
   from leapp import annotate

   leapp.start("mixed_graph", non_traced=["raw_node"])

   x = annotate.input_tensors("raw_node",
                              {"x": torch.tensor([1.0, 2.0, 3.0])})
   raw_y = x * 2.0
   annotate.output_tensors("raw_node",
                           {"y": raw_y},
                           export_with="jit")

   traced_y = annotate.input_tensors("traced_node", {"y": raw_y})
   traced_z = traced_y + 1.0
   annotate.output_tensors("traced_node",
                           {"z": traced_z},
                           export_with="jit")

   leapp.stop()
   leapp.compile_graph(validate=True)

Result:

* ``raw_node`` appears in the graph
* ``raw_node`` outputs still connect to ``traced_node``
* ``raw_node`` does not produce a compiled model artifact
* ``traced_node`` is still traced and exported normally

``compile_graph(dry_run=True)``: skip export after tracing
----------------------------------------------------------

Use this when you want a normal trace session first but want to skip
compile/save/validate at the final export step.

.. code-block:: python

   leapp.start("captured_graph")
   # ... normal tracing ...
   leapp.stop()
   leapp.compile_graph(dry_run=True, validate=False)

This differs from ``start(dry_run=True)``:

* tracing still happens normally during the session
* FX graphs and node traces are still built
* compile/save/validate are skipped only at the end

Choosing the right option
-------------------------

* Use ``start(dry_run=True)`` when the whole graph should be metadata-only.
* Use ``non_traced=[...]`` when only specific nodes should stay uncompiled.
* Use ``compile_graph(dry_run=True)`` when you already did a real trace and
  only want to skip final export work.
