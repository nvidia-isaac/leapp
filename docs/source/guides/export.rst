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
   * - ``"executorch"``
     - ``executorch``
     - ExecuTorch ``.pte``
   * - ``None``
     - ``NoneExportBackend``
     - No compilation


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


ExecuTorch export
=================

``"executorch"`` exports each node with :func:`torch.export.export`, lowers
it with `ExecuTorch <https://docs.pytorch.org/executorch/>`_, and saves a ``.pte``
program. The YAML ``backend`` field is written as ``pte``. Validation and
:class:`~leapp.InferenceManager` run the ``.pte`` file through ExecuTorch's
runtime, so they check the program a deployment runs. ExecuTorch programs
also run from C++ without Python or libtorch.

Install the ``executorch`` extra first (Python 3.11 or newer, since ExecuTorch
1.6 has no Python 3.10 wheels). It needs PyTorch 2.15, whose PyPI wheels use
CUDA 13, so with Warp add the ``warp-cu13`` extra in the same command, not
``warp-cu12``. The ``tensorrt`` delegate also needs Torch-TensorRT 2.15.0 or
newer with its ExecuTorch runtime. On a Jetson, its wheels need JetPack 7,
since they need glibc 2.39:

.. code-block:: bash

   pip install "leapp[executorch]"
   # with Warp:
   pip install "leapp[executorch,warp-cu13]"
   # for delegate="tensorrt":
   pip install "torch_tensorrt[executorch]>=2.15.0"

A TensorRT engine loads only with the TensorRT it was built with, so where
the program runs, install the Torch-TensorRT version that the YAML records as
``torch_tensorrt_version``.

.. code-block:: python

   annotate.output_tensors(
       "policy",
       {"action": action},
       export_with="executorch",
       backend_params={"delegate": "cuda"},
   )

ExecuTorch backend parameters
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``delegate``
     - ``"xnnpack"``
     - ``"xnnpack"`` runs on the CPU (x86 and Arm). ``"cuda"`` runs on an
       NVIDIA GPU with ExecuTorch's CUDA backend, on Linux only.
       ``"tensorrt"`` builds TensorRT engines with Torch-TensorRT, also on
       Linux only, and sends what TensorRT does not take to the CUDA backend.
   * - ``device_io``
     - ``True``
     - GPU delegates only. The program reads and writes GPU memory. ``False``
       makes it read and write CPU memory.
   * - ``tensorrt_options``
     - ``{}``
     - ``"tensorrt"`` only. Settings for
       :func:`torch_tensorrt.dynamo.compile`. They override LEAPP's own.
   * - ``strict``
     - PyTorch default
     - Passed through to :func:`torch.export.export`

LEAPP loads an XNNPACK program with the XNNPACK weight cache on, so a layer
called many times, such as in a loop, keeps one packed copy of its
weights. A C++ app gets the same if it sets the XNNPACK backend option
``weight_cache_enabled`` to true when it loads the program.

With every delegate, LEAPP returns a node's outputs on the device of its
inputs. A node with no inputs, such as one whose inputs were all pruned,
returns its outputs on the GPU when there is one, as TorchScript and
``torch.export`` nodes do. Later ExecuTorch and ONNX nodes follow their
inputs, so their outputs then move to the GPU too.

The ``"cuda"`` delegate needs a CUDA toolkit (``nvcc`` and the CUDA headers)
at export, and so does ``"tensorrt"``, for what it sends to the CUDA backend.
A GPU program is built for the first visible GPU, and LEAPP makes that GPU
current when it loads and runs the program; another runner, such as a C++ app,
must make it current itself. Choose the physical GPU with
``CUDA_VISIBLE_DEVICES``. The program holds native code for the export
machine's CPU architecture and GPU generation, so export on the target device
or on one like it. The CUDA backend
names its ``.ptd`` file after the node, so a GPU node needs a name ExecuTorch
accepts as a data name: up to 200 ASCII letters, digits, ``_``, ``-`` and
``.``, not starting with ``.`` or ``-``, not ending in ``.ptd``, and not a
Windows device name such as ``aux``.

Until the program is saved, Torch-TensorRT keeps the engines it built in files
in the temporary directory, so it needs room there. It puts them in a folder
named after your account, which another account could create first and write
into, so on a machine you share, set ``TMPDIR`` to a folder only you can
write. Torch-TensorRT's engine cache, which ``cache_built_engines`` and
``reuse_cached_engines`` turn on, is one folder in the temporary directory
that every account shares, so the same advice holds. It caches an engine only
when ``immutable_weights`` is ``False``.

``tensorrt_options`` go to :func:`torch_tensorrt.dynamo.compile`, for example
``{"timing_cache_path": "trt/timing_cache.bin"}``. They override LEAPP's own:
``min_block_size`` 1, ``disable_tf32``, ``fallback_pytorch_plugin_ops`` (no
plugin that needs libtorch), and a timing cache
that is private to each build and deleted after it, since Torch-TensorRT's
default cache is one file that every account on the machine shares. To reuse
timings between builds, set ``timing_cache_path`` to a file in a folder you
own; a bare file name fails after the first engine is built. LEAPP accepts
only the named settings of that function. So it also refuses an old name that
the function still reads, such as ``truncate_long_and_double`` (use
``truncate_double``). LEAPP refuses ``tensorrt_options`` with another delegate.
It also refuses these settings, because LEAPP sets the program, its inputs and
the GPU, and always builds the engines: ``exported_program``, ``inputs``,
``arg_inputs``, ``kwarg_inputs``, ``device`` and ``dryrun``. It refuses
``release_weights_after_build`` too, which frees the model's own weights when
they are on the GPU. LEAPP asks the runtime to replay each engine as a CUDA
graph.

Torch-TensorRT cannot keep a buffer or parameter the node writes. It also
cannot save one the node returns unchanged, such as a static output, even
through a step that changes nothing, such as ``detach()``. So LEAPP refuses
such a node with ``"tensorrt"``. Export it with ``"cuda"`` or ``"xnnpack"``,
or, if it writes a parameter, with another backend, since those two cannot
export that either. TensorRT has no float64, so a node that reads or computes
float64 needs ``{"truncate_double": True}``, which computes in float32 while
the inputs and outputs stay float64.

With ``device_io`` on (the default), a pipeline fed CUDA tensors passes them between nodes
without host copies, with two exceptions. An input the node returns
unchanged, and that no GPU operator reads, stays in host memory. An output
that is a slice of another tensor, such as ``x[:, ::2]`` or ``x[:, 0]``, comes
back in GPU memory, but the CUDA backend copies it through host memory on
every call, even with a batch of 1. A block of whole rows taken without a
step, such as ``x[1:3]`` or ``x[0]``, does not take that round trip. A C++ app
passes GPU tensors and sets GPU output buffers with ``Module::set_output``,
since the program does not allocate them. LEAPP's runtime does that, and moves
outputs to the device of the inputs.

The CUDA backend writes the node's weights to ``<node>.ptd`` beside the
``.pte``; keep the two files together. The YAML records its checksum as
``data_sha256sum``, and loading checks it the same way as the ``.pte``. A GPU
program holds compiled native code that runs when it is loaded or run, so
load ExecuTorch exports only from a source you trust.

LEAPP turns TensorRT's TF32 off, so TensorRT computes float32 in full
precision unless ``tensorrt_options`` sets ``disable_tf32`` to ``False``. The
CUDA backend compiles with the matrix multiply precision PyTorch has at
export, so keep ``torch.backends.cuda.matmul.allow_tf32`` at its default,
``False``, when exporting with it. On a GPU, PyTorch runs float32 convolutions
in TF32 by default, so the outputs LEAPP captures there can differ from the
program by about 1e-3. Set ``torch.backends.cudnn.allow_tf32 = False`` before
``leapp.start()``, or pass a larger ``atol`` to ``compile_graph()``. If
TensorRT cannot build a node, use ``delegate="cuda"`` for it.

Boolean masks such as ``x[x > 0]`` do not export with any delegate, and
the GPU delegates do not export ``nonzero``; both fail at export with an
error. Export such a node with another backend. The GPU delegates also refuse
a node that reads or returns a tensor with no elements: the CUDA backend cannot
run one, and with ``"tensorrt"`` LEAPP cannot tell in advance which parts
TensorRT takes. Export such a node with ``"xnnpack"`` or another backend. With
``device_io``, the CUDA backend also refuses a node that returns a buffer it
writes; export such a node with ``device_io=False``.


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
traced outputs. See :doc:`debugging` for validation details and
:doc:`/leapp_runtime` for Python runtime usage.

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

Important behavior:

* LEAPP still records input/output shapes and graph connectivity from the
  traced example tensors.
* If ``model_path`` is provided, LEAPP verifies the file and stores
  checksums in the YAML.
* The model path written into the YAML is made relative to the YAML
  directory when possible.
* ``InferenceManager`` currently runs referenced ``.pt``, ``.pt2``, and
  ``.onnx`` artifacts.
