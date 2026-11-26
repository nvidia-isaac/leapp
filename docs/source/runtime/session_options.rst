Session Options
===============

Set options when opening the session. Peers created by ``new_session`` inherit
those options; CUDA graph replay sessions require an independent session instead
of a peer. Per-call output buffers are configured on the run operation.
Defaults use CPU execution, preserve each output's backend storage location, and
use ordinary ONNX execution with internally managed CUDA streams when needed.

Placement and execution options
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 23 29 48

   * - Setting
     - Python / C++ / Rust
     - Behavior and optimization impact
   * - Execution device
     - ``device`` / ``device`` plus ``device_id`` / ``device``
     - CPU by default. CUDA moves model execution and private state to a GPU;
       supply inputs on that device. Weigh compute savings against transfer costs.
   * - Default output location
     - ``output_device`` / ``output_location`` / ``output_placement.default_location``
     - Unset by default, preserving backend storage without an output device
       transfer. CUDA outputs stay on CUDA; CPU fallback and constant outputs
       can remain on CPU. An explicit location requests a transfer. Pinned host
       placement can help subsequent transfers, with CUDA and page-locked
       allocation costs.
   * - Per-output location
     - ``output_overrides`` / ``output_overrides`` / ``output_placement.overrides``
     - Empty by default. Keys are qualified public output names; each value
       overrides the default location. An unset value preserves backend placement
       for that output even when the default requests a transfer. Unknown names
       fail session construction. Caller-supplied buffers choose their own locations.
   * - Node options
     - ``node_options`` in all languages
     - Empty by default. Maps node names to string options interpreted by each
       node's backend; see :ref:`runtime-node-options`. ONNX nodes can select
       TensorRT, load prebuilt engines, and enable CUDA graph replay.
   * - Caller-owned CUDA streams
     - Not exposed / not exposed / ``cuda_streams``
     - Empty by default. Rust maps CUDA device IDs to borrowed stream handles.
       Use to integrate execution and transfers with application stream ordering.
       Missing device entries use internal streams.

Use ``None`` in Python, ``std::nullopt`` in C++, or ``None`` in Rust to leave the
default or a per-output location unset. Rust wraps explicit locations in
``Some(...)``. Keeping backend placement avoids a device transfer; copies for
independent ownership and feedback isolation can still be needed.

This example runs ``control`` on GPU 0 and returns its six joint targets to the
CPU. Inputs must be CUDA tensors on GPU 0; use CPU options for the NumPy and
CPU-buffer examples in :doc:`getting_started`.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         options = lrt.SessionOptions(
             device="cuda:0",
             output_device="cuda:0",
             output_overrides={"control/joint_targets": "cpu"},
         )

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         lrt::SessionOptions options;
         options.device = lrt::ExecutionDevice::kCuda;
         options.device_id = 0;
         options.output_location = lrt::MemoryLocation{lrt::MemoryType::kCuda, 0};
         options.output_overrides["control/joint_targets"] =
             lrt::MemoryLocation{lrt::MemoryType::kCpu, 0};

   .. tab-item:: Rust
      :sync: rust

      .. code-block:: rust

         use lrt::{ExecutionDevice, MemoryLocation, OutputPlacement, SessionOptions};
         use std::collections::HashMap;

         let options = SessionOptions {
             device: ExecutionDevice::Cuda { device_id: 0 },
             output_placement: OutputPlacement {
                 default_location: Some(MemoryLocation::Cuda { device_id: 0 }),
                 overrides: HashMap::from([
                     ("control/joint_targets".into(), Some(MemoryLocation::Cpu)),
                 ]),
             },
             ..Default::default()
         };


.. _runtime-node-options:

Node options
------------

Set ``SessionOptions.node_options`` to a map from node names to string-valued
options interpreted by each node's backend; nodes without an entry use their
backend's defaults. Keys are ``name`` or ``namespace.name``. Each backend defines
its keys and namespaces: a namespace can hold settings that the runtime
interprets or options forwarded to the backend's own library. Unknown nodes,
keys, and namespaces are rejected when the session opens. ONNX is currently the
only backend that accepts options.

ONNX node options require CUDA execution:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Behavior
   * - ``execution_provider``
     - ``cuda`` (default) or ``tensorrt``. TensorRT falls back to the CUDA
       provider for operations it does not run.
   * - ``engine_path``
     - Loads a :ref:`prebuilt TensorRT engine <runtime-prebuilt-tensorrt-engines>`;
       requires ``tensorrt``.
   * - ``cuda_graph_replay``
     - Enables :ref:`CUDA graph replay <runtime-cuda-graph-replay>` on the
       active provider with ``1`` or ``true``; ``0`` or ``false`` disables it.
   * - ``tensorrt.<option>``
     - ONNX Runtime TensorRT provider option, forwarded unchanged; requires
       ``tensorrt``. The aliases ``precision_mode`` (``FP32``, ``FP16``,
       ``BF16``, or ``INT8``), ``max_workspace_size_bytes``,
       ``int8_calibration_table_name``, and ``int8_use_native_calibration_table``
       are translated to the provider's option names.
   * - ``cuda.<option>``
     - ONNX Runtime CUDA provider option, forwarded unchanged. It also applies
       when the CUDA provider is TensorRT's fallback.

Within the namespaces, ``device_id``, ``user_compute_stream``, and
``has_user_compute_stream`` are controlled by session options, and
``trt_cuda_graph_enable`` and ``enable_cuda_graph`` by ``cuda_graph_replay``;
setting them is rejected. Keys and values must not contain NUL.

Use the documentation matching your installed ONNX Runtime and TensorRT versions:

* `ONNX Runtime TensorRT provider reference
  <https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#execution-provider-options>`_
  and `CUDA provider reference
  <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html>`_
  for option names, accepted values, defaults, and compatibility requirements.
* `NVIDIA TensorRT documentation
  <https://docs.nvidia.com/deeplearning/tensorrt/latest/>`_
  for TensorRT installation, supported hardware, and performance guidance.
  Select the documentation for your installed version.

For mixed precision with TensorRT 11, prepare the ONNX model using ModelOpt
AutoCast before deployment, as described in the
`TensorRT mixed-precision migration guide
<https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-python-api-patterns.html#migrating-from-weak-typing-to-strong-typing>`_.
Precision is encoded in the model rather than selected by a session provider
option. Use a compatible ONNX Runtime and TensorRT combination.

These snippets enable TensorRT and engine caching for the getting-started node;
they do not change the model's precision:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         options = lrt.SessionOptions(
             device="cuda:0",
             node_options={"control": {
                 "execution_provider": "tensorrt",
                 "tensorrt.trt_engine_cache_enable": "1",
                 "tensorrt.trt_engine_cache_path": "/var/cache/my-app/control",
             }},
         )

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         lrt::SessionOptions options;
         options.device = lrt::ExecutionDevice::kCuda;
         options.device_id = 0;
         options.node_options["control"] = {
             {"execution_provider", "tensorrt"},
             {"tensorrt.trt_engine_cache_enable", "1"},
             {"tensorrt.trt_engine_cache_path", "/var/cache/my-app/control"},
         };

   .. tab-item:: Rust
      :sync: rust

      .. code-block:: rust

         let options = lrt::SessionOptions {
             device: lrt::ExecutionDevice::Cuda { device_id: 0 },
             node_options: std::collections::HashMap::from([
                 ("control".into(), std::collections::HashMap::from([
                     ("execution_provider".into(), "tensorrt".into()),
                     ("tensorrt.trt_engine_cache_enable".into(), "1".into()),
                     ("tensorrt.trt_engine_cache_path".into(), "/var/cache/my-app/control".into()),
                 ])),
             ]),
             ..Default::default()
         };


.. _runtime-prebuilt-tensorrt-engines:

Prebuilt TensorRT engines
-------------------------

Add ``engine_path`` to a TensorRT node's options to load a serialized TensorRT
engine, such as one written by ``trtexec --saveEngine``. ONNX Runtime
loads that engine when the session opens instead of building one from the
node's ONNX model, which is not read. The node's other options still apply,
such as ``cuda_graph_replay``; options that configure engine building have no
effect.

.. code-block:: python

   options = lrt.SessionOptions(
       device="cuda:0",
       node_options={"control": {
           "execution_provider": "tensorrt",
           "engine_path": "/opt/my-app/control.engine",
       }},
   )

Build the engine from the node's ONNX model on a compatible GPU, with the
TensorRT version that ONNX Runtime loads; otherwise opening the session fails.
Keep the model's input and output names and dtypes, for example by not changing
I/O formats with ``trtexec --inputIOFormats``. ONNX Runtime binds engine tensors
by name without checking them when loading, so a mismatched engine fails during
execution or can produce incorrect results. Dynamic dimensions must fall within
the engine's optimization profile. The engine is read once when the session
opens.

.. _runtime-cuda-graph-replay:

ONNX CUDA graph replay
----------------------

CUDA graph replay reduces repeated launch work for selected ONNX nodes. Enable
it with the node option ``cuda_graph_replay``; it uses the CUDA provider, or
TensorRT when ``execution_provider`` is ``tensorrt``. ``1`` or ``true`` enables
replay and ``0`` or ``false`` disables it; other values are rejected. Replay is
off unless explicitly requested.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         options = lrt.SessionOptions(
             device="cuda:0",
             output_device="cuda:0",
             node_options={"control": {"cuda_graph_replay": "1"}},
         )

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         lrt::SessionOptions options;
         options.device = lrt::ExecutionDevice::kCuda;
         options.device_id = 0;
         options.output_location = lrt::MemoryLocation{lrt::MemoryType::kCuda, 0};
         options.node_options["control"]["cuda_graph_replay"] = "1";

   .. tab-item:: Rust
      :sync: rust

      .. code-block:: rust

         let options = lrt::SessionOptions {
             device: lrt::ExecutionDevice::Cuda { device_id: 0 },
             output_placement: lrt::OutputPlacement {
                 default_location: Some(lrt::MemoryLocation::Cuda { device_id: 0 }),
                 ..Default::default()
             },
             node_options: std::collections::HashMap::from([(
                 "control".into(),
                 std::collections::HashMap::from([("cuda_graph_replay".into(), "1".into())]),
             )]),
             ..Default::default()
         };

The configuration above selects replay; executing it requires the following
buffer and model contracts. Ordinary calls retain their existing flexibility
when replay is disabled.

* **Fixed bindings:** Use fixed, nonempty tensor shapes and 256-byte-aligned CUDA
  buffers. Keep bound input and output allocations alive at the same addresses
  until close. Values may change in place, but host-side shape/control values
  must remain fixed. Address changes are rejected before execution; the runtime
  adds no public I/O staging copies to stabilize them.
* **Output buffers:** Supply every public output of an enabled node using Python
  ``run(inputs, outputs=...)``, C++ ``Run(inputs, &result, outputs)``, or Rust
  ``run_into``. Omitted outputs from other nodes retain normal allocation rules.
  Private intermediate and fixed-shape feedback bindings stay in their prepared
  buffers; feedback retains its existing copies.
* **Capture:** ONNX Runtime captures on the first call. Perform initial captures
  serially, with no concurrent CUDA work on other host threads. All operations
  must run on the selected GPU provider; fallback regions and control flow are
  unsupported. Disable replay for constant or input-forwarding outputs whose
  destination writes ONNX Runtime may skip.
* **Session lifecycle:** ``new_session`` is unavailable when replay is enabled;
  open an independent session instead. ``reset`` restores feedback without
  discarding capture or relaxing the binding requirements. After a replay
  backend error, close and reopen the session before retrying.

Caller-owned CUDA streams (Rust)
--------------------------------

Construct a borrowed handle with
``unsafe { lrt::CudaStream::from_raw(raw_stream) }`` from a ``cudaStream_t`` or
``CUstream``, then put it in ``SessionOptions.cuda_streams`` under its device ID.
Keep the stream and its primary context alive until the session and every peer
are closed or destroyed. Initialization validates stream/device compatibility.

Null and ``cudaStreamLegacy`` select the legacy default stream.
``cudaStreamPerThread`` is unsupported because sessions may move between threads.
Inputs must already be ready and runs complete before return. Peers inherit
supplied streams and their ordering, which can limit overlap; backends may use
auxiliary streams with synchronization. Python and C++ currently manage streams
internally.
