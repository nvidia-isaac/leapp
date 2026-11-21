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
   * - ONNX TensorRT acceleration
     - ``onnx_tensorrt`` in all languages
     - Empty by default. Maps ONNX node names to string-valued provider options.
       An entry enables TensorRT; an empty parameter map uses SDK defaults.
       Nodes absent from the map retain ordinary execution. Requires CUDA.
   * - ONNX CUDA graph replay
     - ``cuda_graph_replay`` in all languages
     - Empty by default. Select ONNX node names for CUDA graph replay. Uses the
       CUDA provider unless the node is also listed in ``onnx_tensorrt``. Requires
       fixed CUDA bindings and caller-owned public outputs; see
       :ref:`runtime-cuda-graph-replay`.
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


ONNX TensorRT options
---------------------

Set ``SessionOptions.onnx_tensorrt`` to a map from ONNX node names to
string-valued provider options. CUDA execution is required. Each entry enables
TensorRT for that node; an empty option map uses provider defaults. Unknown
nodes and non-ONNX models are rejected.

Use the documentation matching your installed ONNX Runtime and TensorRT versions:

* `ONNX Runtime TensorRT provider reference
  <https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#execution-provider-options>`_
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
             onnx_tensorrt={"control": {
                 "trt_engine_cache_enable": "1",
                 "trt_engine_cache_path": "/var/cache/my-app/control",
             }},
         )

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         lrt::SessionOptions options;
         options.device = lrt::ExecutionDevice::kCuda;
         options.device_id = 0;
         options.onnx_tensorrt["control"] = {
             {"trt_engine_cache_enable", "1"},
             {"trt_engine_cache_path", "/var/cache/my-app/control"},
         };

   .. tab-item:: Rust
      :sync: rust

      .. code-block:: rust

         let options = lrt::SessionOptions {
             device: lrt::ExecutionDevice::Cuda { device_id: 0 },
             onnx_tensorrt: std::collections::HashMap::from([
                 ("control".into(), std::collections::HashMap::from([
                     ("trt_engine_cache_enable".into(), "1".into()),
                     ("trt_engine_cache_path".into(), "/var/cache/my-app/control".into()),
                 ])),
             ]),
             ..Default::default()
         };


``device_id``, ``user_compute_stream``, and ``has_user_compute_stream`` are
controlled by session options and cannot be set in the provider map. Keys and
values must not contain NUL. ``trt_cuda_graph_enable=1`` (or ``true``) also enables
replay for a TensorRT-configured node; it has the same buffer and lifecycle
requirements as ``cuda_graph_replay``. ``0`` and ``false`` disable that provider
option; other values are rejected. Replay is off unless explicitly requested.

.. _runtime-cuda-graph-replay:

ONNX CUDA graph replay
----------------------

CUDA graph replay reduces repeated launch work for selected ONNX nodes. Enable
it with ``cuda_graph_replay`` using node names, not qualified port names. It
uses the CUDA provider by default; nodes also listed in ``onnx_tensorrt`` use
TensorRT instead. An unknown node, a non-ONNX model, or CPU execution is rejected.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

         options = lrt.SessionOptions(
             device="cuda:0",
             output_device="cuda:0",
             cuda_graph_replay={"control"},
         )

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

         lrt::SessionOptions options;
         options.device = lrt::ExecutionDevice::kCuda;
         options.device_id = 0;
         options.output_location = lrt::MemoryLocation{lrt::MemoryType::kCuda, 0};
         options.cuda_graph_replay = {"control"};

   .. tab-item:: Rust
      :sync: rust

      .. code-block:: rust

         let options = lrt::SessionOptions {
             device: lrt::ExecutionDevice::Cuda { device_id: 0 },
             output_placement: lrt::OutputPlacement {
                 default_location: Some(lrt::MemoryLocation::Cuda { device_id: 0 }),
                 ..Default::default()
             },
             cuda_graph_replay: std::collections::HashSet::from(["control".into()]),
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
