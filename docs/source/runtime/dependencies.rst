Model Dependencies
==================

Use the packaged environments or Python extras on :doc:`installation` when
available. To build against your own SDKs, use the standalone commands below;
Pixi is optional. Commands run from ``packages/leapp-rt`` in a matching LEAPP
checkout unless specified otherwise.
Values in ``<...>`` are installation-specific paths.

Build plugins without Pixi
--------------------------

``leapp-build-runtime-backends`` builds and installs TorchScript and PT2 plugins.
It uses only Python 3.10+ and the standard library, CMake 3.24+, a C++20 compiler,
and the selected SDKs. The command is installed with the LEAPP exporter, alongside
``leapp-build-warp-runtime``. You can also run
``../../scripts/leapp-build-runtime-backends`` directly from this checkout
without installing or importing LEAPP.

Set ``TORCH_ROOT`` to a CUDA-enabled LibTorch SDK. PT2 additionally requires
PyTorch v2.12.1 sources with submodules in ``PYTORCH_SOURCE`` and LibTorch 2.12.1.
Use C++11 ABI ``1``. Set ``CUDAToolkit_ROOT`` or ``CUDA_ROOT`` for a nonstandard
CUDA Toolkit; both plugins need it even for CPU execution.

.. code-block:: bash

   leapp-build-runtime-backends --backends torchscript nativert \
     --install-prefix <backend-install-prefix>
   export LEAPP_NATIVE_BACKENDS=<backend-install-prefix>

Select just ``torchscript`` or ``nativert`` when you need one plugin. PT2 builds
its NativeRT dependency automatically; ``NATIVERT_LIBRARY`` can select an
existing matching library instead. The helper installs plugins and the NativeRT
library it builds; LibTorch, CUDA, and other SDK dependencies stay in their
existing locations and must be available to the dynamic loader.

Sources are discovered in the checkout. When running the installed command
elsewhere, pass ``--source-dir <leapp-checkout>/packages/leapp-rt`` from the same
revision as the deployed core. ``--build-dir`` selects the CMake build directory
and ``--jobs`` controls parallelism. The default build directory is
``<runtime-source>/target/standalone-backends``, with plugins installed into its
``backends/`` directory unless ``--install-prefix`` is supplied. Build directories
can be reused, but each invocation starts with a fresh CMake configuration so SDK
paths are rediscovered. The command does not install SDK dependencies.

ONNX needs no LEAPP plugin build. Point ``ORT_DYLIB_PATH`` at a compatible ONNX
Runtime shared library, or place it under
``$LEAPP_NATIVE_BACKENDS/onnxruntime/libonnxruntime.so``. GPU execution needs its
matching CUDA provider libraries; TensorRT acceleration also needs the ONNX
Runtime TensorRT provider and its dependencies.

Build backend packages with Pixi
--------------------------------

Build a local channel on the target Linux architecture in an environment with
the required SDKs. Set ``ORT_DYLIB_PATH`` for ONNX Runtime and ``TORCH_ROOT``
for TorchScript or PT2. PT2 also needs matching ``PYTORCH_SOURCE``. Set
``CUDA_ROOT`` to a compatible CUDA Toolkit for CUDA builds and for the Torch
plugins, including CPU execution. ONNX TensorRT additionally requires
``TENSORRT_ROOT``. The producer does not acquire these vendor SDKs.

.. code-block:: bash

   pixi install --locked
   pixi run build-runtime --output <package-output> --glibc <deployment-glibc> \
     --backends onnx \
     --sdk-origin 'sdk-artifact@sha256:<verified-digest>'
   pixi run package-runtime --input <package-output>

To create backend packages, use the shared ``build-runtime`` producer in
``packaging/build.py``. It runs the required configure, build, and install steps
into a staging directory. Choose the backend with these arguments:

.. list-table::
   :header-rows: 1

   * - Model
     - Build arguments
     - What the helper does
   * - ONNX Runtime
     - ``--backends onnx``
     - Records the supplied ONNX Runtime SDK for packaging; no ONNX plugin
       build is needed.
   * - TorchScript
     - ``--backends torchscript``
     - Builds and stages the plugin against ``TORCH_ROOT``.
   * - PT2
     - ``--backends nativert``
     - Builds matching NativeRT from ``PYTORCH_SOURCE``, then builds and stages
       the PT2 plugin.

Select multiple backends with ``--backends onnx torchscript nativert``. Add
``--cuda`` for ONNX CUDA providers, and ``--cuda --onnx-tensorrt`` for ONNX
TensorRT acceleration. PT2 requires PyTorch v2.12.1 sources with submodules and
LibTorch 2.12.1. Use LibTorch with C++11 ABI ``1`` for the shared build helper.

Choose a fresh ``<package-output>`` directory for each build; the helper rejects
an existing staging directory.

``package-runtime`` collects the SDK libraries and staged plugins into installable
Pixi backend packages. Packages and ``packages.json`` appear in
``<package-output>/channel``; use that directory with the consumer generator in
:doc:`installation`. Shared SDK libraries are packaged once, with relative
lookup paths and their licenses. The compiler, headers, exporter, and build
tools stay out of the consumer environment.

For Python backend wheels from the same build:

.. code-block:: bash

   pixi run package-wheels --input <package-output>

The wheelhouse is ``<package-output>/backend-wheels``. The producer normally
requires a clean matching ``runtime-vVERSION`` tag. Before publication,
``package-wheels --allow-unreleased`` permits a development build; install it
with a core wheel built from the same checkout. For local Pixi Python bindings,
add ``--bindings python`` to ``build-runtime`` and ``--core-source channel`` to
``consumer``. Public runtime releases contain the core; backend distribution
and hosting remain separate release work.

PT2 consumes raw ``torch.export.save`` archives with tensor-only inputs and
outputs in exported-program order. AOTInductor and CUDA Triton delegates are
unsupported. TorchScript accepts positional tensor arguments and a tensor,
flat tuple, or flat list of tensors; bundle port order determines argument order.

.. note::

   **Models with Warp operators:** Follow
   :doc:`Warp installation <../tensor_libraries/installation>` to build the
   optional ONNX/PT2 adapters with
   ``leapp-build-warp-runtime --build-dir <warp-build-dir>``. Add
   ``--cuda --warp <warp-build-dir>`` to ``build-runtime`` above, selecting
   ``onnx``, ``nativert``, or both. Packaging includes the existing adapters;
   backend wheels do not include them. See the
   :ref:`Warp runtime requirements <warp-runtime-deployment>` before deployment.
