Model Dependencies
==================

Use the packaged environments or Python extras on :doc:`installation` when
available. This page describes producing those packages and manually installing
SDKs. Commands run from ``packages/leapp-rt`` in a matching LEAPP checkout.
For manual installs, first set the backend directory in the **SDKs and CMake**
tab on :doc:`installation`. Values in ``<...>`` are installation-specific paths.

.. warning::

   **TODO: Review dependency installation before release.** The Pixi consumer
   generator and backend-wheel producer now automate packaged installations.
   Review the remaining SDK acquisition and manual setup steps, and decide
   whether a separate installer script should handle version selection and
   library placement. The manual recipes below remain provisional.

Build backend packages
----------------------

Build a local channel on the target Linux architecture in an environment with
the required SDKs. Set ``ORT_DYLIB_PATH`` for ONNX Runtime; Torch builds also
need ``TORCH_ROOT`` and matching ``PYTORCH_SOURCE``. CUDA builds need a compatible
CUDA toolchain. ONNX TensorRT additionally requires ``TENSORRT_ROOT``. These
inputs are supplied explicitly; the producer does not acquire vendor SDKs.

.. code-block:: bash

   pixi install --locked
   pixi run build-runtime --output <package-output> --glibc <deployment-glibc> \
     --backends onnx --cuda --onnx-tensorrt \
     --sdk-origin 'sdk-artifact@sha256:<verified-digest>'
   pixi run package-runtime --input <package-output>

Omit ``--cuda --onnx-tensorrt`` for CPU ONNX. Use
``--backends torchscript nativert`` to select the Torch plugins. Pass
``--warp <warp-adapter-directory>`` to package prebuilt Warp adapters from the
same SDK build. Packages and ``packages.json`` appear in
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

Manual SDK installation
-----------------------

The following recipes apply when building plugins and placing SDK libraries
yourself. They are unnecessary when a matching packaged backend is installed.

ONNX Runtime
~~~~~~~~~~~~

Install a compatible ONNX Runtime at
``$LEAPP_NATIVE_BACKENDS/onnxruntime/libonnxruntime.so``, or set
``ORT_DYLIB_PATH`` to its shared library. The ONNX adapter is included in the core; no separate
plugin build is needed. For GPU execution, place CUDA provider libraries beside
ONNX Runtime and expose their CUDA and cuDNN dependencies to the dynamic loader.
For TensorRT acceleration, also install ``libonnxruntime_providers_tensorrt.so``
beside ONNX Runtime, with matching TensorRT runtime, parser, plugin, and builder
resource libraries. See :doc:`session_options`.

TorchScript
~~~~~~~~~~~

Set ``TORCH_ROOT`` to a CUDA-enabled LibTorch SDK containing ``include/`` and
``lib/``. ``TORCH_CXX11_ABI`` defaults to ``1`` and must match the SDK.
``CUDAToolkit_ROOT`` selects a nonstandard CUDA installation.

.. code-block:: bash

   cmake -S src/engines/native/backends/torchscript -B target/torchscript \
     -DCMAKE_BUILD_TYPE=Release -DTORCH_ROOT="$TORCH_ROOT" \
     -DCMAKE_INSTALL_PREFIX="$LEAPP_NATIVE_BACKENDS"
   cmake --build target/torchscript -j4
   cmake --install target/torchscript

PT2
~~~

Use PyTorch v2.12.1 sources with submodules and LibTorch 2.12.1. Build the matching
NativeRT library, then the PT2 plugin. The LibTorch ABI and CUDA settings above
also apply.

.. code-block:: bash

   cmake -S src/engines/triton/backends/torch/nativert -B target/nativert-source \
     -DCMAKE_BUILD_TYPE=Release -DPYTORCH_SOURCE=<pytorch-v2.12.1-source> \
     -DTORCH_ROOT="$TORCH_ROOT"
   cmake --build target/nativert-source -j4
   cmake -S src/engines/native/backends/nativert -B target/pt2 \
     -DCMAKE_BUILD_TYPE=Release -DTORCH_ROOT="$TORCH_ROOT" \
     -DPYTORCH_SOURCE=<pytorch-v2.12.1-source> \
     -DNATIVERT_LIBRARY="$PWD/target/nativert-source/libleapp_nativert.so" \
     -DCMAKE_INSTALL_PREFIX="$LEAPP_NATIVE_BACKENDS"
   cmake --build target/pt2 -j4
   cmake --install target/pt2

PT2 consumes raw ``torch.export.save`` archives with tensor-only inputs and
outputs in exported-program order. AOTInductor and CUDA Triton delegates are
unsupported. TorchScript accepts positional tensor arguments and a tensor,
flat tuple, or flat list of tensors; bundle port order determines argument order.

Warp operators
~~~~~~~~~~~~~~

ONNX and PT2 models containing embedded Warp operators need the optional LEAPP
adapters, built against matching ONNX Runtime and LibTorch versions. Follow
:doc:`../tensor_libraries/installation` for exporter and build prerequisites.
Run the builder from the environment containing those dependencies:

.. code-block:: bash

   leapp-build-warp-runtime --build-dir <warp-build-dir> -- \
     -DLEAPP_WARP_ONNXRUNTIME_VERSION=<installed-onnxruntime-version> \
     -DCMAKE_INSTALL_LIBDIR=lib
   cmake --install <warp-build-dir> --prefix <warp-adapter-prefix>
   mkdir -p "$LEAPP_NATIVE_BACKENDS/onnxruntime" "$LEAPP_NATIVE_BACKENDS/nativert"
   ln -s <warp-adapter-prefix>/lib/libleapp_wrp_onnx_custom_op.so \
     "$LEAPP_NATIVE_BACKENDS/onnxruntime/libleapp_wrp_onnx_custom_op.so"
   ln -s <warp-adapter-prefix>/lib/libleapp_wrp_torch_custom_op.so \
     "$LEAPP_NATIVE_BACKENDS/nativert/libleapp_wrp_torch_custom_op.so"

Use absolute paths for the adapter prefix. ``ORT_INCLUDE_DIR`` can select
installed headers instead of downloading them; ``WARP_ROOT``, ``Torch_DIR``, and
``CUDAToolkit_ROOT`` select existing SDKs. The builder does not install dependencies.
Keep Warp export and runtime releases aligned and expose Warp's ``bin/`` directory
to the loader. Embedded Warp execution requires CUDA device 0, even for CPU model
placement. Warm each model serially before concurrent use. These adapters do not
support Warp operators in TorchScript artifacts. Invalid installed adapters fail
loading even for ordinary models using the same backend.
