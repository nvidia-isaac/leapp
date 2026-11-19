Model Dependencies
==================

Use the packaged environments or Python extras on :doc:`installation` when
available. This page describes building backend packages from existing SDKs
using the runtime's build helpers. Commands run from ``packages/leapp-rt`` in a
matching LEAPP checkout.
Values in ``<...>`` are installation-specific paths.

.. warning::

   **TODO: Review dependency installation before release.** The Pixi consumer
   generator, backend producer, and Warp builder already automate environment
   creation and builds. Review how users obtain compatible SDKs and whether
   that remaining setup needs an installer. These helpers do not acquire SDKs.

Build ONNX, TorchScript, and PT2 backends
-----------------------------------------

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

``build-runtime`` is the shared build helper, implemented in
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

Build Warp adapters
-------------------

ONNX and PT2 models containing embedded Warp operators need the optional LEAPP
adapters, built against matching ONNX Runtime and LibTorch versions. Follow
:doc:`../tensor_libraries/installation` for exporter and build prerequisites.
``leapp-build-warp-runtime`` builds both adapters in the environment containing
those dependencies. Pass its output directory to the shared backend helper to
place the adapters beside the selected backends. For a build that includes Warp,
use this sequence instead of the build above:

.. code-block:: bash

   leapp-build-warp-runtime --build-dir <warp-build-dir> -- \
     -DLEAPP_WARP_ONNXRUNTIME_VERSION=<installed-onnxruntime-version> \
     -DCMAKE_INSTALL_LIBDIR=lib
   pixi run build-runtime --output <package-output> --glibc <deployment-glibc> \
     --backends onnx --cuda --warp <warp-build-dir> \
     --sdk-origin 'sdk-artifact@sha256:<verified-digest>'
   pixi run package-runtime --input <package-output>

For PT2 Warp models, select ``--backends nativert``; use ``--backends onnx nativert``
for both. The Warp builder does not install SDK dependencies, and ``--warp``
packages existing adapters rather than building them. ``ORT_INCLUDE_DIR`` can
select installed ONNX headers instead of downloading them; ``WARP_ROOT``,
``Torch_DIR``, and ``CUDAToolkit_ROOT`` select existing SDKs. Warp adapters use
Pixi packages or manual installation; the backend-wheel producer does not
package them.

Keep Warp export and runtime releases aligned and expose Warp's ``bin/`` directory
to the loader. Embedded Warp execution requires CUDA device 0, even for CPU model
placement. Warm each model serially before concurrent use. These adapters do not
support Warp operators in TorchScript artifacts. Invalid installed adapters fail
loading even for ordinary models using the same backend.
