Installation
============

The runtime targets Linux. Install the ``leapp-rt`` core and the model libraries
needed by your bundle. Python and Rust import it as ``lrt``; C++ uses the
``lrt`` namespace. Bundle inspection needs no model libraries. Opening a session
loads the selected libraries without downloading anything.

.. note::

   The first runtime release provides the core crate, Python source archive,
   and wheels. Registry publication, public Pixi channels, and optional backend
   wheels are separate release work. Commands below that use package registries
   apply once those artifacts are available. Until then, use matching local
   release artifacts, backend wheelhouses, and Pixi channels. Values enclosed
   in ``<...>`` are placeholders for your paths or release identifiers.

Language interface
------------------

Choose a language, then an installation method. Commands for the runtime's
packaging tools and CMake builds run from ``packages/leapp-rt`` in the LEAPP
checkout. Keep the core and backend packages on the same release and build.

.. tab-set::

   .. tab-item:: Python
      :sync: python

      Python 3.10+ is required. Release wheels target CPython 3.10–3.14 on Linux
      ``x86-64`` and ``aarch64`` with glibc 2.28+. Backend libraries can require a newer
      deployment OS. NumPy is a dependency; PyTorch and CuPy are optional
      DLPack producers and consumers.

      .. rubric:: pip

      Install the core for bundle inspection, or select the required backends
      with extras as described under **Model libraries** below:

      .. code-block:: bash

         python -m pip install leapp-rt

      Before registry publication, install the matching core wheel and backend
      wheels supplied by your build:

      .. code-block:: bash

         python -m pip install --find-links=<backend-wheelhouse> '<core-wheel-path>[onnx]'

      Import the runtime with ``import lrt``. The core wheel contains no model
      runtimes or NVIDIA libraries; optional backend wheels supply those libraries.

      .. rubric:: Pixi

      Use Pixi 0.84+ to generate a locked application environment with the core
      and the selected model libraries. This example selects CPU ONNX:

      .. code-block:: bash

         pixi run consumer --channel <backend-channel> --output <application-env> --backends onnx
         pixi install --manifest-path <application-env>/pixi.toml
         pixi run --manifest-path <application-env>/pixi.toml python -c 'import lrt'

      The generator installs the PyPI core by default. For a local channel
      containing development bindings, add ``--core-source channel`` to the
      generator command. An inspection-only environment needs no backend
      channel: use ``pixi run consumer --output <application-env>``.

   .. tab-item:: C++
      :sync: cpp

      Applications require C++20. Building the core with CMake requires Rust
      1.88+, CMake 3.22+, and a C++20 compiler. Bazel supplies its own Rust
      toolchain. An installed runtime library needs neither Rust nor model
      SDK headers at deployment.

      .. rubric:: Pixi

      Provision model libraries without installing Python bindings, then link
      your application to the C++ core using CMake:

      .. code-block:: bash

         pixi run consumer --channel <backend-channel> --output <application-env> --backends onnx --binding none
         pixi install --manifest-path <application-env>/pixi.toml

      Build the core from the source revision recorded in the channel's
      ``packages.json``. Run the application through ``pixi run --manifest-path
      <application-env>/pixi.toml`` to activate its libraries.

      .. rubric:: CMake

      Build and install the SDK:

      .. code-block:: bash

         cmake -S bindings/cxx -B target/cxx-build -DCMAKE_BUILD_TYPE=Release -DLEAPP_BUILD_TESTING=OFF
         cmake --build target/cxx-build -j
         cmake --install target/cxx-build --prefix <runtime-sdk-prefix>

      Set ``CMAKE_PREFIX_PATH`` to that prefix and link the installed SDK:

      .. code-block:: cmake

         find_package(leapp-rt CONFIG REQUIRED)
         target_link_libraries(my_application PRIVATE lrt::runtime)

      The shared library is ``libleapp_rt.so``. To embed a checkout instead,
      add its C++ binding directory to your application's build:

      .. code-block:: cmake

         add_subdirectory(<leapp-checkout>/packages/leapp-rt/bindings/cxx leapp-rt)
         target_link_libraries(my_application PRIVATE lrt::runtime)

      FetchContent users must retain the full repository layout and select
      ``packages/leapp-rt/bindings/cxx`` as ``SOURCE_SUBDIR``. Pin the source to
      the runtime release commit. Embedded builds leave tests off by default;
      use ``LEAPP_BUILD_TESTING=ON`` to enable them. Debug builds use Rust's ``dev``
      profile; other configurations use its release profile. Build on the target
      Linux architecture; cross-compilation is not configured.

      .. rubric:: Bazel

      The runtime is a standalone Bzlmod module named ``leapp_rt``. Build on
      Linux with a C++20 compiler; Bazel downloads the declared Rust toolchain,
      dependencies, and code generators. Building the core needs no Cargo,
      CMake, Pixi, Python, or model SDK installation.

      Until registry publication, add a pinned source dependency to your
      application's ``MODULE.bazel``. Choose a full commit containing Bazel
      support and retain ``strip_prefix`` to select the runtime module:

      .. code-block:: python

         bazel_dep(name="leapp_rt", version="0.1.0")
         bazel_dep(name="rules_cc", version="0.2.17")
         git_override(
             module_name="leapp_rt",
             remote="https://github.com/nvidia-isaac/leapp.git",
             commit="<full-commit-sha>",
             strip_prefix="packages/leapp-rt",
         )

      For a local checkout, replace ``git_override`` with
      ``local_path_override(module_name="leapp_rt",
      path="<leapp-checkout>/packages/leapp-rt")`` using an absolute path.
      No dependency setup macros or application toolchain registration are
      required. Link ``@leapp_rt//:runtime`` and include ``<leapp/runtime.h>``;
      :doc:`getting_started` provides a complete application build definition.
      Rust applications can depend on ``@leapp_rt//:lrt`` using ``rules_rust``.

      Bazel supplies ``libleapp_rt.so`` when running with ``bazel run``. For
      deployment outside Bazel, ship that shared library and make it available
      to the dynamic loader. Model libraries remain separate: use the
      **Model libraries** instructions below, including Pixi or manual SDK
      installation. Optional backend plugin builds use the existing workflows
      in :doc:`dependencies`.

   .. tab-item:: Rust
      :sync: rust

      Rust 1.88+ is required. Building the core needs no model-runtime or CUDA SDK.

      .. rubric:: Pixi

      Provision the model libraries separately from the core Cargo dependency:

      .. code-block:: bash

         pixi run consumer --channel <backend-channel> --output <application-env> --backends onnx --binding none
         pixi install --manifest-path <application-env>/pixi.toml

      Build the core from the source revision recorded in ``packages.json``.
      Run your application through ``pixi run --manifest-path
      <application-env>/pixi.toml`` so backend discovery uses that environment.

      .. rubric:: Crate

      For a published release:

      .. code-block:: bash

         cargo add leapp-rt@0.1

      The Cargo package is ``leapp-rt`` and its library import is ``lrt``.
      It currently pins the Rust ``ort`` binding to ``=2.0.0-rc.13``. This is
      separate from the dynamically loaded ONNX Runtime library, which must
      support C API 24.

Model libraries
---------------

Install only what the bundle uses. Model libraries are separate from the core;
adding a supported backend does not rebuild it or require Cargo features.
Pixi is the default for locked deployments. Python applications can also use
pip extras. Deployment needs compatible shared libraries and, for GPU work,
a compatible NVIDIA driver, rather than the exporter environment or build tools.

.. tab-set::

   .. tab-item:: pip extras (Python)

      These extras select companion backend wheels when available in your
      package index or local wheelhouse:

      .. code-block:: bash

         python -m pip install 'leapp-rt[onnx]'
         python -m pip install 'leapp-rt[onnx-cuda]'
         python -m pip install 'leapp-rt[onnx,tensorrt]'
         python -m pip install 'leapp-rt[torchscript,nativert]'

      Choose the command for your bundle. ``onnx`` stays CPU-only even when GPU
      wheels are available. The GPU and Torch extras default to CUDA 13. Use
      ``onnx-cuda12``, ``tensorrt-cuda12``, ``torchscript-cuda12``, or
      ``nativert-cuda12`` for CUDA 12; corresponding ``cuda13`` extras also exist.
      Keep one CUDA family per environment.

      Backend wheels include the required SDK libraries. Torch backends do not
      require the Python ``torch`` package. Embedded Warp operators still need
      Pixi or manual adapter installation. Installing TensorRT enables its
      libraries; configure individual ONNX nodes using :doc:`session_options`.

      Backend wheels are discovered when a session opens and need neither Pixi
      activation nor ``LD_LIBRARY_PATH``. Explicit ``LEAPP_NATIVE_BACKENDS`` or
      ``ORT_DYLIB_PATH`` settings select the manual or Pixi installation instead.
      All SDK components must match the core version and come from one producer
      build; do not mix wheelhouses from separate builds.

   .. tab-item:: Pixi packages

      The consumer generator selects the required packages from a matching local
      channel. Add ``--cuda`` for ONNX CUDA execution, ``--onnx-tensorrt`` for
      TensorRT acceleration, or ``--warp onnx`` for embedded Warp operators.
      ONNX TensorRT and Warp both require ``--cuda``. Select multiple models with, for example,
      ``--backends onnx torchscript nativert``. Omit GPU options and use a CPU-only
      ONNX channel when CUDA is unnecessary.

      .. list-table::
         :header-rows: 1

         * - Package
           - Purpose
         * - ``leapp-backend-onnx``
           - ONNX Runtime
         * - ``leapp-backend-torchscript``
           - TorchScript plugin and dependencies
         * - ``leapp-backend-nativert``
           - NativeRT library and PT2 plugin
         * - ``leapp-onnx-cuda``
           - ONNX Runtime CUDA provider and dependencies
         * - ``leapp-onnx-tensorrt``
           - ONNX Runtime TensorRT provider and dependencies

      Shared libraries are installed once. Activation sets
      ``LEAPP_NATIVE_BACKENDS`` to
      ``$CONDA_PREFIX/lib/leapp/native-backends`` unless explicitly overridden.
      Packaged libraries use relative lookup paths and need no
      ``LD_LIBRARY_PATH``. Avoid inheriting SDK search paths when using an
      isolated application environment.

      Commit the generated ``pixi.toml`` and ``pixi.lock`` with your application.
      To add or remove a backend, use its exact release and build from that
      manifest:

      .. code-block:: bash

         pixi add --manifest-path <application-env>/pixi.toml 'leapp-backend-torchscript ==0.1.0 <build-from-manifest>'
         pixi remove --manifest-path <application-env>/pixi.toml leapp-backend-torchscript

   .. tab-item:: SDKs and CMake

      Use ``leapp-build-runtime-backends`` to build TorchScript and PT2 plugins
      without Pixi. Follow :doc:`dependencies` for the command and SDK setup, or
      for producing a local backend channel and wheelhouse. Plugin builds require
      CMake 3.24+ and C++20. SDK headers and sources are build dependencies;
      deployment needs the resulting libraries and their shared dependencies.

      Set the backend directory before launching the application:

      .. code-block:: bash

         export LEAPP_NATIVE_BACKENDS=<backend-install-prefix>

      The fallback is ``/opt/leapp-rt/backends``. Install libraries at these
      relative paths:

      .. list-table::
         :header-rows: 1

         * - Export
           - Library below the backend directory
           - Requirements
         * - ONNX
           - ``onnxruntime/libonnxruntime.so``
           - ONNX Runtime supporting C API 24; CUDA provider and cuDNN for GPU
         * - TorchScript (``jit``, ``jit-script``, ``jit-trace``)
           - ``torchscript/libleapp_torchscript.so``
           - CUDA-enabled LibTorch, including for CPU execution
         * - ExportedProgram (``pt2``)
           - ``nativert/libleapp_nativert_backend.so``
           - Matching LibTorch and NativeRT

Backend packages must match the deployment architecture and glibc baseline.
Jetson SDK packages also record a Jetson Linux release, which must match the
host. ARM server and Jetson packages are not interchangeable solely because
both use ``aarch64``. Follow the consumer generator's compatibility checks; copying
a manifest and running ``pixi install`` alone does not validate Jetson support.

.. toctree::
   :maxdepth: 1

   dependencies
