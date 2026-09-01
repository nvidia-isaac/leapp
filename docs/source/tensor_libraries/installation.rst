============
Installation
============

A default ``pip install leapp`` does not install
``warp-lang`` support libraries for leapp. Tracing and exporting Warp APIC currently requires:

* Linux
* a CUDA-capable GPU
* ``warp-lang>=1.16.0``
* ``nvcc`` from the CUDA Toolkit, matching the installed CUDA version

Building the native runners additionally requires a C++17 compiler, CMake
3.18 or newer, and network access so CMake can download matching ONNX
Runtime C API headers.

Python packages (Export Environment Only)
=========================================

Install the extra that matches the CUDA toolkit on the machine:

.. code-block:: bash

   pip install "leapp[warp-cu12]"   # CUDA 12
   pip install "leapp[warp-cu13]"   # CUDA 13

These extras also install ``onnxruntime-gpu`` with the matching CUDA and
cuDNN runtime libraries, including the CUDA execution provider used to
validate exported ONNX models. ``warp-cu13`` needs Python 3.11 or newer
for that GPU package.

Do not install ``cupti-python`` independently. Its major version must match
the CUDA toolkit's ``libcupti`` major version, or CUPTI initialization
fails.

From a source checkout:

.. code-block:: bash

   uv sync --extra test --extra warp-cu12   # CUDA 12
   uv sync --extra test --extra warp-cu13   # CUDA 13

Native runners
==============

Exported ONNX and PT2 models that contain a Warp runner need LEAPP's native
custom-operator libraries. After the Python extra is installed, build both
adapters in the same environment that will run LEAPP:

.. code-block:: bash

   leapp-build-warp-runtime
   leapp-build-warp-runtime --status

The build command does not install dependencies. It checks for the Warp
library and APIC header, the ONNX Runtime shared library, PyTorch CMake
resources, CMake, and ``nvcc``, then fails with the missing item
if any of those are absent.

Libraries are written to a user cache. LEAPP discovers a valid cached
build automatically:

* Linux: ``${XDG_CACHE_HOME:-~/.cache}/leapp/warp-runtime/build``
* Windows: ``%LOCALAPPDATA%\leapp\warp-runtime\build``

Rebuild after changing PyTorch, Warp, ONNX Runtime, or CUDA, and restart
Python processes that already loaded either library. The cache is not
versioned per environment, so the most recent build is the one every
environment without an override discovers.

Overrides
=========

Environment variables can point at explicit library paths:

.. code-block:: bash

   export LEAPP_WARP_ONNX_CUSTOM_OP_LIBRARY="/path/to/libleapp_wrp_onnx_custom_op.so"
   export LEAPP_WARP_PT2_CUSTOM_OP_LIBRARY="/path/to/libleapp_wrp_torch_custom_op.so"

An invalid override is an error and does not fall back to the cache.
Windows uses the corresponding ``.dll`` names. A Windows build also needs
a Warp import library; the ``warp-lang`` wheel does not currently ship
``warp.lib``, so provide one through ``LEAPP_WARP_IMPORT_LIBRARY``.
