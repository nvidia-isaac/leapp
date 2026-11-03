============
Optimization
============

TensorRT op-catalog checks, GraphSurgeon optimizations, and DAG merge live in
the in-tree ``isaac_deploy_trt`` package (also the ``graph-surgery`` extra),
not in the LEAPP tracing CLI. After ``compile_graph()``, optimize exported
ONNX in place with ``leapp.optimize_graph(trt_compatible=True)``. That
updates YAML checksums for files that changed. ``compile_graph()`` does not
replace operators with TensorRT-compatible ones.

With no ``--tensorrt`` flag, ``check``, ``optimize``, ``optimize-dir``, and
``merge-pipeline --optimize`` use the minimal catalog,
``tensorrt_catalogs/minimal/tensorrt_layers.json``. An ONNX operator or
native layer counts as supported only when every checked-in TensorRT release
and platform marks it supported. ``leapp.optimize_graph(trt_compatible=True)``
uses that catalog for unsupported-operator optimizations. Pass
``trt_version``, for example
``leapp.optimize_graph(trt_compatible=True, trt_version="10.14")``, to use
one release's catalog. The result of that call is the compatibility record:
``{"trt_compatible": ["10.14"]}`` when every model is supported, or no
``trt_compatible`` key when any model is not. UINT8 intermediate
Casts, Resize ``antialias``, and rank-9 Reshape edits always run.

When ``optimize_graph()`` changes an ONNX file, it reruns that file with the
captured trace inputs and compares its outputs to the captured outputs.
Validation uses ``rtol=1e-3`` and ``atol=1e-5`` by default, matching
``compile_graph()``. Pass ``rtol``, ``atol``, or ``strict=False`` to adjust
the check, or ``validate=False`` to skip it. Captured trace values are
available only when optimization runs in the same Python process as
``compile_graph()``.

``--tensorrt`` selects one release, for example ``10.16``. ``--platform``
(``x86``, ``orin``, ``thor``, or ``dgx``) applies only together with
``--tensorrt``. ``--tensorrt 11.2`` with no platform uses the built-in 11.2
parser table. ``--catalog`` replaces those flags with one JSON file.

.. code-block:: bash

   pip install -e ".[graph-surgery]"
   isaac_deploy_trt check --onnx path/to/model.onnx --strict
   isaac_deploy_trt check --onnx path/to/model.onnx --tensorrt 10.16 --platform orin --strict
   isaac_deploy_trt check --onnx-dir path/to/export_dir --build --strict
   isaac_deploy_trt optimize-dir path/to/export_dir -o path/to/export_dir_trt
   isaac_deploy_trt optimize-dir path/to/export_dir -o path/to/export_dir_trt --tensorrt 10.16 --platform orin
   isaac_deploy_trt merge-pipeline path/to/export.yaml -o pipeline.onnx --optimize

``--build`` also runs TensorRT parse/build (requires ``tensorrt``).
``--strict`` exits non-zero if catalog or TensorRT checks fail.

``compile_graph()`` does not run a TensorRT catalog check and does not write
TensorRT fields into model parameters. ``optimize_graph()`` removes any
leftover ``optimization`` or ``tensorrt_version`` parameter. Operators that
fail the catalog are written to ``log.txt`` as
``tensorrt_blockers <model.onnx>: ...``.
