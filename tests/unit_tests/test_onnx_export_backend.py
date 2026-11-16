#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper

from leapp.backends.onnx_export_backend import ONNXDynamoExportBackend


def test_compile_materializes_once_and_saves_repaired_model():
    """Avoid repeated weight serialization while preserving pruning and repairs."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Slice", ["x", "zero", "one"], ["sliced"]),
                helper.make_node("Gather", ["x", "zero"], ["picked"], axis=0),
            ],
            "shared_scalar",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])],
            [
                helper.make_tensor_value_info("sliced", TensorProto.FLOAT, [1]),
                helper.make_tensor_value_info("picked", TensorProto.FLOAT, []),
            ],
            [
                numpy_helper.from_array(np.array(0, dtype=np.int64), name="zero"),
                numpy_helper.from_array(np.array(1, dtype=np.int64), name="one"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )
    accesses = []

    class Program:
        @property
        def model_proto(self):
            # Real ONNXProgram rebuilds a proto on every access.
            accesses.append(None)
            return onnx.ModelProto.FromString(model.SerializeToString())

    context = SimpleNamespace(
        name="shared_scalar",
        inputs=[
            SimpleNamespace(name=name, value=torch.ones(3)) for name in ("x", "unused")
        ],
        outputs=[SimpleNamespace(name=name) for name in ("sliced", "picked")],
        graph=torch.fx.Graph(),
        trimmed_inputs=set(),
    )
    backend = ONNXDynamoExportBackend(context)
    with (
        mock.patch("torch.onnx.export", return_value=Program()),
        mock.patch("leapp.backends.onnx_export_backend.patch_onnxscript_matcher"),
    ):
        backend.compile(torch.nn.Identity())

    # Count is the performance regression contract: each read copies all weights.
    assert len(accesses) == 1
    assert [item.name for item in context.inputs] == ["x"]
    assert context.trimmed_inputs == {"unused"}
    path = Path(backend.compiled_model._source_dir) / "shared_scalar.onnx"
    saved = onnx.load(path)
    onnx.checker.check_model(saved, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    for values in ([2.0, 3.0, 4.0], [-5.0, 0.0, 7.0]):
        x = np.array(values, dtype=np.float32)
        sliced, picked = session.run(None, {"x": x})
        np.testing.assert_array_equal(sliced, x[:1])
        np.testing.assert_array_equal(picked, x[0])
