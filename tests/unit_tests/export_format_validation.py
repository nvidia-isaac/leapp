#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Shared helpers for validating FX graphs across export formats."""

from __future__ import annotations

import pathlib
import tempfile
from typing import Callable, Sequence

import numpy as np
import onnxruntime as ort
import torch


def _as_input_tuple(inputs):
    if isinstance(inputs, tuple):
        return inputs
    return (inputs,)


def _clone_for_export(inputs: Sequence) -> tuple:
    prepared = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            prepared.append(value.clone())
        else:
            prepared.append(value)
    return tuple(prepared)


def verify_torchscript(
    testcase,
    graph_module,
    inputs,
    expected,
    *,
    test_name: str = "test",
    atol: float = 1e-5,
) -> None:
    inputs = _as_input_tuple(inputs)
    try:
        scripted = torch.jit.script(graph_module)
        actual = scripted(*inputs)
        testcase.assertTrue(
            torch.allclose(actual, expected, atol=atol),
            f"{test_name}: TorchScript output doesn't match expected",
        )
    except Exception as exc:
        testcase.fail(f"{test_name}: TorchScript export/execution failed: {exc}")


def verify_onnx(
    testcase,
    graph_module,
    inputs,
    expected,
    *,
    test_name: str = "test",
    atol: float = 1e-5,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    onnx_opset: int = 17,
) -> None:
    inputs = _as_input_tuple(inputs)
    if input_names is None:
        input_names = [f"input_{i}" for i in range(len(inputs))]
    if output_names is None:
        output_names = ["output"]

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = pathlib.Path(tmpdir) / f"{test_name}.onnx"
            torch.onnx.export(
                graph_module,
                inputs,
                onnx_path,
                dynamo=False,
                export_params=True,
                opset_version=onnx_opset,
                input_names=list(input_names),
                output_names=list(output_names),
            )
            session = ort.InferenceSession(str(onnx_path))
            onnx_inputs = {
                name: inp.numpy()
                for name, inp in zip(input_names, inputs)
            }
            output_onnx = session.run(None, onnx_inputs)[0]
            testcase.assertTrue(
                np.allclose(output_onnx, expected.detach().numpy(), atol=atol),
                f"{test_name}: ONNX output doesn't match expected",
            )
    except Exception as exc:
        testcase.fail(f"{test_name}: ONNX export/execution failed: {exc}")


def verify_onnx_dynamo(
    testcase,
    graph_module,
    inputs,
    expected,
    *,
    test_name: str = "test",
    atol: float = 1e-5,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    onnx_opset: int | None = None,
) -> None:
    inputs = _as_input_tuple(inputs)
    if input_names is None:
        input_names = [f"input_{i}" for i in range(len(inputs))]
    if output_names is None:
        output_names = ["output"]

    export_kwargs = {}
    if onnx_opset is not None:
        export_kwargs["opset_version"] = onnx_opset

    try:
        onnx_program = torch.onnx.export(
            graph_module,
            _clone_for_export(inputs),
            None,
            dynamo=True,
            export_params=True,
            input_names=list(input_names),
            output_names=list(output_names),
            **export_kwargs,
        )
        actual = onnx_program(*inputs)[0]
        comparable_expected = _align_expected_dtype(actual, expected)
        testcase.assertTrue(
            torch.allclose(actual, comparable_expected, atol=atol),
            f"{test_name}: ONNX dynamo output doesn't match expected",
        )
    except Exception as exc:
        testcase.fail(f"{test_name}: ONNX dynamo export/execution failed: {exc}")


def _align_expected_dtype(actual, expected):
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        if actual.dtype != expected.dtype:
            return expected.to(actual.dtype)
    return expected


def verify_exported_program(
    testcase,
    graph_module,
    inputs,
    expected,
    *,
    test_name: str = "test",
    atol: float = 1e-5,
) -> None:
    inputs = _as_input_tuple(inputs)
    try:
        exported = torch.export.export(graph_module, _clone_for_export(inputs))
        module = exported.module()
        actual = module(*inputs)
        comparable_expected = _align_expected_dtype(actual, expected)
        testcase.assertTrue(
            torch.allclose(actual, comparable_expected, atol=atol),
            f"{test_name}: ExportedProgram output doesn't match expected",
        )
    except Exception as exc:
        testcase.fail(
            f"{test_name}: ExportedProgram export/execution failed: {exc}")


def verify_exported_program_on_random_inputs(
    testcase,
    graph_module,
    input_shape: Sequence[int],
    func: Callable,
    *,
    test_name: str = "test",
    num_inputs: int = 5,
    atol: float = 1e-5,
    func_kwargs: dict | None = None,
) -> None:
    func_kwargs = func_kwargs or {}
    try:
        sample_input = torch.randn(*input_shape)
        exported = torch.export.export(graph_module, (sample_input,))
        module = exported.module()
        for index in range(num_inputs):
            test_input = torch.randn(*input_shape)
            expected = func(test_input, **func_kwargs)
            actual = module(test_input)
            testcase.assertTrue(
                torch.allclose(actual, expected, atol=atol),
                f"{test_name}: ExportedProgram output mismatch on input "
                f"{index + 1}/{num_inputs}",
            )
    except Exception as exc:
        testcase.fail(
            f"{test_name}: ExportedProgram export/execution failed: {exc}")
