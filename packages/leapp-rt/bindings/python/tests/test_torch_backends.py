# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared TorchScript/NativeRT integration checks for either engine build."""

import os
from pathlib import Path

import lrt as lr
import pytest


@pytest.mark.parametrize("backend", ["torchscript", "pt2"])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_torch_dlpack_outputs_survive_reuse_and_session_close(backend, device):
    if device != "cpu" and os.environ.get("LEAPP_TEST_CUDA") != "1":
        pytest.skip("set LEAPP_TEST_CUDA=1 to exercise CUDA")
    variable = (
        "LEAPP_TORCHSCRIPT_FIXTURES"
        if backend == "torchscript"
        else "LEAPP_PT2_FIXTURES"
    )
    root = os.environ.get(variable)
    if root is None:
        pytest.skip(
            f"set {variable} after generating fixtures and installing the backend"
        )
    torch = pytest.importorskip("torch")
    filename = (
        "scripted.yaml"
        if backend == "torchscript"
        else f"weighted_{'cpu' if device == 'cpu' else 'cuda'}.yaml"
    )
    bundle = lr.Bundle.open(Path(root) / filename)
    values = torch.arange(1.0, 5.0, device=device)
    bias = torch.arange(1.0, 5.0, device=device) * 10
    retained = []
    with lr.Session.open(
        bundle,
        lr.SessionOptions(
            engine=os.environ.get("LEAPP_TEST_ENGINE", "triton"),
            device=device,
            output_device=device,
        ),
    ) as session:
        for offset in [0.0, 7.0, -2.0]:
            values.copy_(torch.arange(1.0, 5.0, device=device) + offset)
            if backend == "torchscript":
                inputs = {"model/INPUT__9": bias, "model/z_values": values}
                weighted_name, transposed_name = "model/z_result", "model/OUTPUT__7"
            else:
                inputs = {"model/values": values}
                weighted_name, transposed_name = "model/output_0", "model/output_1"
            expected = values * torch.arange(2.0, 6.0, device=device) + bias
            target = torch.empty_like(values) if offset == 7.0 else None
            if target is not None:
                with pytest.raises(lr.RuntimeError, match="overlap"):
                    session.run(
                        inputs,
                        outputs={
                            weighted_name: target,
                            transposed_name: target.reshape(2, 2),
                        },
                    )
            result = session.run(
                inputs, outputs={weighted_name: target} if target is not None else None
            )
            weighted = (
                target
                if target is not None
                else torch.from_dlpack(result[weighted_name])
            )
            if target is not None:
                assert weighted_name not in result.keys()
            transposed = torch.from_dlpack(result[transposed_name])
            assert weighted.device == values.device
            torch.testing.assert_close(transposed, values.reshape(2, 2).t())
            retained.append((weighted, expected.clone()))
    for actual, expected in retained:
        torch.testing.assert_close(actual, expected)
