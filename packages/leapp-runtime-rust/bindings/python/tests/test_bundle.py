# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundle inspection needs only YAML and opaque artifacts, with no model SDK."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import leapp_runtime as lr
import pytest

YAML = (Path(__file__).parents[3] / "tests/fixtures/pipeline.yaml").read_text()


@pytest.fixture
def bundle_path(tmp_path):
    (tmp_path / "model.bin").write_bytes(b"abc")
    path = tmp_path / "pipeline.yaml"
    path.write_text(YAML)
    return path


def test_ordered_inspection_and_detached_values(bundle_path):
    bundle = lr.Bundle.open(bundle_path)
    assert bundle.yaml_path() == bundle_path
    assert bundle.model_names() == ["z_observation", "a_policy"]
    assert list(bundle.pipeline_inputs()) == ["z_observation/joint_pos"]
    assert list(bundle.pipeline_outputs()) == [
        "a_policy/next_state",
        "a_policy/actions",
        "z_observation/features",
    ]
    spec = bundle.output_info("a_policy", "actions")
    assert spec.dtype == "float32"
    assert spec.shape == [3]
    assert spec.tensor_type == "tensor"
    assert spec.metadata["temporal_period_ms"] == 100
    assert bundle.system_information()["leapp config version"] == "1.3"
    assert bundle.pipeline_configs()["frequency"] == 50
    parameters = bundle.model_parameters("z_observation")
    assert parameters["model_path"] == "model.bin"
    parameters["custom_options"]["flags"].clear()
    spec.metadata["temporal_period_ms"] = 999
    assert bundle.model_parameters("z_observation")["custom_options"]["flags"] == [
        "one",
        "two",
    ]
    assert (
        bundle.output_info("a_policy", "actions").metadata["temporal_period_ms"] == 100
    )
    assert bundle.input_info("absent", "port") is None
    assert bundle.model_metadata("absent") is None
    assert bundle.model_parameters("absent") is None
    with pytest.raises(AttributeError):
        spec.dtype = "int8"
    del bundle
    assert spec.shape == [3]


def test_model_io_returns_detached_internal_port_names(bundle_path):
    bundle = lr.Bundle.open(bundle_path)
    expected = (["features", "state"], ["actions", "next_state"])
    inputs, outputs = bundle.model_io(model_name="a_policy")
    assert (inputs, outputs) == expected
    assert bundle.model_io("absent") is None
    inputs.clear()
    outputs.append("extra")
    assert bundle.model_io("a_policy") == expected


def test_custom_backend_and_unresolved_input(bundle_path):
    bundle_path.write_text(
        YAML.replace("backend: pt2", "backend: custom").replace(
            "inputs: {z_observation: [joint_pos]}", "inputs: {absent: [port]}"
        )
    )
    bundle = lr.Bundle.open(bundle_path)
    assert bundle.model_parameters("a_policy")["backend"] == "custom"
    assert bundle.pipeline_inputs() == {"absent/port": None}


def test_lossless_metadata(bundle_path):
    bundle_path.write_text(
        YAML
        + """
sensor: {axes: [x, y], enabled: true}
sequence_keys: {[x, y]: coordinates}
colliding_keys: {true: boolean, 1: integer}
unsigned: 18446744073709551615
"""
    )
    metadata = lr.Bundle.open(bundle_path).metadata()
    assert metadata["sensor"] == {"axes": ["x", "y"], "enabled": True}
    assert isinstance(metadata["sequence_keys"], lr.MetadataMapping)
    assert metadata["sequence_keys"].entries == [(["x", "y"], "coordinates")]
    assert isinstance(metadata["colliding_keys"], lr.MetadataMapping)
    assert metadata["colliding_keys"].entries == [(True, "boolean"), (1, "integer")]
    assert metadata["unsigned"] == 18446744073709551615


@pytest.mark.parametrize(
    "failure,code,phase",
    [
        ("missing", lr.ErrorCode.NOT_FOUND, "open"),
        ("parse", lr.ErrorCode.INVALID_ARGUMENT, "parse"),
        ("artifact", lr.ErrorCode.INVALID_ARGUMENT, "artifact"),
    ],
)
def test_diagnostics(bundle_path, failure, code, phase):
    if failure == "missing":
        bundle_path.unlink()
    elif failure == "parse":
        bundle_path.write_text("models: [")
    else:
        (bundle_path.parent / "model.bin").write_bytes(b"wrong hash")
    with pytest.raises(lr.BundleError) as caught:
        lr.Bundle.open(bundle_path)
    assert caught.value.code is code
    assert caught.value.phase == phase
    assert caught.value.path == bundle_path
    assert caught.value.message


def test_snapshot_and_concurrent_reads(bundle_path):
    bundle = lr.Bundle.open(bundle_path)
    bundle_path.unlink()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda _: bundle.input_info("a_policy", "features").shape, range(24)
            )
        )
    assert results == [[3]] * 24
    assert bundle.model_names() == ["z_observation", "a_policy"]
