# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Description validation rejects invalid inputs before loading model artifacts."""

import pytest
import torch
import yaml

from leapp import InferenceManager


@pytest.fixture
def description():
    return {
        "models": {
            "policy": {
                "inputs": [],
                "outputs": [],
                "parameters": {
                    "model_path": "unused.pt",
                    "md5sum": "unused",
                    "sha256sum": "unused",
                    "backend": "jit",
                },
            }
        },
        "pipeline": {},
        "system information": {},
    }


def test_missing_description_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="Leapp description file not found"):
        InferenceManager(str(tmp_path / "missing.yaml"))


def test_description_requires_yaml_extension(tmp_path):
    path = tmp_path / "graph.txt"
    path.write_text("")
    with pytest.raises(ValueError, match=r"must end with \.yaml"):
        InferenceManager(str(path))


@pytest.mark.parametrize("missing_key", ["models", "pipeline", "system information"])
def test_description_requires_top_level_keys(tmp_path, description, missing_key):
    description.pop(missing_key)
    path = tmp_path / "graph.yaml"
    path.write_text(yaml.safe_dump(description))
    with pytest.raises(
        ValueError,
        match="Leapp description file must contain models, pipeline, and system_info",
    ):
        InferenceManager(str(path))


@pytest.mark.parametrize(
    "missing_key", ["model_path", "md5sum", "sha256sum", "backend"]
)
def test_description_requires_model_parameters(tmp_path, description, missing_key):
    description["models"]["policy"]["parameters"].pop(missing_key)
    path = tmp_path / "graph.yaml"
    path.write_text(yaml.safe_dump(description))
    with pytest.raises(
        ValueError,
        match="Model description must contain model_path, md5sum, sha256sum, and backend",
    ):
        InferenceManager(str(path))


@pytest.mark.parametrize(
    "source, actual, expected",
    [
        pytest.param([100000], [100001], False, id="token-id-exact"),
        pytest.param([100000], [100000], True, id="token-id-equal"),
        pytest.param([True], [False], False, id="boolean-exact"),
        pytest.param([1.0], [1.0001], True, id="float-tolerance"),
    ],
)
def test_shared_validation_preserves_text_output_comparisons(source, actual, expected):
    """Text nodes compare token IDs exactly; float outputs keep the tolerance."""
    from leapp.leapp_graph.leapp_node import LeappNode

    node = LeappNode("text")
    node._validation_exact_discrete = True
    node.export_backend.compiled_model = lambda: torch.tensor(actual)
    mismatches = node._sample_mismatches(
        "sample", (), (torch.tensor(source),), rtol=1e-3, atol=1e-5
    )
    assert (not mismatches) is expected
