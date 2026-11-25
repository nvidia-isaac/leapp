# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Description validation rejects invalid inputs before loading model artifacts."""

import pytest
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
