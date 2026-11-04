#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Variable dimensions through export, serialization, and runtime inference."""

import os

import numpy as np
import pytest
import torch
import yaml

import leapp
from leapp import Axis, InferenceManager, TensorSemantics, Variable, annotate
from leapp.export_manager import ExportManager
from leapp.leapp import _MANAGER
from tests.functional_tests.base import LEAPPFunctionalTestBase


class TestDynamicShapeRoundTrip(LEAPPFunctionalTestBase):
    def test_dynamic_shapes_survive_export_and_reload(self):
        """Round-trip inferred axes through YAML, connected models, and runtime checks."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        x = annotate.input_tensors(
            "scale",
            leapp.TensorSemantics(
                "x",
                torch.arange(16, dtype=torch.float32).reshape(2, 8),
                axes=[
                    leapp.Axis(kind="element", names=["left", "right"]),
                    leapp.Axis(
                        variable=leapp.Variable("length", min=4, max=32, multiple_of=2)
                    ),
                ],
            ),
        )
        y = annotate.output_tensors("scale", {"y": x * 2}, export_with="pt2")
        y = annotate.input_tensors("pairs", {"y": y})
        annotate.output_tensors(
            "pairs", {"z": y.reshape(2, y.size(1) // 2, 2)}, export_with="pt2"
        )
        leapp.stop()
        leapp.compile_graph(visualize=False, validate=True, strict=True)
        path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        with open(path) as file:
            models = yaml.safe_load(file)["models"]
        for model in models.values():
            for desc in model["inputs"] + model["outputs"]:
                self.assertTrue(all(isinstance(size, int) for size in desc["shape"]))
                self.assertTrue(
                    {
                        "example_shape",
                        "dimension_ranges",
                        "dimension_expressions",
                    }.isdisjoint(desc)
                )
                self.assertEqual(
                    desc["axes"][1]["variable"]["min"], 2 if desc["name"] == "z" else 4
                )
        self.assertEqual(
            models["scale"]["inputs"][0]["axes"][0]["names"], ["left", "right"]
        )
        self.assertIn(
            "expression", models["pairs"]["outputs"][0]["axes"][1]["variable"]
        )
        manager = InferenceManager(path)
        self.assertEqual(tuple(manager.get_mock_input()["scale/x"].shape), (2, 8))
        for length in (4, 12, 32):
            fresh = torch.arange(2 * length, dtype=torch.float32).reshape(2, length)
            actual = manager({"scale/x": fresh})["pairs/z"]
            torch.testing.assert_close(actual, (fresh * 2).reshape(2, length // 2, 2))
        for length in (5, 34):
            with self.assertRaises(ValueError):
                manager({"scale/x": torch.ones(2, length)})


@pytest.fixture
def dynamic_session(tmp_path):
    leapp.start("dynamic_shapes", save_path=str(tmp_path))
    try:
        yield tmp_path / "dynamic_shapes" / "dynamic_shapes.yaml"
    finally:
        if ExportManager.is_interpret_graph_enabled():
            leapp.stop()
        _MANAGER.reset_tracing_lock()
        _MANAGER.reset_nodes()
        _MANAGER.set_dry_run_and_non_traced(False, [])


@pytest.mark.parametrize(
    "backend", ["pt2", "onnx-dynamo", "onnx-torchscript", "jit-script", "jit-trace"]
)
@pytest.mark.parametrize("library", [torch, np], ids=["torch", "numpy"])
def test_exported_models_accept_different_input_shapes(
    dynamic_session, backend, library
):
    """Check saved backend artifacts on sizes never used for capture."""

    def pair_tokens(x):
        return x.reshape(2, x.shape[1] // 2, 2)

    x = annotate.input_tensors(
        "pairs",
        TensorSemantics(
            "x",
            library.arange(16, dtype=library.float32).reshape(2, 8),
            axes=[
                None,
                Axis(variable=Variable("length", min=4, max=32, multiple_of=2)),
            ],
        ),
    )
    annotate.output_tensors("pairs", {"y": pair_tokens(x)}, export_with=backend)
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True, strict=True)
    manager = InferenceManager(str(dynamic_session))
    for length in (4, 12, 32):
        fresh = library.arange(2 * length, dtype=library.float32).reshape(2, length)
        torch.testing.assert_close(
            manager({"pairs/x": torch.as_tensor(fresh)})["pairs/y"],
            torch.as_tensor(pair_tokens(fresh)),
        )


@pytest.mark.parametrize("factory_name", ["zeros", "ones", "full", "arange"])
@pytest.mark.parametrize("library", [torch, np], ids=["torch", "numpy"])
def test_factories_use_runtime_shape(dynamic_session, factory_name, library):
    """Check factory capture and each library's output values, shapes, and dtypes."""

    def factory(length):
        if factory_name == "full":
            return library.full((length,), 3)
        return getattr(library, factory_name)(length)

    x = annotate.input_tensors(
        "factory",
        TensorSemantics(
            "x",
            library.ones(8, dtype=library.float32),
            axes=[Axis(variable=Variable("length", min=4, max=32))],
        ),
    )
    annotate.output_tensors("factory", {"y": factory(x.shape[0])}, export_with="pt2")
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True, strict=True)
    manager = InferenceManager(str(dynamic_session))
    for length in (4, 12, 32):
        actual = manager({"factory/x": torch.ones(length)})["factory/y"]
        torch.testing.assert_close(actual, torch.as_tensor(factory(length)))


@pytest.mark.parametrize(
    "other",
    [
        Variable("length", min=4, max=64),
        Variable("length", min=4, max=32, multiple_of=2),
    ],
    ids=["bounds", "divisibility"],
)
def test_conflicting_axis_declarations_are_rejected(dynamic_session, other):
    """Compilation rejects conflicting declarations across otherwise valid nodes."""
    for name, variable in (
        ("first", Variable("length", min=4, max=32)),
        ("second", other),
    ):
        x = annotate.input_tensors(
            name, TensorSemantics("x", torch.ones(8), axes=[Axis(variable=variable)])
        )
        annotate.output_tensors(name, {"y": x + 1}, export_with="pt2")
    leapp.stop()
    with pytest.raises(
        ValueError, match="Conflicting bounds for shared variable length"
    ):
        leapp.compile_graph(visualize=False, validate=True, strict=True)


def _export_two_input_model(function, inputs, path):
    left, right = annotate.input_tensors("combine", inputs)
    annotate.output_tensors("combine", {"y": function(left, right)}, export_with="pt2")
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True, strict=True)
    return InferenceManager(str(path))


def test_inputs_share_a_dynamic_dimension(dynamic_session):
    """Reloaded models enforce the same runtime length across both inputs."""
    axis = Axis(variable=Variable("length", min=4, max=32))
    manager = _export_two_input_model(
        torch.mul,
        [
            TensorSemantics("left", torch.ones(2, 8, 3), axes=[None, axis, None]),
            TensorSemantics("right", torch.ones(2, 8, 3), axes=[None, axis, None]),
        ],
        dynamic_session,
    )
    for length in (4, 12, 32):
        left = torch.arange(6 * length, dtype=torch.float32).reshape(2, length, 3)
        right = left + 1
        actual = manager({"combine/left": left, "combine/right": right})["combine/y"]
        torch.testing.assert_close(actual, left * right)
    with pytest.raises(ValueError, match="shared dimension length"):
        manager(
            {"combine/left": torch.ones(2, 12, 3), "combine/right": torch.ones(2, 8, 3)}
        )


def test_inputs_have_independent_dynamic_dimensions(dynamic_session):
    """Reloaded models derive output length from independently varying inputs."""

    def concatenate(left, right):
        return torch.cat((left, right), dim=1)

    manager = _export_two_input_model(
        concatenate,
        [
            TensorSemantics(
                "left",
                torch.ones(2, 8, 3),
                axes=[None, Axis(variable=Variable("length", min=4, max=32)), None],
            ),
            TensorSemantics(
                "right",
                torch.ones(2, 8, 3),
                axes=[
                    None,
                    Axis(variable=Variable("right_length", min=4, max=32)),
                    None,
                ],
            ),
        ],
        dynamic_session,
    )
    for left_length, right_length in ((4, 8), (12, 8), (12, 32)):
        left = torch.arange(6 * left_length, dtype=torch.float32).reshape(
            2, left_length, 3
        )
        right = torch.arange(6 * right_length, dtype=torch.float32).reshape(
            2, right_length, 3
        )
        actual = manager({"combine/left": left, "combine/right": right})["combine/y"]
        torch.testing.assert_close(actual, concatenate(left, right))


def test_dynamic_input_with_static_input(dynamic_session):
    """Reloaded models allow variable lengths while enforcing the static bias shape."""
    bias = torch.arange(3, dtype=torch.float32)
    manager = _export_two_input_model(
        torch.add,
        [
            TensorSemantics(
                "left",
                torch.ones(2, 8, 3),
                axes=[None, Axis(variable=Variable("length", min=4, max=32)), None],
            ),
            TensorSemantics("right", bias, axes=[None]),
        ],
        dynamic_session,
    )
    for length in (4, 12, 32):
        left = torch.arange(6 * length, dtype=torch.float32).reshape(2, length, 3)
        actual = manager({"combine/left": left, "combine/right": bias})["combine/y"]
        torch.testing.assert_close(actual, left + bias)
    with pytest.raises(ValueError, match="expected fixed dimension 3"):
        manager({"combine/left": torch.ones(2, 12, 3), "combine/right": torch.ones(4)})
