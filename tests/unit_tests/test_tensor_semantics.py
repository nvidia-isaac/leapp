#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import inspect
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

import leapp
from leapp import (
    Axis,
    AxisKind,
    ExpressionFrame,
    Kind,
    TensorSemantics,
    GraphConfigs,
    TemporalAxis,
)
from leapp.leapp import _MANAGER as annotate
from leapp.utils.enums import InputKindEnum, OutputKindEnum
from tests.functional_tests.base import LEAPPFunctionalTestBase
from leapp.utils.tensor_description import TensorDescription


@pytest.mark.parametrize("axis_kind", [AxisKind.SYSTEM, "system"])
def test_system_axis_yaml(axis_kind):
    ref = torch.zeros(2, 3)
    annotation = leapp.joint_position(
        "q",
        ref,
        axes=[
            Axis(axis_kind, names=["machine0", "machine1"]),
            Axis(AxisKind.ELEMENT, names=["joint0", "joint1", "joint2"]),
        ],
    )
    entry = yaml.safe_load(yaml.dump(TensorDescription("q", ref, annotation).dict()))
    assert entry["axes"] == [
        {"kind": "system", "names": ["machine0", "machine1"]},
        {"kind": "element", "names": ["joint0", "joint1", "joint2"]},
    ]
    assert entry["shape"] == [2, 3]


def test_partial_and_custom_axes_yaml():
    ref = torch.zeros(2, 3, 4, 5)
    annotation = TensorSemantics(
        "tokens",
        ref,
        kind="embedding",
        axes=[Axis(names=["a", "b"]), Axis("token"), None, Axis(AxisKind.COMPONENT)],
    )
    entry = yaml.safe_load(
        yaml.dump(TensorDescription("tokens", ref, annotation).dict())
    )
    assert entry["axes"] == [
        {"names": ["a", "b"]},
        {"kind": "token"},
        None,
        {"kind": "component"},
    ]
    assert entry["kind"] == "embedding"
    assert entry["shape"] == [2, 3, 4, 5]
    assert "is_setpoint" not in entry


@pytest.mark.parametrize("axes", [[], [None, None], [Axis(names=["a"])]])
def test_axis_shape_mismatch(axes):
    with pytest.raises(ValueError):
        TensorSemantics("x", torch.zeros(2), axes=axes)


@pytest.mark.parametrize("period", [0, -1, float("nan"), float("inf"), True])
def test_invalid_time_period(period):
    with pytest.raises(ValueError):
        Axis("time", period_ms=period)


def test_time_axis_metadata_and_constraints():
    metadata = leapp.joint_effort(
        "actions",
        torch.zeros(4, 2),
        axes=[Axis(AxisKind.TIME, period_ms=100), Axis(AxisKind.ELEMENT)],
    ).to_dict()
    assert metadata["axes"][0] == {"kind": "time", "period_ms": 100}
    with pytest.raises(ValueError, match="requires period_ms"):
        Axis("time")
    with pytest.raises(ValueError, match="only valid for a time axis"):
        Axis("token", period_ms=100)
    with pytest.raises(ValueError, match="at most one time"):
        TensorSemantics(
            "x",
            torch.zeros(2, 2),
            axes=[Axis("time", period_ms=1), Axis("time", period_ms=2)],
        )


@pytest.mark.parametrize(
    "expressed_in, expected",
    [
        (ExpressionFrame.SELF, {"selector": "self"}),
        (ExpressionFrame.REFERENCE, {"selector": "reference"}),
        ("self", {"frame": "self"}),
        ("world", {"frame": "world"}),
    ],
)
def test_expression_selector_is_distinct_from_frame_name(expressed_in, expected):
    annotation = leapp.frame_twist(
        "v", torch.zeros(6), reference="base", expressed_in=expressed_in
    )
    assert annotation.to_dict()["spatial"]["expressed_in"] == expected


@pytest.mark.parametrize("kind", [Kind.FRAME_TWIST, Kind.FRAME_WRENCH])
def test_direct_spatial_defaults(kind):
    assert (
        TensorSemantics("x", torch.zeros(6), kind=kind).to_dict()["spatial"][
            "expressed_in"
        ]
        is None
    )


@pytest.mark.parametrize(
    "kind",
    [
        Kind.FRAME_POSE,
        "frame/position",
        Kind.FRAME_ORIENTATION,
        Kind.FRAME_WRENCH,
        "frame/force_application_position",
        "custom_quantity",
        None,
    ],
)
@pytest.mark.parametrize(
    "expression, expected",
    [
        ("base", {"frame": "base"}),
        (ExpressionFrame.SELF, {"selector": "self"}),
    ],
)
def test_generic_spatial_metadata_survives_yaml_export(kind, expression, expected):
    ref = torch.zeros(3)
    annotation = TensorSemantics(
        "value",
        ref,
        kind=kind,
        reference="base",
        expressed_in=expression,
    )
    entry = yaml.safe_load(
        yaml.dump(TensorDescription("value", ref, annotation).dict())
    )
    assert entry["spatial"] == {"reference": "base", "expressed_in": expected}


def test_spatial_helper_signatures_and_required_arguments():
    ref = torch.zeros(6)
    with pytest.raises(TypeError):
        leapp.frame_pose("pose", ref, reference="world", expressed_in="world")
    with pytest.raises(TypeError):
        leapp.frame_wrench("w", ref, reference="world")
    with pytest.raises(ValueError, match="cannot use REFERENCE"):
        leapp.frame_wrench("w", ref, expressed_in=ExpressionFrame.REFERENCE)
    assert leapp.frame_twist("v", ref).to_dict()["spatial"] == {
        "reference": None,
        "expressed_in": None,
    }
    assert leapp.frame_twist("v", ref, expressed_in=None).to_dict()["spatial"] == {
        "reference": None,
        "expressed_in": None,
    }
    assert (
        leapp.frame_twist("v", ref, expressed_in=ExpressionFrame.REFERENCE).reference
        is None
    )
    assert leapp.frame_pose("p", torch.zeros(7)).to_dict()["spatial"] == {
        "reference": None
    }


@pytest.mark.parametrize("axis_kind", [AxisKind.COMPONENT, "component"])
def test_helper_checks_components_beyond_generic_constructor(axis_kind):
    axes = [Axis(axis_kind, names=["r", "g", "b"])]
    ref = torch.zeros(3)
    TensorSemantics("v", ref, kind=Kind.FRAME_TWIST, axes=axes, reference="base")
    with pytest.raises(ValueError, match="components"):
        leapp.frame_twist("v", ref, axes=axes, reference="base")
    # An unspecified layout remains valid.
    assert leapp.frame_twist("v", ref, reference="base").axes is None


def test_pose_component_order_is_preserved():
    names = ["qw", "qx", "qy", "qz", "z", "y", "x"]
    annotation = leapp.frame_pose(
        "pose", torch.zeros(7), reference="world", axes=[Axis("component", names=names)]
    )
    assert annotation.to_dict()["axes"][0]["names"] == names


@pytest.mark.parametrize("kind", list(Kind))
def test_all_kinds_have_public_helpers(kind):
    helper = getattr(leapp, kind.name.lower())
    assert (
        inspect.signature(helper).parameters["ref"].default is inspect.Parameter.empty
    )
    with pytest.raises(TypeError, match="ref"):
        helper("value")
    kwargs = (
        {"reference": "base"}
        if "reference" in inspect.signature(helper).parameters
        else {}
    )
    annotation = helper("value", torch.zeros(1), **kwargs)
    assert annotation.kind == kind
    assert annotation.axes is None
    assert annotation.is_setpoint is None


@pytest.mark.parametrize(
    "ref",
    [torch.zeros(3, 4, 5, dtype=torch.uint8), np.zeros((3, 4, 5), dtype=np.uint8)],
)
def test_image_preserves_tensor_dtype_and_partial_axes(ref):
    annotation = leapp.image(
        "camera", ref, axes=[Axis("component", names=["r", "g", "b"]), None, None]
    )
    assert annotation.ref is ref
    entry = TensorDescription("camera", ref, annotation).dict()
    assert entry["dtype"] == "uint8"
    assert entry["shape"] == [3, 4, 5]


@pytest.mark.parametrize(
    "factory", [TensorSemantics, *[getattr(leapp, k.name.lower()) for k in Kind]]
)
@pytest.mark.parametrize(
    "kwargs", [{}, {"is_setpoint": None}, {"is_setpoint": False}, {"is_setpoint": True}]
)
def test_optional_setpoint_yaml(factory, kwargs):
    ref = torch.zeros(1)
    annotation = factory("value", ref, **kwargs)
    entry = yaml.safe_load(
        yaml.dump(TensorDescription("value", ref, annotation).dict())
    )
    expected = kwargs.get("is_setpoint")
    assert annotation.is_setpoint is expected
    if expected is None:
        assert "is_setpoint" not in entry
    else:
        assert entry["is_setpoint"] is expected


def test_setpoint_updates_preserve_false_and_clear_with_none():
    annotation = leapp.joint_position("q", torch.zeros(1), is_setpoint=True)
    annotation.update({"is_setpoint": False})
    assert annotation.to_dict()["is_setpoint"] is False
    annotation.update({"is_setpoint": None})
    assert "is_setpoint" not in annotation.to_dict()


@pytest.mark.parametrize("value", [0, 1, "false", "true", [], {}])
def test_setpoint_rejects_non_boolean_values(value):
    with pytest.raises(TypeError, match="boolean or None"):
        TensorSemantics("x", torch.zeros(1), is_setpoint=value)


def test_setpoint_is_boolean_and_not_overridable_in_extra():
    with pytest.raises(TypeError, match="boolean"):
        TensorSemantics("x", torch.zeros(1), is_setpoint="false")
    with pytest.raises(ValueError, match="rather than extra"):
        TensorSemantics(
            "x", torch.zeros(1), kind=Kind.JOINT_POSITION, extra={"is_setpoint": True}
        )


def test_warp_image_metadata_without_native_export_runtime():
    wp = pytest.importorskip("warp")
    ref = wp.zeros((3, 2, 2), dtype=wp.uint8, device="cpu")
    annotation = leapp.image(
        "camera", ref, axes=[Axis("component", names=["r", "g", "b"]), None, None]
    )
    entry = TensorDescription("camera", ref, annotation).dict()
    assert entry["dtype"] == "uint8"
    assert entry["shape"] == [3, 2, 2]


def test_update_preserves_spatial_defaults_and_custom_metadata():
    annotation = leapp.frame_twist("v", torch.zeros(6), reference="base")
    annotation.update({"axes": [Axis("component")], "description": "measured"})
    assert annotation.to_dict()["spatial"]["expressed_in"] is None
    assert annotation.to_dict()["description"] == "measured"
    with pytest.raises(TypeError, match="kind must"):
        TensorSemantics("x", torch.zeros(1), kind=42, axes=[None])


@pytest.mark.parametrize("expression", [ExpressionFrame.SELF, None, "base"])
def test_force_application_position(expression):
    axes = [
        Axis(AxisKind.ELEMENT, names=["tool"]),
        Axis(AxisKind.COMPONENT, names=["z", "x", "y"]),
    ]
    annotation = leapp.force_application_position(
        "point", torch.zeros(1, 3), axes=axes, expressed_in=expression
    )
    assert annotation.kind == Kind.FORCE_APPLICATION_POSITION
    assert annotation.expressed_in == expression
    with pytest.raises(ValueError, match="REFERENCE"):
        leapp.force_application_position(
            "point", torch.zeros(1, 3), expressed_in=ExpressionFrame.REFERENCE
        )
    axes[1] = Axis(AxisKind.COMPONENT, names=["vx", "vy", "vz"])
    with pytest.raises(ValueError, match="components"):
        leapp.force_application_position("point", torch.zeros(1, 3), axes=axes)


@pytest.mark.parametrize(
    "helper", [leapp.frame_twist, leapp.frame_wrench, leapp.force_application_position]
)
def test_omission_and_none_both_mean_world(helper):
    ref = torch.zeros(3)
    assert (
        helper("value", ref).to_dict()
        == helper("value", ref, expressed_in=None).to_dict()
    )
    assert helper("value", ref).expressed_in is None
    assert (
        helper("value", ref, expressed_in=ExpressionFrame.SELF).expressed_in
        is ExpressionFrame.SELF
    )
    assert helper("value", ref, expressed_in="world").to_dict()["spatial"][
        "expressed_in"
    ] == {"frame": "world"}


@pytest.mark.parametrize("kind", [*leapp.InputKindEnum, *leapp.OutputKindEnum])
def test_existing_enum_yaml_is_preserved(kind):
    ref = torch.zeros(2)
    annotation = TensorSemantics("q", ref, kind, ["hip", "knee"], {"units": "rad"})
    entry = yaml.safe_load(yaml.dump(TensorDescription("q", ref, annotation).dict()))
    assert entry == {
        "name": "q",
        "dtype": "float32",
        "shape": [2],
        "type": "tensor",
        "kind": kind.value,
        "element_names": [["hip", "knee"]],
        "units": "rad",
    }


def test_existing_constructor_and_temporal_update_are_preserved():
    annotation = TensorSemantics(
        "actions",
        torch.zeros(2, 3),
        leapp.OutputKindEnum.JOINT_TORQUES,
        [leapp.TemporalAxis(20), ["hip", "knee", "ankle"]],
        {"units": "Nm"},
    )
    annotation.update({"description": "commands"})
    assert annotation.to_dict() == {
        "kind": leapp.OutputKindEnum.JOINT_TORQUES,
        "element_names": ["__temporal_axis__", ["hip", "knee", "ankle"]],
        "temporal_period_ms": 20,
        "units": "Nm",
        "description": "commands",
    }
    params = inspect.signature(TensorSemantics).parameters
    for name in ("axes", "is_setpoint", "reference", "expressed_in"):
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("kind", ["frame/twist", "frame/custom", "", 42, None])
def test_existing_custom_metadata_is_not_reinterpreted(kind):
    extra = {"axes": "custom", "spatial": {"custom": True}, "is_setpoint": "custom"}
    annotation = TensorSemantics("x", torch.zeros(1), kind=kind, extra=extra.copy())
    expected = dict(extra)
    if kind is not None:
        expected["kind"] = kind
    assert annotation.to_dict() == expected
    annotation.update({"reference": "custom"})
    assert annotation.to_dict() == {**expected, "reference": "custom"}
    assert annotation.reference is None


def test_legacy_extra_keys_remain_extra_during_update():
    annotation = TensorSemantics("x", torch.zeros(1))
    metadata = {"axes": "custom", "is_setpoint": "custom", "expressed_in": "custom"}
    annotation.update(metadata)
    assert annotation.extra == metadata
    assert annotation.to_dict() == metadata


def test_axis_formats_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either element_names or axes"):
        TensorSemantics(
            "x", torch.zeros(1), element_names=["joint"], axes=[Axis("element")]
        )


def test_existing_enums_can_use_experimental_axes():
    annotation = TensorSemantics(
        "q",
        torch.zeros(1),
        kind=leapp.InputKindEnum.JOINT_POSITION,
        axes=[Axis(AxisKind.ELEMENT, names=["joint"])],
    )
    entry = TensorDescription("q", annotation.ref, annotation).dict()
    assert entry["kind"] == "state/joint/position"
    assert entry["axes"] == [{"kind": "element", "names": ["joint"]}]


@pytest.fixture
def semantic_session(tmp_path):
    leapp.start("semantics", save_path=str(tmp_path))
    try:
        yield tmp_path / "semantics"
    finally:
        if leapp.ExportManager.is_interpret_graph_enabled():
            leapp.stop()
        annotate.reset_nodes()


def _capture_semantic_policy(backend="jit"):
    x = leapp.annotate.input_tensors("policy", {"state": torch.tensor([1.0, 2.0])})
    leapp.annotate.output_tensors("policy", {"action": x * 2}, export_with=backend)


def test_metadata_without_values_validates_intrinsic_rules():
    assert TensorSemantics("value").ref is None
    with pytest.raises(TypeError, match="traceable"):
        TensorSemantics("value", ref=[1, 2])
    with pytest.raises(ValueError, match="at most one time"):
        TensorSemantics("value", axes=[Axis("time", period_ms=1)] * 2)
    with pytest.raises(TypeError, match="Axis or None"):
        TensorSemantics("value", axes=[{}])


@pytest.mark.parametrize("when", ["before", "registered", "captured", "stopped"])
def test_set_semantics_before_compilation(semantic_session, when):
    inputs = [
        TensorSemantics(
            "state", kind=Kind.JOINT_POSITION, axes=[Axis("element", names=["a", "b"])]
        )
    ]
    outputs = [
        TensorSemantics(
            "action", kind=Kind.JOINT_EFFORT, axes=[Axis("element", names=["a", "b"])]
        )
    ]

    def set_metadata():
        leapp.annotate.set_semantics("policy", inputs=inputs, outputs=outputs)

    if when == "before":
        set_metadata()
        assert not annotate.nodes  # A declaration alone creates no graph computation.
    x = leapp.annotate.input_tensors("policy", {"state": torch.tensor([1.0, 2.0])})
    if when == "registered":
        set_metadata()
    leapp.annotate.output_tensors("policy", {"action": x * 2}, export_with="jit")
    if when == "captured":
        set_metadata()
    leapp.stop()
    if when == "stopped":
        set_metadata()
    assert leapp.compile_graph(visualize=False)
    path = semantic_session / "semantics.yaml"
    description = yaml.safe_load(path.read_text())["models"]["policy"]
    assert description["inputs"][0]["kind"] == "joint/position"
    assert description["outputs"][0]["kind"] == "joint/effort"
    assert description["outputs"][0]["axes"] == [
        {"kind": "element", "names": ["a", "b"]}
    ]
    assert inputs[0].ref is None and outputs[0].ref is None
    runtime = leapp.InferenceManager(str(path))
    fresh = torch.tensor([-3.0, 7.0])
    torch.testing.assert_close(
        runtime({"policy/state": fresh})["policy/action"], fresh * 2
    )


def test_replacement_is_atomic_and_preserves_other_ports(semantic_session):
    leapp.annotate.set_semantics(
        "policy",
        inputs=[TensorSemantics("state", kind=Kind.JOINT_POSITION)],
        outputs=[
            TensorSemantics("action", kind=Kind.JOINT_EFFORT, extra={"units": "Nm"})
        ],
    )
    _capture_semantic_policy()
    node = annotate.nodes["policy"]
    original = node.get_description()
    with pytest.raises(ValueError, match="label count"):
        leapp.annotate.set_semantics(
            "policy",
            inputs=[TensorSemantics("state", kind=Kind.JOINT_VELOCITY)],
            outputs=[
                TensorSemantics(
                    "action", kind=Kind.JOINT_POSITION, axes=[Axis(names=["wrong"])]
                )
            ],
        )
    assert node.get_description() == original
    leapp.annotate.set_semantics(
        "policy", outputs=[TensorSemantics("action", kind=Kind.JOINT_POSITION)]
    )
    assert node.inputs[0].get_semantics() == {"kind": Kind.JOINT_POSITION}
    assert node.outputs[0].get_semantics() == {"kind": Kind.JOINT_POSITION}
    # Replacement clears omitted fields, including a formerly declared kind.
    leapp.annotate.set_semantics("policy", outputs=[TensorSemantics("action")])
    assert node.outputs[0].get_semantics() == {}
    leapp.annotate.set_semantics("policy", inputs=[], outputs=[])
    assert node.inputs[0].get_semantics() == {"kind": Kind.JOINT_POSITION}


@pytest.mark.parametrize(
    "bad",
    [
        {"outputs": [TensorSemantics()]},
        {"outputs": [TensorSemantics(" ")]},
        {"outputs": [TensorSemantics("action"), TensorSemantics("action")]},
        {
            "outputs": [
                TensorSemantics("action", kind=Kind.JOINT_POSITION, is_setpoint=True)
            ]
        },
        {"outputs": [TensorSemantics("action", extra={"shape": [99]})]},
        {"inputs": [TensorSemantics("state")], "states": [TensorSemantics("state")]},
    ],
)
def test_invalid_named_semantics_leave_store_unchanged(semantic_session, bad):
    leapp.annotate.set_semantics(
        "policy", outputs=[TensorSemantics("action", kind=Kind.JOINT_EFFORT)]
    )
    with pytest.raises(ValueError):
        leapp.annotate.set_semantics("policy", **bad)
    _capture_semantic_policy()
    assert annotate.nodes["policy"].outputs[0].get_semantics() == {
        "kind": Kind.JOINT_EFFORT
    }


def test_deferred_shape_validation_and_unresolved_names(semantic_session):
    leapp.annotate.set_semantics(
        "policy", inputs=[TensorSemantics("state", axes=[Axis(names=["wrong"])])]
    )
    with pytest.raises(ValueError, match="label count"):
        leapp.annotate.input_tensors("policy", {"state": torch.zeros(2)})
    leapp.annotate.set_semantics(
        "policy", inputs=[TensorSemantics("state", kind=Kind.JOINT_POSITION)]
    )
    _capture_semantic_policy()
    leapp.annotate.set_semantics("policy", outputs=[TensorSemantics("typo")])
    leapp.stop()
    with pytest.raises(ValueError, match="Unresolved outputs.*typo"):
        leapp.compile_graph(visualize=False)


def test_bound_wrappers_share_store_without_retaining_refs(semantic_session):
    annotation = leapp.joint_position(
        "state", torch.zeros(2), axes=[Axis(names=["a", "b"])]
    )
    leapp.annotate.set_semantics("policy", inputs=[annotation])
    annotation.axes[0].names[0] = "mutated"
    x = leapp.annotate.input_tensors("policy", {"state": torch.ones(2)})
    description = annotate.nodes["policy"].inputs[0]
    assert description.semantics.ref is None
    assert description.semantics.axes[0].names == ["a", "b"]
    leapp.annotate.set_semantics(
        "policy", outputs=[TensorSemantics("action", kind=Kind.JOINT_POSITION)]
    )
    leapp.annotate.output_tensors(
        "policy", leapp.joint_effort("action", x + 1), export_with="jit"
    )
    assert annotate.nodes["policy"].outputs[0].get_semantics() == {
        "kind": Kind.JOINT_EFFORT
    }
    leapp.annotate.set_semantics(
        "policy", outputs=[TensorSemantics("action", kind=Kind.JOINT_VELOCITY)]
    )
    assert annotate.nodes["policy"].outputs[0].get_semantics() == {
        "kind": Kind.JOINT_VELOCITY
    }


def test_raw_reentry_preserves_explicit_replacements(semantic_session):
    for index in range(2):
        if index:
            leapp.annotate.set_semantics(
                "policy", outputs=[TensorSemantics("action", kind=Kind.JOINT_VELOCITY)]
            )
        x = leapp.annotate.input_tensors("policy", {"state": torch.ones(2) * index})
        leapp.annotate.output_tensors("policy", {"action": x * 2}, export_with="jit")
    assert annotate.nodes["policy"].outputs[0].get_semantics() == {
        "kind": Kind.JOINT_VELOCITY
    }


def test_state_only_node_and_state_metadata_replacements(semantic_session):
    leapp.annotate.set_semantics(
        "policy", states=[TensorSemantics("memory", kind=Kind.JOINT_POSITION)]
    )
    for index in range(2):
        memory = leapp.annotate.state_tensors(
            "policy", {"memory": torch.ones(2) * index}
        )
        leapp.annotate.update_state("policy", {"memory": memory + 1})
        leapp.annotate.output_tensors("policy", {}, export_with="jit")
    node = annotate.nodes["policy"]
    assert (
        node.inputs[0].get_semantics()
        == node.outputs[0].get_semantics()
        == {"kind": Kind.JOINT_POSITION}
    )
    leapp.annotate.set_semantics(
        "policy", states=[TensorSemantics("memory", kind=Kind.JOINT_VELOCITY)]
    )
    assert (
        node.inputs[0].get_semantics()
        == node.outputs[0].get_semantics()
        == {"kind": Kind.JOINT_VELOCITY}
    )
    leapp.stop()
    assert leapp.compile_graph(visualize=False)
    path = semantic_session / "semantics.yaml"
    runtime = leapp.InferenceManager(str(path))
    with pytest.raises(RuntimeError, match="before calling compile_graph"):
        leapp.annotate.set_semantics(
            "policy", states=[TensorSemantics("memory", kind=Kind.JOINT_EFFORT)]
        )
    description = yaml.safe_load(path.read_text())
    assert description["models"]["policy"]["inputs"][0]["kind"] == "joint/velocity"
    assert description["models"]["policy"]["outputs"][0]["kind"] == "joint/velocity"
    assert description["pipeline"]["feedback_flow"] == {
        "policy/memory": ["policy/memory"]
    }
    runtime({})
    state = runtime.value_dict["policy"]["memory"]
    torch.testing.assert_close(state, torch.ones_like(state))


def test_state_annotations_support_bound_semantics(semantic_session):
    memory = leapp.annotate.state_tensors(
        "policy", leapp.joint_position("memory", torch.zeros(2))
    )
    leapp.annotate.update_state("policy", leapp.joint_velocity("memory", memory + 1))
    leapp.annotate.output_tensors("policy", {}, export_with="jit")
    node = annotate.nodes["policy"]
    assert node.inputs[0].get_semantics() == {"kind": Kind.JOINT_POSITION}
    assert node.outputs[0].get_semantics() == {"kind": Kind.JOINT_VELOCITY}


def test_input_output_names_are_separate_namespaces(semantic_session):
    leapp.annotate.set_semantics(
        "policy",
        inputs=[TensorSemantics("q", kind=Kind.JOINT_POSITION)],
        outputs=[TensorSemantics("q", kind=Kind.JOINT_EFFORT)],
    )
    x = leapp.annotate.input_tensors("policy", {"q": torch.ones(2)})
    leapp.annotate.output_tensors("policy", {"q": x * 2}, export_with="jit")
    assert (
        annotate.nodes["policy"].inputs[0].get_semantics()
        != annotate.nodes["policy"].outputs[0].get_semantics()
    )


def test_semantics_cannot_change_after_compilation(semantic_session):
    leapp.annotate.set_semantics(
        "policy", outputs=[TensorSemantics("action", kind=Kind.JOINT_EFFORT)]
    )
    _capture_semantic_policy()
    node = annotate.nodes["policy"]
    leapp.stop()
    assert leapp.compile_graph(visualize=False)
    path = semantic_session / "semantics.yaml"
    original = path.read_bytes()
    description = node.get_description()
    replacement = [TensorSemantics("action", kind=Kind.JOINT_POSITION)]
    updates = [
        lambda: leapp.annotate.set_semantics("policy", outputs=replacement),
        lambda: leapp.annotate.set_semantics("uncaptured", outputs=replacement),
        lambda: node.set_semantics(outputs=replacement),
        lambda: node.outputs[0].update_semantics({"kind": Kind.JOINT_POSITION}),
    ]
    for update in updates:
        with pytest.raises(RuntimeError, match="before calling compile_graph"):
            update()
    assert node.get_description() == description
    assert path.read_bytes() == original

    leapp.start("semantics", save_path=str(semantic_session.parent))
    leapp.annotate.set_semantics("policy", outputs=replacement)
    _capture_semantic_policy()
    leapp.stop()
    assert leapp.compile_graph(visualize=False)
    assert (
        yaml.safe_load(path.read_text())["models"]["policy"]["outputs"][0]["kind"]
        == "joint/position"
    )


def test_semantics_freeze_before_backend_compilation(semantic_session, monkeypatch):
    _capture_semantic_policy()
    leapp.stop()

    def compile_backend():
        with pytest.raises(RuntimeError, match="before calling compile_graph"):
            leapp.annotate.set_semantics("policy", outputs=[TensorSemantics("action")])
        raise ValueError("backend failed")

    monkeypatch.setattr(annotate, "compile_models", compile_backend)
    with pytest.raises(ValueError, match="backend failed"):
        leapp.compile_graph(visualize=False)


def test_precompiled_semantics_load_in_fresh_process(semantic_session):
    leapp.annotate.set_semantics(
        "policy",
        inputs=[TensorSemantics("state", kind=Kind.JOINT_POSITION)],
        outputs=[
            TensorSemantics(
                "action",
                kind=Kind.FRAME_TWIST,
                reference="base",
                expressed_in=ExpressionFrame.SELF,
            )
        ],
    )
    _capture_semantic_policy()
    leapp.stop()
    assert leapp.compile_graph(visualize=False)
    path = semantic_session / "semantics.yaml"
    model = semantic_session / "policy.pt"
    original_yaml, original_model = path.read_bytes(), model.read_bytes()
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys, torch, leapp
runtime = leapp.InferenceManager(sys.argv[1])
node = runtime.nodes['policy']
assert node.input_descriptions[0]['kind'] == 'joint/position'
assert node.output_descriptions[0]['spatial'] == {'reference': 'base', 'expressed_in': {'selector': 'self'}}
fresh = torch.tensor([11., -3.])
torch.testing.assert_close(runtime({'policy/state': fresh})['policy/action'], fresh * 2)
""",
            str(path),
        ],
        check=True,
    )
    assert path.read_bytes() == original_yaml
    assert model.read_bytes() == original_model


def test_precompiled_semantics_survive_onnx_port_renaming(semantic_session):
    leapp.annotate.set_semantics(
        "policy",
        inputs=[TensorSemantics("q", kind=Kind.JOINT_POSITION)],
        outputs=[TensorSemantics("q", kind=Kind.JOINT_EFFORT)],
    )
    q = leapp.annotate.input_tensors("policy", {"q": torch.ones(2)})
    leapp.annotate.output_tensors(
        "policy", {"q": q * 2}, export_with="onnx-torchscript"
    )
    leapp.stop()
    assert leapp.compile_graph(visualize=False)
    path = semantic_session / "semantics.yaml"
    description = yaml.safe_load(path.read_text())["models"]["policy"]
    assert description["inputs"][0]["name"] == "q_in"
    assert description["inputs"][0]["kind"] == "joint/position"
    assert description["outputs"][0]["name"] == "q_out"
    assert description["outputs"][0]["kind"] == "joint/effort"
    assert "aliases" not in description["inputs"][0]
    assert "aliases" not in description["outputs"][0]
    runtime = leapp.InferenceManager(str(path))
    fresh = torch.tensor([7.0, -3.0])
    torch.testing.assert_close(
        runtime({"policy/q_in": fresh})["policy/q_out"], fresh * 2
    )


def test_pt2_metadata_roundtrip_in_fresh_process(tmp_path):
    # PyTorch's PT2 exporter initializes its CUDA profiler. Keep that process
    # separate from the Warp CUPTI subscriptions used by other tests.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import pathlib, sys, torch, leapp
leapp.start('pt2_semantics', save_path=sys.argv[1], global_patching=False)
leapp.annotate.set_semantics('policy', outputs=[leapp.TensorSemantics('action', kind=leapp.Kind.JOINT_POSITION)])
x = leapp.annotate.input_tensors('policy', {'state': torch.ones(2)})
leapp.annotate.output_tensors('policy', {'action': x * 2}, export_with='pt2')
leapp.stop()
leapp.annotate.set_semantics('policy', outputs=[leapp.TensorSemantics('action', kind=leapp.Kind.JOINT_EFFORT)])
assert leapp.compile_graph(visualize=False)
directory = pathlib.Path(sys.argv[1]) / 'pt2_semantics'
runtime = leapp.InferenceManager(str(directory / 'pt2_semantics.yaml'))
assert runtime.nodes['policy'].output_descriptions[0]['kind'] == 'joint/effort'
fresh = torch.tensor([7., -3.])
torch.testing.assert_close(runtime({'policy/state': fresh})['policy/action'], fresh * 2)
""",
            str(tmp_path),
        ],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


def test_set_semantics_on_method_node(semantic_session):
    @leapp.annotate.method(node_name="policy", export_with="jit")
    def policy(state):
        action = state * 2
        return action

    leapp.annotate.set_semantics(
        "policy",
        inputs=[TensorSemantics("state", kind=Kind.JOINT_POSITION)],
        outputs=[TensorSemantics("action", kind=Kind.JOINT_EFFORT)],
    )
    policy(torch.ones(2))
    leapp.stop()
    assert leapp.compile_graph(visualize=False)
    description = yaml.safe_load((semantic_session / "semantics.yaml").read_text())
    assert description["models"]["policy"]["outputs"][0]["kind"] == "joint/effort"


class TestSemanticExport(LEAPPFunctionalTestBase):
    """Semantic metadata and runtime coverage alongside descriptor tests."""

    def _load_yaml(self):
        """Load the generated YAML config file."""
        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        self.assertTrue(os.path.exists(yaml_path), f"YAML file not found: {yaml_path}")
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

    def _get_node_io_from_yaml(self, config, node_name):
        """Get inputs and outputs dicts for a node from the YAML config."""
        self.assertIn("models", config)
        self.assertIn(node_name, config["models"])
        node = config["models"][node_name]
        return node.get("inputs", []), node.get("outputs", [])

    def _find_io_by_name(self, io_list, name):
        """Find an input/output entry by name in a YAML io list."""
        for entry in io_list:
            if entry["name"] == name:
                return entry
        return None

    def test_input_td_list_with_kind(self):
        """Test that kind metadata from input TensorSemantics appears in YAML."""
        joint_pos = torch.randn(1, 12)
        joint_vel = torch.randn(1, 12)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced_pos, traced_vel = annotate.input_tensors(
            "policy",
            [
                TensorSemantics(
                    name="joint_pos", ref=joint_pos, kind=InputKindEnum.JOINT_POSITION
                ),
                TensorSemantics(
                    name="joint_vel", ref=joint_vel, kind=InputKindEnum.JOINT_VELOCITY
                ),
            ],
        )

        output = traced_pos + traced_vel
        annotate.output_tensors("policy", {"command": output})
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, outputs = self._get_node_io_from_yaml(config, "policy")

        pos_entry = self._find_io_by_name(inputs, "joint_pos")
        vel_entry = self._find_io_by_name(inputs, "joint_vel")
        cmd_entry = self._find_io_by_name(outputs, "command")

        self.assertIsNotNone(pos_entry)
        self.assertIsNotNone(vel_entry)
        self.assertEqual(pos_entry["kind"], "state/joint/position")
        self.assertEqual(vel_entry["kind"], "state/joint/velocity")
        # Output without semantics should have no kind
        self.assertNotIn("kind", cmd_entry)

    def test_input_single_td(self):
        """Test passing a single TensorSemantics directly (not in a list)."""
        tensor = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "single_node",
            TensorSemantics(name="pos", ref=tensor, kind=InputKindEnum.JOINT_POSITION),
        )

        output = traced * 2.0
        annotate.output_tensors("single_node", {"out": output})
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, _ = self._get_node_io_from_yaml(config, "single_node")

        pos_entry = self._find_io_by_name(inputs, "pos")
        self.assertIsNotNone(pos_entry)
        self.assertEqual(pos_entry["kind"], "state/joint/position")
        self.assertEqual(pos_entry["shape"], [1, 4])

    def test_input_td_with_element_names(self):
        """Test that element_names from input TensorSemantics appears in YAML."""
        tensor = torch.randn(1, 3)
        names = ["x", "y", "z"]

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "elem_node",
            TensorSemantics(name="position", ref=tensor, element_names=names),
        )

        output = traced + 1.0
        annotate.output_tensors("elem_node", {"out": output})
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, _ = self._get_node_io_from_yaml(config, "elem_node")

        pos_entry = self._find_io_by_name(inputs, "position")
        self.assertIsNotNone(pos_entry)
        # element_names should be normalized to [["x", "y", "z"]]
        self.assertEqual(pos_entry["element_names"], [["x", "y", "z"]])

    def test_input_td_with_kind_and_element_names(self):
        """Test that both kind and element_names appear together in YAML."""
        tensor = torch.randn(1, 6)
        joint_names = ["hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"]

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "full_meta_node",
            TensorSemantics(
                name="joint_pos",
                ref=tensor,
                kind=InputKindEnum.JOINT_POSITION,
                element_names=joint_names,
            ),
        )

        output = traced * 0.5
        annotate.output_tensors("full_meta_node", {"out": output})
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, _ = self._get_node_io_from_yaml(config, "full_meta_node")

        entry = self._find_io_by_name(inputs, "joint_pos")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["kind"], "state/joint/position")
        self.assertEqual(entry["element_names"], [joint_names])
        self.assertEqual(entry["dtype"], "float32")
        self.assertEqual(entry["shape"], [1, 6])

    def test_input_and_output_td_without_source(self):
        """Test that TensorSemantics YAML no longer includes a source field."""
        tensor = torch.randn(1, 6)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "source_node",
            TensorSemantics(
                name="imu",
                ref=tensor,
                kind=InputKindEnum.BODY_ANGULAR_VELOCITY,
            ),
        )

        annotate.output_tensors(
            "source_node",
            [
                TensorSemantics(
                    name="filtered_imu",
                    ref=traced * 0.5,
                    kind=OutputKindEnum.BODY_ANGULAR_ACCELERATION,
                ),
            ],
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, outputs = self._get_node_io_from_yaml(config, "source_node")

        input_entry = self._find_io_by_name(inputs, "imu")
        output_entry = self._find_io_by_name(outputs, "filtered_imu")

        self.assertIsNotNone(input_entry)
        self.assertIsNotNone(output_entry)
        self.assertNotIn("source", input_entry)
        self.assertNotIn("source", output_entry)
        self.assertEqual(input_entry["kind"], "state/body/angular_velocity")
        self.assertEqual(output_entry["kind"], "target/body/angular_acceleration")

    def test_input_td_with_string_kind(self):
        """Test that a string kind appears in YAML."""
        tensor = torch.randn(1, 6)
        joint_names = ["hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"]

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "full_meta_node",
            TensorSemantics(
                name="joint_pos",
                ref=tensor,
                kind="my/custom/kind",
                element_names=joint_names,
            ),
        )

        output = traced * 0.5
        annotate.output_tensors("full_meta_node", {"out": output})
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, _ = self._get_node_io_from_yaml(config, "full_meta_node")

        entry = self._find_io_by_name(inputs, "joint_pos")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["kind"], "my/custom/kind")
        self.assertEqual(entry["element_names"], [joint_names])
        self.assertEqual(entry["dtype"], "float32")
        self.assertEqual(entry["shape"], [1, 6])

    def test_output_td_with_kind(self):
        """Test that kind metadata from output TensorSemantics appears in YAML."""
        tensor = torch.randn(1, 6)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors("out_kind_node", {"pos": tensor})

        command = traced * 2.0

        annotate.output_tensors(
            "out_kind_node",
            [
                TensorSemantics(
                    name="command", ref=command, kind=OutputKindEnum.JOINT_TORQUES
                ),
            ],
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        _, outputs = self._get_node_io_from_yaml(config, "out_kind_node")

        cmd_entry = self._find_io_by_name(outputs, "command")
        self.assertIsNotNone(cmd_entry)
        self.assertEqual(cmd_entry["kind"], "target/joint/torques")

    def test_output_td_with_element_names(self):
        """Test that element_names from output TensorSemantics appears in YAML."""
        tensor = torch.randn(1, 3)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors("out_elem_node", {"input": tensor})

        result = traced + 1.0

        annotate.output_tensors(
            "out_elem_node",
            [
                TensorSemantics(name="rgb", ref=result, element_names=["r", "g", "b"]),
            ],
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        _, outputs = self._get_node_io_from_yaml(config, "out_elem_node")

        rgb_entry = self._find_io_by_name(outputs, "rgb")
        self.assertIsNotNone(rgb_entry)
        self.assertEqual(rgb_entry["element_names"], [["r", "g", "b"]])

    def test_static_output_td_with_kind(self):
        """Test that kind metadata from static output TensorSemantics appears in YAML."""
        tensor = torch.randn(1, 6)
        static_output = torch.ones(1, 6)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors("static_out_kind_node", {"pos": tensor})

        annotate.output_tensors(
            "static_out_kind_node",
            {"computed": traced * 2.0},
            static_outputs=TensorSemantics(
                name="command_bias",
                ref=static_output,
                kind=OutputKindEnum.JOINT_TORQUES,
            ),
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        _, outputs = self._get_node_io_from_yaml(config, "static_out_kind_node")

        static_entry = self._find_io_by_name(outputs, "command_bias")
        self.assertIsNotNone(static_entry)
        self.assertEqual(static_entry["kind"], "target/joint/torques")

    def test_both_input_and_output_td(self):
        """Test TensorSemantics on both inputs and outputs in the same node."""
        pos = torch.randn(1, 6)
        vel = torch.randn(1, 6)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced_pos, traced_vel = annotate.input_tensors(
            "both_node",
            [
                TensorSemantics(name="pos", ref=pos, kind=InputKindEnum.JOINT_POSITION),
                TensorSemantics(name="vel", ref=vel, kind=InputKindEnum.JOINT_VELOCITY),
            ],
        )

        command = traced_pos + traced_vel

        annotate.output_tensors(
            "both_node",
            [
                TensorSemantics(
                    name="torques", ref=command, kind=OutputKindEnum.JOINT_TORQUES
                ),
            ],
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, outputs = self._get_node_io_from_yaml(config, "both_node")

        self.assertEqual(
            self._find_io_by_name(inputs, "pos")["kind"], "state/joint/position"
        )
        self.assertEqual(
            self._find_io_by_name(inputs, "vel")["kind"], "state/joint/velocity"
        )
        self.assertEqual(
            self._find_io_by_name(outputs, "torques")["kind"], "target/joint/torques"
        )

    def test_input_td_with_extra_fields_flattened_into_yaml(self):
        """Test that TensorSemantics.extra fields are emitted as top-level YAML keys."""
        tensor = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "extra_meta_node",
            TensorSemantics(
                name="joint_pos",
                ref=tensor,
                kind=InputKindEnum.JOINT_POSITION,
                extra={"id": "abc", "frame": "base"},
            ),
        )

        output = traced * 2.0
        annotate.output_tensors("extra_meta_node", {"out": output})
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, _ = self._get_node_io_from_yaml(config, "extra_meta_node")

        entry = self._find_io_by_name(inputs, "joint_pos")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["kind"], "state/joint/position")
        self.assertEqual(entry["id"], "abc")
        self.assertEqual(entry["frame"], "base")
        self.assertNotIn("extra", entry)

    def test_graph_configs_appear_in_pipeline_yaml(self):
        """Test GraphConfigs fields and extras are emitted as graph-level metadata."""
        tensor = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors("frequency_node", {"input": tensor})
        annotate.output_tensors("frequency_node", {"output": traced + 1.0})
        leapp.stop()
        leapp.compile_graph(
            visualize=False,
            graph_configs=GraphConfigs(
                frequency=50,
                extra={"runtime": "isaac_lab"},
            ),
        )

        config = self._load_yaml()

        self.assertIn("pipeline", config)
        self.assertEqual(config["pipeline"]["configs"]["frequency"], 50)
        self.assertEqual(config["pipeline"]["configs"]["runtime"], "isaac_lab")
        self.assertNotIn("frequency", config["pipeline"])
        self.assertNotIn("runtime", config["pipeline"])

    def test_temporal_period_marker_appears_in_output_yaml(self):
        """Test TemporalAxis emits temporal axis and period metadata."""
        tensor = torch.randn(2, 3)
        names = ["hip", "knee", "ankle"]

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors("chunk_node", {"input": tensor})

        annotate.output_tensors(
            "chunk_node",
            [
                TensorSemantics(
                    name="actions",
                    ref=traced + 1.0,
                    element_names=[TemporalAxis(period_ms=100), names],
                ),
                TensorSemantics(name="plain_output", ref=traced - 1.0),
            ],
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        _, outputs = self._get_node_io_from_yaml(config, "chunk_node")

        action_entry = self._find_io_by_name(outputs, "actions")
        plain_entry = self._find_io_by_name(outputs, "plain_output")

        self.assertEqual(action_entry["element_names"], ["__temporal_axis__", names])
        self.assertEqual(action_entry["temporal_period_ms"], 100)
        self.assertNotIn("temporal_period_ms", plain_entry)
        self.assertNotIn("element_names", plain_entry)

    def test_no_metadata_no_extra_yaml_fields(self):
        """Test that tensors without TensorSemantics have no semantic fields in YAML."""
        tensor = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors("plain_node", {"input": tensor})

        output = traced + 1.0
        annotate.output_tensors("plain_node", {"output": output})
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, outputs = self._get_node_io_from_yaml(config, "plain_node")

        input_entry = self._find_io_by_name(inputs, "input")
        output_entry = self._find_io_by_name(outputs, "output")

        self.assertNotIn("kind", input_entry)
        self.assertNotIn("source", input_entry)
        self.assertNotIn("element_names", input_entry)
        self.assertNotIn("kind", output_entry)
        self.assertNotIn("source", output_entry)
        self.assertNotIn("element_names", output_entry)

    def test_tensor_semantics_rejects_temporal_period_ms_argument(self):
        """Test temporal period is only set through TemporalAxis."""
        tensor = torch.randn(2, 3)

        with self.assertRaises(TypeError):
            TensorSemantics(
                name="actions",
                ref=tensor,
                element_names=[["hip", "knee", "ankle"]],
                temporal_period_ms=100,
            )

    def test_mixed_td_and_raw_raises(self):
        """Test that mixing TensorSemantics and raw tensors in a list raises TypeError."""
        t1 = torch.randn(1, 3)
        t2 = torch.randn(1, 3)

        leapp.start(name=self.TEST_GRAPH_NAME)
        with self.assertRaises(TypeError):
            annotate.input_tensors(
                "fail_node",
                [
                    t1,
                    TensorSemantics(
                        name="td", ref=t2, kind=InputKindEnum.JOINT_POSITION
                    ),
                ],
            )
        leapp.stop()

    def test_duplicate_td_names_raises(self):
        """Test that two TensorSemantics with the same name raise an error."""
        t1 = torch.randn(1, 3)
        t2 = torch.randn(1, 3)

        leapp.start(name=self.TEST_GRAPH_NAME)
        with self.assertRaises(Exception):
            annotate.input_tensors(
                "dup_node",
                [
                    TensorSemantics(name="same_name", ref=t1),
                    TensorSemantics(name="same_name", ref=t2),
                ],
            )
        leapp.stop()

    def test_td_inputs_graph_structure(self):
        """Test that TensorSemantics inputs produce correct graph structure (node count, connections)."""
        pos = torch.randn(1, 4)
        vel = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced_pos, traced_vel = annotate.input_tensors(
            "struct_node",
            [
                TensorSemantics(name="pos", ref=pos, kind=InputKindEnum.JOINT_POSITION),
                TensorSemantics(name="vel", ref=vel, kind=InputKindEnum.JOINT_VELOCITY),
            ],
        )

        output = traced_pos + traced_vel
        annotate.output_tensors("struct_node", {"command": output})
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=2, outputs=1, internal_connections=0
        )

    def test_semantic_input_reentry_does_not_crash(self):
        """Reentry with TensorSemantics inputs must not KeyError on semantic-only keys."""
        joint_pos = torch.randn(1, 6)

        leapp.start(name=self.TEST_GRAPH_NAME)

        for _ in range(2):
            traced = annotate.input_tensors(
                "policy",
                [
                    TensorSemantics(
                        name="joint_pos",
                        ref=joint_pos,
                        kind=InputKindEnum.JOINT_POSITION,
                    ),
                ],
            )
            out = traced * 2.0
            annotate.output_tensors("policy", {"cmd": out}, export_with="jit")

        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, _ = self._get_node_io_from_yaml(config, "policy")
        entry = self._find_io_by_name(inputs, "joint_pos")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["kind"], "state/joint/position")

    def test_semantic_output_reentry_does_not_crash(self):
        """Reentry with TensorSemantics outputs must not KeyError on semantic-only keys."""
        tensor = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)

        for _ in range(2):
            traced = annotate.input_tensors("out_reentry", {"x": tensor})
            result = traced + 1.0
            annotate.output_tensors(
                "out_reentry",
                [
                    TensorSemantics(
                        name="torques", ref=result, kind=OutputKindEnum.JOINT_TORQUES
                    ),
                ],
                export_with="jit",
            )

        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        _, outputs = self._get_node_io_from_yaml(config, "out_reentry")
        entry = self._find_io_by_name(outputs, "torques")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["kind"], "target/joint/torques")

    def test_adopted_declaration_supplies_semantics_the_port_lacks(self):
        """The surviving port keeps its name but gains a description it lacked.

        Adoption bypasses the registration that normally attaches semantics, so
        a kind written on the second declaration would otherwise be dropped even
        though exactly one port exists to carry it. The name is not adopted with
        it, because the first declaration already decided the interface.
        """
        buffer = torch.tensor([1.0, 2.0, 3.0])
        alias = buffer.detach()

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced_buffer = annotate.input_tensors("policy", {"buffer": buffer})
        annotate.input_tensors(
            "policy",
            [
                TensorSemantics(
                    name="alias", ref=alias, kind=InputKindEnum.JOINT_POSITION
                ),
            ],
        )
        annotate.output_tensors(
            "policy", {"action": traced_buffer * 2.0}, export_with="jit"
        )
        node = annotate.get_nodes()["policy"]
        leapp.stop()

        ports = [description.name_str for description in node.inputs]
        self.assertEqual(ports, ["buffer"], "adoption changed the exported port")
        self.assertEqual(
            node.inputs[0].semantics.kind,
            InputKindEnum.JOINT_POSITION,
            "the adopted declaration's semantics were dropped",
        )


class TestStructuredSemantics(LEAPPFunctionalTestBase):
    def test_helpers_export_and_runtime(self):
        from leapp import (
            Axis,
            AxisKind,
            Kind,
            frame_twist,
            image,
            joint_effort,
            joint_position,
        )

        q = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[3.0, 5.0]])
        velocity = torch.zeros(1, 1, 6)
        pixels = torch.zeros(1, 3, 2, 2, dtype=torch.uint8)
        joints = [
            Axis(AxisKind.SYSTEM, names=["robot0"]),
            Axis(AxisKind.ELEMENT, names=["shoulder", "elbow"]),
        ]
        leapp.start(name=self.TEST_GRAPH_NAME)
        tq, tt, tv, ti = leapp.annotate.input_tensors(
            "policy",
            [
                joint_position("q", q, axes=joints),
                joint_position("target", target, axes=joints, is_setpoint=True),
                frame_twist(
                    "twist",
                    velocity,
                    reference="base",
                    axes=[
                        Axis(AxisKind.SYSTEM),
                        Axis(AxisKind.ELEMENT, names=["tool"]),
                        Axis(
                            AxisKind.COMPONENT,
                            names=["vx", "vy", "vz", "wx", "wy", "wz"],
                        ),
                    ],
                ),
                image("camera", pixels, axes=[Axis(AxisKind.SYSTEM), None, None, None]),
            ],
        )
        command = tt - tq + tv.sum() + ti.float().sum()
        leapp.annotate.output_tensors(
            "policy", joint_effort("effort", command, axes=joints), export_with="jit"
        )
        leapp.stop()
        self.assertTrue(leapp.compile_graph(visualize=False))
        path = os.path.join(self.TEST_GRAPH_NAME, self.TEST_GRAPH_NAME + ".yaml")
        with open(path) as stream:
            config = yaml.safe_load(stream)
        inputs = config["models"]["policy"]["inputs"]
        self.assertEqual(inputs[1]["is_setpoint"], True)
        self.assertNotIn("is_setpoint", inputs[0])
        self.assertEqual(
            inputs[2]["spatial"],
            {"reference": "base", "expressed_in": None},
        )
        self.assertEqual(inputs[2]["axes"][1], {"kind": "element", "names": ["tool"]})
        self.assertEqual(
            config["models"]["policy"]["outputs"][0]["kind"], Kind.JOINT_EFFORT.value
        )
        runtime = leapp.InferenceManager(path)
        result = runtime(
            {
                "policy/q": q,
                "policy/target": target,
                "policy/twist": velocity,
                "policy/camera": pixels,
            }
        )
        torch.testing.assert_close(result["policy/effort"], target - q)

    def test_optional_setpoint_values_survive_export(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        omitted, unspecified, measured, target = leapp.annotate.input_tensors(
            "policy",
            [
                leapp.joint_position("omitted", torch.ones(2)),
                leapp.joint_position("unspecified", torch.ones(2), is_setpoint=None),
                leapp.joint_position("measured", torch.ones(2), is_setpoint=False),
                leapp.joint_position("target", torch.ones(2), is_setpoint=True),
            ],
        )
        leapp.annotate.output_tensors(
            "policy",
            leapp.joint_effort(
                "effort", omitted + unspecified + target - measured, is_setpoint=False
            ),
            export_with="jit",
        )
        leapp.stop()
        self.assertTrue(leapp.compile_graph(visualize=False))
        path = os.path.join(self.TEST_GRAPH_NAME, self.TEST_GRAPH_NAME + ".yaml")
        with open(path) as stream:
            node = yaml.safe_load(stream)["models"]["policy"]
        self.assertNotIn("is_setpoint", node["inputs"][0])
        self.assertNotIn("is_setpoint", node["inputs"][1])
        self.assertIs(node["inputs"][2]["is_setpoint"], False)
        self.assertIs(node["inputs"][3]["is_setpoint"], True)
        self.assertIs(node["outputs"][0]["is_setpoint"], False)
        runtime = leapp.InferenceManager(path)
        result = runtime(
            {
                f"policy/{name}": torch.ones(2)
                for name in ("omitted", "unspecified", "measured", "target")
            }
        )
        torch.testing.assert_close(result["policy/effort"], torch.full((2,), 2.0))

    def test_setpoints_rejected_on_outputs_and_static_outputs(self):
        for static in (False, True):
            with self.subTest(static=static):
                leapp.start(name=self.TEST_GRAPH_NAME)
                value = leapp.annotate.input_tensors("policy", {"x": torch.zeros(2)})
                bad = leapp.joint_position("target", value + 1, is_setpoint=True)
                with self.assertRaisesRegex(ValueError, "only to inputs"):
                    if static:
                        leapp.annotate.output_tensors(
                            "policy", {"out": value + 1}, static_outputs=bad
                        )
                    else:
                        leapp.annotate.output_tensors("policy", bad)
                leapp.stop()


class TestMixedSemantics(LEAPPFunctionalTestBase):
    def test_existing_and_experimental_annotations_share_graph_and_version(self):
        q = torch.tensor([1.0, 2.0])
        target = torch.tensor([3.0, 5.0])
        leapp.start(name=self.TEST_GRAPH_NAME)
        for _ in range(2):
            measured, requested = leapp.annotate.input_tensors(
                "policy",
                [
                    TensorSemantics("q", q, InputKindEnum.JOINT_POSITION, ["a", "b"]),
                    leapp.joint_position(
                        "target",
                        target,
                        is_setpoint=True,
                        axes=[leapp.Axis(leapp.AxisKind.ELEMENT, names=["a", "b"])],
                    ),
                ],
            )
            leapp.annotate.output_tensors(
                "policy",
                [
                    TensorSemantics(
                        "legacy",
                        requested - measured,
                        OutputKindEnum.JOINT_TORQUES,
                        ["a", "b"],
                    ),
                    leapp.joint_effort("new", requested - measured),
                ],
                export_with="jit",
            )
        leapp.stop()
        self.assertTrue(leapp.compile_graph(visualize=False))
        path = os.path.join(self.TEST_GRAPH_NAME, self.TEST_GRAPH_NAME + ".yaml")
        with open(path) as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(
            config["system information"]["leapp version"], leapp.__version__
        )
        self.assertEqual(
            config["system information"]["leapp config version"],
            leapp.__config_version__,
        )
        node = config["models"]["policy"]
        self.assertEqual(node["inputs"][0]["kind"], "state/joint/position")
        self.assertEqual(node["inputs"][0]["element_names"], [["a", "b"]])
        self.assertNotIn("axes", node["inputs"][0])
        self.assertEqual(node["inputs"][1]["kind"], "joint/position")
        self.assertTrue(node["inputs"][1]["is_setpoint"])
        self.assertEqual(node["outputs"][0]["kind"], "target/joint/torques")
        self.assertEqual(node["outputs"][1]["kind"], "joint/effort")
        runtime = leapp.InferenceManager(path)
        result = runtime({"policy/q": q, "policy/target": target})
        torch.testing.assert_close(result["policy/legacy"], target - q)
        torch.testing.assert_close(result["policy/new"], target - q)

    def test_existing_output_extra_setpoint_is_not_reinterpreted(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        value = leapp.annotate.input_tensors("policy", {"q": torch.ones(2)})
        leapp.annotate.output_tensors(
            "policy",
            TensorSemantics(
                "out",
                value + 1,
                OutputKindEnum.JOINT_POSITION,
                extra={
                    "is_setpoint": True,
                    "axes": "custom",
                    "spatial": {"custom": True},
                },
            ),
            export_with="jit",
        )
        leapp.stop()
        self.assertTrue(leapp.compile_graph(visualize=False))
        path = os.path.join(self.TEST_GRAPH_NAME, self.TEST_GRAPH_NAME + ".yaml")
        with open(path) as stream:
            config = yaml.safe_load(stream)
        entry = config["models"]["policy"]["outputs"][0]
        self.assertEqual(entry["kind"], "target/joint/position")
        self.assertIs(entry["is_setpoint"], True)
        self.assertEqual(entry["axes"], "custom")
        self.assertEqual(entry["spatial"], {"custom": True})
