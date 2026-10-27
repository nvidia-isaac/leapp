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

import numpy as np
import pytest
import torch
import yaml

import leapp
from leapp import Axis, AxisKind, ExpressionFrame, Kind, TensorSemantics
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
