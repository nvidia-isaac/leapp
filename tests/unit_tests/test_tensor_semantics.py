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


def test_partial_and_custom_axes_yaml():
    ref = torch.zeros(2, 3, 4, 5)
    annotation = TensorSemantics(
        "tokens", ref, kind="embedding",
        axes=[Axis(names=["a", "b"]), Axis("token"), None, Axis(AxisKind.COMPONENT)],
    )
    entry = yaml.safe_load(yaml.dump(TensorDescription("tokens", ref, annotation).dict()))
    assert entry["axes"] == [
        {"names": ["a", "b"]}, {"kind": "token"}, None, {"kind": "component"}]
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
        "actions", torch.zeros(4, 2),
        axes=[Axis(AxisKind.TIME, period_ms=100), Axis(AxisKind.ELEMENT)],
    ).to_dict()
    assert metadata["axes"][0] == {"kind": "time", "period_ms": 100}
    with pytest.raises(ValueError, match="requires period_ms"):
        Axis("time")
    with pytest.raises(ValueError, match="only valid for a time axis"):
        Axis("token", period_ms=100)
    with pytest.raises(ValueError, match="at most one time"):
        TensorSemantics("x", torch.zeros(2, 2),
                        axes=[Axis("time", period_ms=1), Axis("time", period_ms=2)])


@pytest.mark.parametrize("expressed_in, expected", [
    (ExpressionFrame.SELF, {"selector": "self"}),
    (ExpressionFrame.REFERENCE, {"selector": "reference"}),
    ("self", {"frame": "self"}),
    ("world", {"frame": "world"}),
])
def test_expression_selector_is_distinct_from_frame_name(expressed_in, expected):
    annotation = leapp.frame_twist("v", torch.zeros(6), reference="base", expressed_in=expressed_in)
    assert annotation.to_dict()["expressed_in"] == expected


@pytest.mark.parametrize("kind", [Kind.FRAME_TWIST, Kind.FRAME_WRENCH, "frame/twist"])
def test_direct_spatial_defaults(kind):
    assert TensorSemantics("x", torch.zeros(6), kind=kind).to_dict()["expressed_in"] == {"selector": "self"}


def test_spatial_helper_signatures_and_required_arguments():
    ref = torch.zeros(6)
    with pytest.raises(TypeError):
        leapp.frame_pose("pose", ref, reference="world", expressed_in="world")
    with pytest.raises(TypeError):
        leapp.frame_wrench("w", ref, reference="world")
    with pytest.raises(ValueError, match="requires reference"):
        leapp.frame_wrench("w", ref, expressed_in=ExpressionFrame.REFERENCE)
    with pytest.raises(ValueError, match="required"):
        leapp.frame_twist("v", ref, reference=None)


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
    annotation = leapp.frame_pose("pose", torch.zeros(7), reference="world",
                                  axes=[Axis("component", names=names)])
    assert annotation.to_dict()["axes"][0]["names"] == names


@pytest.mark.parametrize("kind", list(Kind))
def test_all_kinds_have_public_helpers(kind):
    helper = getattr(leapp, kind.name.lower())
    kwargs = {"reference": "base"} if "reference" in inspect.signature(helper).parameters else {}
    annotation = helper("value", torch.zeros(1), **kwargs)
    assert annotation.kind == kind
    assert annotation.axes is None
    assert not annotation.is_setpoint


@pytest.mark.parametrize("ref", [torch.zeros(3, 4, 5, dtype=torch.uint8), np.zeros((3, 4, 5), dtype=np.uint8)])
def test_image_preserves_tensor_dtype_and_partial_axes(ref):
    annotation = leapp.image("camera", ref, axes=[Axis("component", names=["r", "g", "b"]), None, None])
    assert annotation.ref is ref
    entry = TensorDescription("camera", ref, annotation).dict()
    assert entry["dtype"] == "uint8"
    assert entry["shape"] == [3, 4, 5]


def test_setpoint_is_boolean_and_not_overridable_in_extra():
    with pytest.raises(TypeError, match="boolean"):
        TensorSemantics("x", torch.zeros(1), is_setpoint="false")
    with pytest.raises(ValueError, match="rather than extra"):
        TensorSemantics("x", torch.zeros(1), extra={"is_setpoint": True})


def test_warp_image_metadata_without_native_export_runtime():
    wp = pytest.importorskip("warp")
    ref = wp.zeros((3, 2, 2), dtype=wp.uint8, device="cpu")
    annotation = leapp.image("camera", ref, axes=[Axis("component", names=["r", "g", "b"]), None, None])
    entry = TensorDescription("camera", ref, annotation).dict()
    assert entry["dtype"] == "uint8"
    assert entry["shape"] == [3, 2, 2]


def test_update_preserves_spatial_defaults_and_custom_metadata():
    annotation = leapp.frame_twist("v", torch.zeros(6), reference="base")
    annotation.update({"axes": [Axis("component")], "description": "measured"})
    assert annotation.to_dict()["expressed_in"] == {"selector": "self"}
    assert annotation.to_dict()["description"] == "measured"
    with pytest.raises(TypeError, match="kind must"):
        TensorSemantics("x", torch.zeros(1), kind=42)
