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
import os
import unittest
from unittest import mock
import yaml
import torch
import leapp
from leapp import GraphConfigs, TensorSemantics, Axis, AxisKind
from leapp.leapp import _MANAGER as annotate
from leapp.utils import utils
from leapp.utils.enums import Kind
from .base import LEAPPFunctionalTestBase


class TestConfigGeneration(LEAPPFunctionalTestBase):
    """Tests that semantic metadata from TensorSemantics appears in generated YAML configs."""

    def _load_yaml(self):
        """Load the generated YAML config file."""
        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        self.assertTrue(os.path.exists(yaml_path), f"YAML file not found: {yaml_path}")
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_node_io_from_yaml(self, config, node_name):
        """Get inputs and outputs dicts for a node from the YAML config."""
        self.assertIn('models', config)
        self.assertIn(node_name, config['models'])
        node = config['models'][node_name]
        return node.get('inputs', []), node.get('outputs', [])

    def _find_io_by_name(self, io_list, name):
        """Find an input/output entry by name in a YAML io list."""
        for entry in io_list:
            if entry['name'] == name:
                return entry
        return None

    # =========================================================================
    # Input TensorSemantics tests
    # =========================================================================

    def test_input_td_list_with_kind(self):
        """Test that kind metadata from input TensorSemantics appears in YAML."""
        joint_pos = torch.randn(1, 12)
        joint_vel = torch.randn(1, 12)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced_pos, traced_vel = annotate.input_tensors("policy", [
            TensorSemantics(name="joint_pos", ref=joint_pos, kind=Kind.JOINT_POSITION),
            TensorSemantics(name="joint_vel", ref=joint_vel, kind=Kind.JOINT_VELOCITY),
        ])

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
        self.assertEqual(pos_entry['kind'], "joint/position")
        self.assertEqual(vel_entry['kind'], "joint/velocity")
        # Output without semantics should have no kind
        self.assertNotIn('kind', cmd_entry)

    def test_input_single_td(self):
        """Test passing a single TensorSemantics directly (not in a list)."""
        tensor = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "single_node",
            TensorSemantics(name="pos", ref=tensor, kind=Kind.JOINT_POSITION),
        )

        output = traced * 2.0
        annotate.output_tensors("single_node", {"out": output})
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, _ = self._get_node_io_from_yaml(config, "single_node")

        pos_entry = self._find_io_by_name(inputs, "pos")
        self.assertIsNotNone(pos_entry)
        self.assertEqual(pos_entry['kind'], "joint/position")
        self.assertEqual(pos_entry['shape'], [1, 4])

    def test_input_td_with_axes(self):
        """Test that axes from input TensorSemantics appears in YAML."""
        tensor = torch.randn(1, 3)
        names = ["x", "y", "z"]

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "elem_node",
            TensorSemantics(name="position", ref=tensor, axes=[None, Axis(names=names)]),
        )

        output = traced + 1.0
        annotate.output_tensors("elem_node", {"out": output})
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, _ = self._get_node_io_from_yaml(config, "elem_node")

        pos_entry = self._find_io_by_name(inputs, "position")
        self.assertIsNotNone(pos_entry)
        # Preserve the unlabeled batch dimension.
        self.assertEqual(pos_entry['axes'], [None, {'names': ["x", "y", "z"]}])

    def test_input_td_with_kind_and_axes(self):
        """Test that both kind and axes appear together in YAML."""
        tensor = torch.randn(1, 6)
        joint_names = ["hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"]

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "full_meta_node",
            TensorSemantics(name="joint_pos", ref=tensor,
                            kind=Kind.JOINT_POSITION,
                            axes=[None, Axis(AxisKind.ELEMENT, names=joint_names)]),
        )

        output = traced * 0.5
        annotate.output_tensors("full_meta_node", {"out": output})
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, _ = self._get_node_io_from_yaml(config, "full_meta_node")

        entry = self._find_io_by_name(inputs, "joint_pos")
        self.assertIsNotNone(entry)
        self.assertEqual(entry['kind'], "joint/position")
        self.assertEqual(entry['axes'], [None, {'kind': 'element', 'names': joint_names}])
        self.assertEqual(entry['dtype'], "float32")
        self.assertEqual(entry['shape'], [1, 6])

    def test_input_and_output_td_without_source(self):
        """Test that TensorSemantics YAML no longer includes a source field."""
        tensor = torch.randn(1, 6)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "source_node",
            TensorSemantics(
                name="imu",
                ref=tensor,
                kind=Kind.FRAME_ANGULAR_VELOCITY,
            ),
        )

        annotate.output_tensors("source_node", [
            TensorSemantics(
                name="filtered_imu",
                ref=traced * 0.5,
                kind=Kind.FRAME_ANGULAR_ACCELERATION,
            ),
        ])
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
        self.assertEqual(input_entry["kind"], "frame/angular_velocity")
        self.assertEqual(output_entry["kind"], "frame/angular_acceleration")
    
    def test_input_td_with_string_kind(self):
        """Test that a string kind appears in YAML."""
        tensor = torch.randn(1, 6)
        joint_names = ["hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"]

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "full_meta_node",
            TensorSemantics(name="joint_pos", ref=tensor,
                            kind="my/custom/kind",
                            axes=[None, Axis(AxisKind.ELEMENT, names=joint_names)]),
        )

        output = traced * 0.5
        annotate.output_tensors("full_meta_node", {"out": output})
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, _ = self._get_node_io_from_yaml(config, "full_meta_node")

        entry = self._find_io_by_name(inputs, "joint_pos")
        self.assertIsNotNone(entry)
        self.assertEqual(entry['kind'], "my/custom/kind")
        self.assertEqual(entry['axes'], [None, {'kind': 'element', 'names': joint_names}])
        self.assertEqual(entry['dtype'], "float32")
        self.assertEqual(entry['shape'], [1, 6])

    # =========================================================================
    # Output TensorSemantics tests
    # =========================================================================

    def test_output_td_with_kind(self):
        """Test that kind metadata from output TensorSemantics appears in YAML."""
        tensor = torch.randn(1, 6)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors("out_kind_node", {"pos": tensor})

        command = traced * 2.0

        annotate.output_tensors("out_kind_node", [
            TensorSemantics(name="command", ref=command, kind=Kind.JOINT_EFFORT),
        ])
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        _, outputs = self._get_node_io_from_yaml(config, "out_kind_node")

        cmd_entry = self._find_io_by_name(outputs, "command")
        self.assertIsNotNone(cmd_entry)
        self.assertEqual(cmd_entry['kind'], "joint/effort")

    def test_output_td_with_axes(self):
        """Test that axes from output TensorSemantics appears in YAML."""
        tensor = torch.randn(1, 3)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors("out_elem_node", {"input": tensor})

        result = traced + 1.0

        annotate.output_tensors("out_elem_node", [
            TensorSemantics(name="rgb", ref=result, axes=[None, Axis(AxisKind.COMPONENT, names=["r", "g", "b"])]),
        ])
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        _, outputs = self._get_node_io_from_yaml(config, "out_elem_node")

        rgb_entry = self._find_io_by_name(outputs, "rgb")
        self.assertIsNotNone(rgb_entry)
        self.assertEqual(rgb_entry['axes'], [None, {'kind': 'component', 'names': ["r", "g", "b"]}])

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
                kind=Kind.JOINT_EFFORT,
            ),
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        _, outputs = self._get_node_io_from_yaml(config, "static_out_kind_node")

        static_entry = self._find_io_by_name(outputs, "command_bias")
        self.assertIsNotNone(static_entry)
        self.assertEqual(static_entry['kind'], "joint/effort")

    # =========================================================================
    # Both inputs and outputs
    # =========================================================================

    def test_both_input_and_output_td(self):
        """Test TensorSemantics on both inputs and outputs in the same node."""
        pos = torch.randn(1, 6)
        vel = torch.randn(1, 6)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced_pos, traced_vel = annotate.input_tensors("both_node", [
            TensorSemantics(name="pos", ref=pos, kind=Kind.JOINT_POSITION),
            TensorSemantics(name="vel", ref=vel, kind=Kind.JOINT_VELOCITY),
        ])

        command = traced_pos + traced_vel

        annotate.output_tensors("both_node", [
            TensorSemantics(name="torques", ref=command, kind=Kind.JOINT_EFFORT),
        ])
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, outputs = self._get_node_io_from_yaml(config, "both_node")

        self.assertEqual(self._find_io_by_name(inputs, "pos")['kind'], "joint/position")
        self.assertEqual(self._find_io_by_name(inputs, "vel")['kind'], "joint/velocity")
        self.assertEqual(self._find_io_by_name(outputs, "torques")['kind'], "joint/effort")

    def test_input_td_with_extra_fields_flattened_into_yaml(self):
        """Test that TensorSemantics.extra fields are emitted as top-level YAML keys."""
        tensor = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "extra_meta_node",
            TensorSemantics(
                name="joint_pos",
                ref=tensor,
                kind=Kind.JOINT_POSITION,
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
        self.assertEqual(entry["kind"], "joint/position")
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

    def test_generated_yaml_includes_warp_metadata(self):
        """Test node parameters and system info include warp segment/version metadata."""
        tensor = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors("frequency_node", {"input": tensor})
        annotate.output_tensors("frequency_node", {"output": traced + 1.0})
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()

        parameters = config["models"]["frequency_node"]["parameters"]
        self.assertIn("warp_segments", parameters)
        self.assertIsInstance(parameters["warp_segments"], int)
        self.assertEqual(parameters["warp_segments"], 0)

        system_info = config["system information"]
        self.assertIn("warp version", system_info)
        self.assertEqual(system_info["warp version"], utils._get_warp_version())
        warp_version = system_info["warp version"]
        self.assertTrue(warp_version is None or isinstance(warp_version, str))
        if warp_version is not None:
            self.assertTrue(warp_version)

    def test_system_info_reports_null_warp_version_when_unavailable(self):
        """Warp version should be null in YAML when warp cannot be imported."""
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "warp":
                raise ImportError("warp not installed")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            self.assertIsNone(utils._get_warp_version())
            system_info = utils.get_system_info()["system information"]
            self.assertIn("warp version", system_info)
            self.assertIsNone(system_info["warp version"])

        dumped = yaml.safe_load(yaml.dump(system_info))
        self.assertIsNone(dumped["warp version"])

    def test_temporal_period_marker_appears_in_output_yaml(self):
        """Test a time axis emits temporal axis and period metadata."""
        tensor = torch.randn(2, 3)
        names = ["hip", "knee", "ankle"]

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors("chunk_node", {"input": tensor})

        annotate.output_tensors("chunk_node", [
            TensorSemantics(
                name="actions",
                ref=traced + 1.0,
                axes=[Axis(AxisKind.TIME, period_ms=100), Axis(names=names)],
            ),
            TensorSemantics(name="plain_output", ref=traced - 1.0),
        ])
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        _, outputs = self._get_node_io_from_yaml(config, "chunk_node")

        action_entry = self._find_io_by_name(outputs, "actions")
        plain_entry = self._find_io_by_name(outputs, "plain_output")

        self.assertEqual(action_entry["axes"], [
            {"kind": "time", "period_ms": 100}, {"names": names}])
        self.assertNotIn("temporal_period_ms", plain_entry)
        self.assertNotIn("axes", plain_entry)

    # =========================================================================
    # No metadata (baseline)
    # =========================================================================

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

        self.assertNotIn('kind', input_entry)
        self.assertNotIn('source', input_entry)
        self.assertNotIn('axes', input_entry)
        self.assertNotIn('kind', output_entry)
        self.assertNotIn('source', output_entry)
        self.assertNotIn('axes', output_entry)

    def test_tensor_semantics_rejects_temporal_period_ms_argument(self):
        """Test temporal period is only set through a time axis."""
        tensor = torch.randn(2, 3)

        with self.assertRaises(TypeError):
            TensorSemantics(
                name="actions",
                ref=tensor,
                axes=[None, Axis(names=["hip", "knee", "ankle"])],
                temporal_period_ms=100,
            )

    # =========================================================================
    # Error cases
    # =========================================================================

    def test_mixed_td_and_raw_raises(self):
        """Test that mixing TensorSemantics and raw tensors in a list raises TypeError."""
        t1 = torch.randn(1, 3)
        t2 = torch.randn(1, 3)

        leapp.start(name=self.TEST_GRAPH_NAME)
        with self.assertRaises(TypeError):
            annotate.input_tensors("fail_node", [
                t1,
                TensorSemantics(name="td", ref=t2, kind=Kind.JOINT_POSITION),
            ])
        leapp.stop()

    def test_duplicate_td_names_raises(self):
        """Test that two TensorSemantics with the same name raise an error."""
        t1 = torch.randn(1, 3)
        t2 = torch.randn(1, 3)

        leapp.start(name=self.TEST_GRAPH_NAME)
        with self.assertRaises(Exception):
            annotate.input_tensors("dup_node", [
                TensorSemantics(name="same_name", ref=t1),
                TensorSemantics(name="same_name", ref=t2),
            ])
        leapp.stop()

    # =========================================================================
    # Graph structure verification
    # =========================================================================

    def test_td_inputs_graph_structure(self):
        """Test that TensorSemantics inputs produce correct graph structure (node count, connections)."""
        pos = torch.randn(1, 4)
        vel = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced_pos, traced_vel = annotate.input_tensors("struct_node", [
            TensorSemantics(name="pos", ref=pos, kind=Kind.JOINT_POSITION),
            TensorSemantics(name="vel", ref=vel, kind=Kind.JOINT_VELOCITY),
        ])

        output = traced_pos + traced_vel
        annotate.output_tensors("struct_node", {"command": output})
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=2, outputs=1, internal_connections=0)

    # =========================================================================
    # Reentry with TensorSemantics
    # =========================================================================

    def test_semantic_input_reentry_does_not_crash(self):
        """Reentry with TensorSemantics inputs must not KeyError on semantic-only keys."""
        joint_pos = torch.randn(1, 6)

        leapp.start(name=self.TEST_GRAPH_NAME)

        for _ in range(2):
            traced = annotate.input_tensors("policy", [
                TensorSemantics(name="joint_pos", ref=joint_pos,
                                kind=Kind.JOINT_POSITION),
            ])
            out = traced * 2.0
            annotate.output_tensors("policy", {"cmd": out}, export_with="jit")

        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, _ = self._get_node_io_from_yaml(config, "policy")
        entry = self._find_io_by_name(inputs, "joint_pos")
        self.assertIsNotNone(entry)
        self.assertEqual(entry['kind'], "joint/position")

    def test_semantic_output_reentry_does_not_crash(self):
        """Reentry with TensorSemantics outputs must not KeyError on semantic-only keys."""
        tensor = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)

        for _ in range(2):
            traced = annotate.input_tensors("out_reentry", {"x": tensor})
            result = traced + 1.0
            annotate.output_tensors("out_reentry", [
                TensorSemantics(name="torques", ref=result,
                                kind=Kind.JOINT_EFFORT),
            ], export_with="jit")

        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        _, outputs = self._get_node_io_from_yaml(config, "out_reentry")
        entry = self._find_io_by_name(outputs, "torques")
        self.assertIsNotNone(entry)
        self.assertEqual(entry['kind'], "joint/effort")


if __name__ == '__main__':
    unittest.main(verbosity=2)


class TestStructuredSemantics(LEAPPFunctionalTestBase):
    def test_helpers_export_and_runtime(self):
        from leapp import Axis, AxisKind, Kind, frame_twist, image, joint_effort, joint_position

        q = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[3.0, 5.0]])
        velocity = torch.zeros(1, 1, 6)
        pixels = torch.zeros(1, 3, 2, 2, dtype=torch.uint8)
        joints = [Axis(AxisKind.ROBOT, names=["robot0"]),
                  Axis(AxisKind.ELEMENT, names=["shoulder", "elbow"])]
        leapp.start(name=self.TEST_GRAPH_NAME)
        tq, tt, tv, ti = leapp.annotate.input_tensors("policy", [
            joint_position("q", q, axes=joints),
            joint_position("target", target, axes=joints, is_setpoint=True),
            frame_twist("twist", velocity, reference="base", axes=[
                Axis(AxisKind.ROBOT), Axis(AxisKind.ELEMENT, names=["tool"]),
                Axis(AxisKind.COMPONENT, names=["vx", "vy", "vz", "wx", "wy", "wz"])]),
            image("camera", pixels, axes=[Axis(AxisKind.ROBOT), None, None, None]),
        ])
        command = tt - tq + tv.sum() + ti.float().sum()
        leapp.annotate.output_tensors("policy", joint_effort("effort", command, axes=joints), export_with="jit")
        leapp.stop()
        self.assertTrue(leapp.compile_graph(visualize=False))
        path = os.path.join(self.TEST_GRAPH_NAME, self.TEST_GRAPH_NAME + ".yaml")
        with open(path) as stream:
            config = yaml.safe_load(stream)
        inputs = config["models"]["policy"]["inputs"]
        self.assertEqual(inputs[1]["is_setpoint"], True)
        self.assertNotIn("is_setpoint", inputs[0])
        self.assertEqual(inputs[2]["expressed_in"], {"selector": "self"})
        self.assertEqual(inputs[2]["axes"][1], {"kind": "element", "names": ["tool"]})
        self.assertEqual(config["models"]["policy"]["outputs"][0]["kind"], Kind.JOINT_EFFORT.value)
        runtime = leapp.InferenceManager(path)
        result = runtime({"policy/q": q, "policy/target": target,
                          "policy/twist": velocity, "policy/camera": pixels})
        torch.testing.assert_close(result["policy/effort"], target - q)

    def test_setpoints_rejected_on_outputs_and_static_outputs(self):
        for static in (False, True):
            with self.subTest(static=static):
                leapp.start(name=self.TEST_GRAPH_NAME)
                value = leapp.annotate.input_tensors("policy", {"x": torch.zeros(2)})
                bad = leapp.joint_position("target", value + 1, is_setpoint=True)
                with self.assertRaisesRegex(ValueError, "only to inputs"):
                    if static:
                        leapp.annotate.output_tensors("policy", {"out": value + 1}, static_outputs=bad)
                    else:
                        leapp.annotate.output_tensors("policy", bad)
                leapp.stop()
