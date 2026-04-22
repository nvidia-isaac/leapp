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
import yaml
import torch
import leapp
from leapp import TensorSemantics
from leapp.leapp import _MANAGER as annotate
from leapp.utils.enums import InputKindEnum, OutputKindEnum
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
            TensorSemantics(name="joint_pos", ref=joint_pos, kind=InputKindEnum.JOINT_POSITION),
            TensorSemantics(name="joint_vel", ref=joint_vel, kind=InputKindEnum.JOINT_VELOCITY),
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
        self.assertEqual(pos_entry['kind'], "state/joint/position")
        self.assertEqual(vel_entry['kind'], "state/joint/velocity")
        # Output without semantics should have no kind
        self.assertNotIn('kind', cmd_entry)

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
        self.assertEqual(pos_entry['kind'], "state/joint/position")
        self.assertEqual(pos_entry['shape'], [1, 4])

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
        self.assertEqual(pos_entry['element_names'], [["x", "y", "z"]])

    def test_input_td_with_kind_and_element_names(self):
        """Test that both kind and element_names appear together in YAML."""
        tensor = torch.randn(1, 6)
        joint_names = ["hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"]

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "full_meta_node",
            TensorSemantics(name="joint_pos", ref=tensor,
                            kind=InputKindEnum.JOINT_POSITION,
                            element_names=joint_names),
        )

        output = traced * 0.5
        annotate.output_tensors("full_meta_node", {"out": output})
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, _ = self._get_node_io_from_yaml(config, "full_meta_node")

        entry = self._find_io_by_name(inputs, "joint_pos")
        self.assertIsNotNone(entry)
        self.assertEqual(entry['kind'], "state/joint/position")
        self.assertEqual(entry['element_names'], [joint_names])
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
                kind=InputKindEnum.BODY_ANGULAR_VELOCITY,
            ),
        )

        annotate.output_tensors("source_node", [
            TensorSemantics(
                name="filtered_imu",
                ref=traced * 0.5,
                kind=OutputKindEnum.BODY_ANGULAR_ACCELERATION,
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
        self.assertEqual(input_entry["kind"], "state/body/angular_velocity")
        self.assertEqual(output_entry["kind"], "target/body/angular_acceleration")
    
    def test_input_td_with_string_kind(self):
        """Test that a string kind appears in YAML."""
        tensor = torch.randn(1, 6)
        joint_names = ["hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"]

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "full_meta_node",
            TensorSemantics(name="joint_pos", ref=tensor,
                            kind="my/custom/kind",
                            element_names=joint_names),
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
        self.assertEqual(entry['element_names'], [joint_names])
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
            TensorSemantics(name="command", ref=command, kind=OutputKindEnum.JOINT_TORQUES),
        ])
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        _, outputs = self._get_node_io_from_yaml(config, "out_kind_node")

        cmd_entry = self._find_io_by_name(outputs, "command")
        self.assertIsNotNone(cmd_entry)
        self.assertEqual(cmd_entry['kind'], "target/joint/torques")

    def test_output_td_with_element_names(self):
        """Test that element_names from output TensorSemantics appears in YAML."""
        tensor = torch.randn(1, 3)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors("out_elem_node", {"input": tensor})

        result = traced + 1.0

        annotate.output_tensors("out_elem_node", [
            TensorSemantics(name="rgb", ref=result, element_names=["r", "g", "b"]),
        ])
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        _, outputs = self._get_node_io_from_yaml(config, "out_elem_node")

        rgb_entry = self._find_io_by_name(outputs, "rgb")
        self.assertIsNotNone(rgb_entry)
        self.assertEqual(rgb_entry['element_names'], [["r", "g", "b"]])

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
        self.assertEqual(static_entry['kind'], "target/joint/torques")

    # =========================================================================
    # Both inputs and outputs
    # =========================================================================

    def test_both_input_and_output_td(self):
        """Test TensorSemantics on both inputs and outputs in the same node."""
        pos = torch.randn(1, 6)
        vel = torch.randn(1, 6)

        leapp.start(name=self.TEST_GRAPH_NAME)
        traced_pos, traced_vel = annotate.input_tensors("both_node", [
            TensorSemantics(name="pos", ref=pos, kind=InputKindEnum.JOINT_POSITION),
            TensorSemantics(name="vel", ref=vel, kind=InputKindEnum.JOINT_VELOCITY),
        ])

        command = traced_pos + traced_vel

        annotate.output_tensors("both_node", [
            TensorSemantics(name="torques", ref=command, kind=OutputKindEnum.JOINT_TORQUES),
        ])
        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, outputs = self._get_node_io_from_yaml(config, "both_node")

        self.assertEqual(self._find_io_by_name(inputs, "pos")['kind'], "state/joint/position")
        self.assertEqual(self._find_io_by_name(inputs, "vel")['kind'], "state/joint/velocity")
        self.assertEqual(self._find_io_by_name(outputs, "torques")['kind'], "target/joint/torques")

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
        self.assertNotIn('element_names', input_entry)
        self.assertNotIn('kind', output_entry)
        self.assertNotIn('source', output_entry)
        self.assertNotIn('element_names', output_entry)

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
                TensorSemantics(name="td", ref=t2, kind=InputKindEnum.JOINT_POSITION),
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
            TensorSemantics(name="pos", ref=pos, kind=InputKindEnum.JOINT_POSITION),
            TensorSemantics(name="vel", ref=vel, kind=InputKindEnum.JOINT_VELOCITY),
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
                                kind=InputKindEnum.JOINT_POSITION),
            ])
            out = traced * 2.0
            annotate.output_tensors("policy", {"cmd": out}, export_with="jit")

        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, _ = self._get_node_io_from_yaml(config, "policy")
        entry = self._find_io_by_name(inputs, "joint_pos")
        self.assertIsNotNone(entry)
        self.assertEqual(entry['kind'], "state/joint/position")

    def test_semantic_output_reentry_does_not_crash(self):
        """Reentry with TensorSemantics outputs must not KeyError on semantic-only keys."""
        tensor = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)

        for _ in range(2):
            traced = annotate.input_tensors("out_reentry", {"x": tensor})
            result = traced + 1.0
            annotate.output_tensors("out_reentry", [
                TensorSemantics(name="torques", ref=result,
                                kind=OutputKindEnum.JOINT_TORQUES),
            ], export_with="jit")

        leapp.stop()
        leapp.compile_graph(visualize=False)

        config = self._load_yaml()
        _, outputs = self._get_node_io_from_yaml(config, "out_reentry")
        entry = self._find_io_by_name(outputs, "torques")
        self.assertIsNotNone(entry)
        self.assertEqual(entry['kind'], "target/joint/torques")


if __name__ == '__main__':
    unittest.main(verbosity=2)
