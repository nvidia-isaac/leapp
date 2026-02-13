#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from leapp import annotate, TensorSemantics
from leapp.utils.enums import inputKindEnum, outputKindEnum
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

        annotate.start(name=self.TEST_GRAPH_NAME)
        traced_pos, traced_vel = annotate.input_tensors("policy", [
            TensorSemantics(name="joint_pos", ref=joint_pos, kind=inputKindEnum.JOINT_POSITION),
            TensorSemantics(name="joint_vel", ref=joint_vel, kind=inputKindEnum.JOINT_VELOCITY),
        ])

        output = traced_pos + traced_vel
        annotate.output_tensors("policy", {"command": output})
        annotate.stop()
        annotate.compile_graph(visualize=False)

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

        annotate.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "single_node",
            TensorSemantics(name="pos", ref=tensor, kind=inputKindEnum.JOINT_POSITION),
        )

        output = traced * 2.0
        annotate.output_tensors("single_node", {"out": output})
        annotate.stop()
        annotate.compile_graph(visualize=False)

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

        annotate.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "elem_node",
            TensorSemantics(name="position", ref=tensor, element_names=names),
        )

        output = traced + 1.0
        annotate.output_tensors("elem_node", {"out": output})
        annotate.stop()
        annotate.compile_graph(visualize=False)

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

        annotate.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "full_meta_node",
            TensorSemantics(name="joint_pos", ref=tensor,
                            kind=inputKindEnum.JOINT_POSITION,
                            element_names=joint_names),
        )

        output = traced * 0.5
        annotate.output_tensors("full_meta_node", {"out": output})
        annotate.stop()
        annotate.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, _ = self._get_node_io_from_yaml(config, "full_meta_node")

        entry = self._find_io_by_name(inputs, "joint_pos")
        self.assertIsNotNone(entry)
        self.assertEqual(entry['kind'], "state/joint/position")
        self.assertEqual(entry['element_names'], [joint_names])
        self.assertEqual(entry['dtype'], "float32")
        self.assertEqual(entry['shape'], [1, 6])
    
    def test_input_td_with_string_kind(self):
        """Test that a string kind appears in YAML."""
        tensor = torch.randn(1, 6)
        joint_names = ["hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"]

        annotate.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors(
            "full_meta_node",
            TensorSemantics(name="joint_pos", ref=tensor,
                            kind="my/custom/kind",
                            element_names=joint_names),
        )

        output = traced * 0.5
        annotate.output_tensors("full_meta_node", {"out": output})
        annotate.stop()
        annotate.compile_graph(visualize=False)

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

        annotate.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors("out_kind_node", {"pos": tensor})

        command = traced * 2.0

        annotate.output_tensors("out_kind_node", [
            TensorSemantics(name="command", ref=command, kind=outputKindEnum.JOINT_TORQUES),
        ])
        annotate.stop()
        annotate.compile_graph(visualize=False)

        config = self._load_yaml()
        _, outputs = self._get_node_io_from_yaml(config, "out_kind_node")

        cmd_entry = self._find_io_by_name(outputs, "command")
        self.assertIsNotNone(cmd_entry)
        self.assertEqual(cmd_entry['kind'], "target/joint/torques")

    def test_output_td_with_element_names(self):
        """Test that element_names from output TensorSemantics appears in YAML."""
        tensor = torch.randn(1, 3)

        annotate.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors("out_elem_node", {"input": tensor})

        result = traced + 1.0

        annotate.output_tensors("out_elem_node", [
            TensorSemantics(name="rgb", ref=result, element_names=["r", "g", "b"]),
        ])
        annotate.stop()
        annotate.compile_graph(visualize=False)

        config = self._load_yaml()
        _, outputs = self._get_node_io_from_yaml(config, "out_elem_node")

        rgb_entry = self._find_io_by_name(outputs, "rgb")
        self.assertIsNotNone(rgb_entry)
        self.assertEqual(rgb_entry['element_names'], [["r", "g", "b"]])

    # =========================================================================
    # Both inputs and outputs
    # =========================================================================

    def test_both_input_and_output_td(self):
        """Test TensorSemantics on both inputs and outputs in the same node."""
        pos = torch.randn(1, 6)
        vel = torch.randn(1, 6)

        annotate.start(name=self.TEST_GRAPH_NAME)
        traced_pos, traced_vel = annotate.input_tensors("both_node", [
            TensorSemantics(name="pos", ref=pos, kind=inputKindEnum.JOINT_POSITION),
            TensorSemantics(name="vel", ref=vel, kind=inputKindEnum.JOINT_VELOCITY),
        ])

        command = traced_pos + traced_vel

        annotate.output_tensors("both_node", [
            TensorSemantics(name="torques", ref=command, kind=outputKindEnum.JOINT_TORQUES),
        ])
        annotate.stop()
        annotate.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, outputs = self._get_node_io_from_yaml(config, "both_node")

        self.assertEqual(self._find_io_by_name(inputs, "pos")['kind'], "state/joint/position")
        self.assertEqual(self._find_io_by_name(inputs, "vel")['kind'], "state/joint/velocity")
        self.assertEqual(self._find_io_by_name(outputs, "torques")['kind'], "target/joint/torques")

    # =========================================================================
    # No metadata (baseline)
    # =========================================================================

    def test_no_metadata_no_extra_yaml_fields(self):
        """Test that tensors without TensorSemantics have no kind or element_names in YAML."""
        tensor = torch.randn(1, 4)

        annotate.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors("plain_node", {"input": tensor})

        output = traced + 1.0
        annotate.output_tensors("plain_node", {"output": output})
        annotate.stop()
        annotate.compile_graph(visualize=False)

        config = self._load_yaml()
        inputs, outputs = self._get_node_io_from_yaml(config, "plain_node")

        input_entry = self._find_io_by_name(inputs, "input")
        output_entry = self._find_io_by_name(outputs, "output")

        self.assertNotIn('kind', input_entry)
        self.assertNotIn('element_names', input_entry)
        self.assertNotIn('kind', output_entry)
        self.assertNotIn('element_names', output_entry)

    # =========================================================================
    # Error cases
    # =========================================================================

    def test_mixed_td_and_raw_raises(self):
        """Test that mixing TensorSemantics and raw tensors in a list raises TypeError."""
        t1 = torch.randn(1, 3)
        t2 = torch.randn(1, 3)

        annotate.start(name=self.TEST_GRAPH_NAME)
        with self.assertRaises(TypeError):
            annotate.input_tensors("fail_node", [
                t1,
                TensorSemantics(name="td", ref=t2, kind=inputKindEnum.JOINT_POSITION),
            ])
        annotate.stop()

    def test_duplicate_td_names_raises(self):
        """Test that two TensorSemantics with the same name raise an error."""
        t1 = torch.randn(1, 3)
        t2 = torch.randn(1, 3)

        annotate.start(name=self.TEST_GRAPH_NAME)
        with self.assertRaises(Exception):
            annotate.input_tensors("dup_node", [
                TensorSemantics(name="same_name", ref=t1),
                TensorSemantics(name="same_name", ref=t2),
            ])
        annotate.stop()

    # =========================================================================
    # Graph structure verification
    # =========================================================================

    def test_td_inputs_graph_structure(self):
        """Test that TensorSemantics inputs produce correct graph structure (node count, connections)."""
        pos = torch.randn(1, 4)
        vel = torch.randn(1, 4)

        annotate.start(name=self.TEST_GRAPH_NAME)
        traced_pos, traced_vel = annotate.input_tensors("struct_node", [
            TensorSemantics(name="pos", ref=pos, kind=inputKindEnum.JOINT_POSITION),
            TensorSemantics(name="vel", ref=vel, kind=inputKindEnum.JOINT_VELOCITY),
        ])

        output = traced_pos + traced_vel
        annotate.output_tensors("struct_node", {"command": output})
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=2, outputs=1, internal_connections=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
