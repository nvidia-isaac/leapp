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
from leapp.leapp import _MANAGER as annotate
from leapp.utils import utils
from .base import LEAPPFunctionalTestBase


class TestConfigGeneration(LEAPPFunctionalTestBase):
    """Generated node parameters and environment metadata."""

    def _load_yaml(self):
        """Load the generated YAML config file."""
        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        self.assertTrue(os.path.exists(yaml_path), f"YAML file not found: {yaml_path}")
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
