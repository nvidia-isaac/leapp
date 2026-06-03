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
import shutil
import tempfile
import unittest

import torch

import leapp
from leapp.leapp import _MANAGER as annotate
from .base import LEAPPFunctionalTestBase


class TestStartNamePathNormalization(LEAPPFunctionalTestBase):
    def setUp(self):
        super().setUp()
        self._tmp_root = tempfile.mkdtemp(prefix="leapp_start_path_test_")
        self._cwd_cleanup_dirs = []

    def tearDown(self):
        try:
            super().tearDown()
        finally:
            shutil.rmtree(self._tmp_root, ignore_errors=True)
            for d in self._cwd_cleanup_dirs:
                shutil.rmtree(d, ignore_errors=True)

    def _build_trivial_graph(self):
        @annotate.method(export_with="jit")
        def identity(inputA: torch.Tensor):
            return inputA + 1.0

        identity(torch.tensor([1.0, 2.0, 3.0]))
        leapp.stop()
        leapp.compile_graph(visualize=False, validate=False)

    def _track_cwd_dir(self, top_level_dir):
        self._cwd_cleanup_dirs.append(os.path.abspath(top_level_dir))

    def test_simple_name_preserves_legacy_layout(self):
        leapp.start(self.TEST_GRAPH_NAME)
        self.assertEqual(annotate.get_graph_name(), self.TEST_GRAPH_NAME)
        self.assertEqual(os.path.normpath(annotate.get_save_path()),
                         os.path.normpath(self.TEST_GRAPH_NAME))
        self._build_trivial_graph()
        self._assert_artifacts(self.TEST_GRAPH_NAME, self.TEST_GRAPH_NAME)

    def test_relative_path_uses_basename_as_graph_name(self):
        self._track_cwd_dir("foo")

        leapp.start("foo/bar")
        self.assertEqual(annotate.get_graph_name(), "bar")
        self.assertEqual(os.path.normpath(annotate.get_save_path()),
                         os.path.join("foo", "bar"))
        self._build_trivial_graph()
        self._assert_artifacts(os.path.join("foo", "bar"), "bar")

    def test_absolute_path_overrides_save_path(self):
        absolute_target = os.path.join(self._tmp_root, "exported_gr00t")

        leapp.start(absolute_target)
        self.assertEqual(annotate.get_graph_name(), "exported_gr00t")
        self.assertEqual(os.path.normpath(annotate.get_save_path()),
                         os.path.normpath(absolute_target))
        self._build_trivial_graph()
        self._assert_artifacts(absolute_target, "exported_gr00t")
        # Critically, the YAML must NOT land one dir above the models, which
        # was the original bug.
        bad_path = os.path.join(self._tmp_root, "exported_gr00t.yaml")
        self.assertFalse(os.path.exists(bad_path),
                         f"YAML must not be emitted to sibling-dir path {bad_path}")

    def test_trailing_slash_path_is_normalized(self):
        absolute_target = os.path.join(self._tmp_root, "exported_gr00t")
        name_with_slash = absolute_target + os.sep

        leapp.start(name_with_slash)
        self.assertEqual(annotate.get_graph_name(), "exported_gr00t")
        self.assertEqual(os.path.normpath(annotate.get_save_path()),
                         os.path.normpath(absolute_target))
        self._build_trivial_graph()
        self._assert_artifacts(absolute_target, "exported_gr00t")

    def test_save_path_combined_with_path_like_name(self):
        save_path = os.path.join(self._tmp_root, "outputs")
        leapp.start("nested/bar", save_path=save_path)
        expected_dir = os.path.join(save_path, "nested", "bar")
        self.assertEqual(annotate.get_graph_name(), "bar")
        self.assertEqual(os.path.normpath(annotate.get_save_path()),
                         os.path.normpath(expected_dir))
        self._build_trivial_graph()
        self._assert_artifacts(expected_dir, "bar")

    def test_empty_name_rejected(self):
        with self.assertRaises(ValueError):
            leapp.start("")

    def test_root_only_name_rejected(self):
        with self.assertRaises(ValueError):
            leapp.start(os.sep)

    def test_non_string_name_rejected(self):
        with self.assertRaises(TypeError):
            leapp.start(123)  # type: ignore[arg-type]


class TestCompileGraphLogging(LEAPPFunctionalTestBase):
    def test_compile_graph_logs_saved_model_locations(self):
        @annotate.method(export_with="jit")
        def identity(inputA: torch.Tensor):
            return inputA + 1.0

        leapp.start(self.TEST_GRAPH_NAME)
        identity(torch.tensor([1.0, 2.0, 3.0]))
        leapp.stop()

        expected_model_path = os.path.abspath(
            os.path.join(self.TEST_GRAPH_NAME, "identity.pt"))
        with self.assertLogs("leapp", level=25) as logs:
            leapp.compile_graph(visualize=False, validate=False)

        logged_output = "\n".join(logs.output)
        self.assertIn("Model artifacts saved:", logged_output)
        self.assertIn(f"- identity: {expected_model_path}", logged_output)


if __name__ == "__main__":
    unittest.main()
