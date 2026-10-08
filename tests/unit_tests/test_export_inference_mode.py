#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import tempfile
import unittest
from unittest import mock

import torch

from leapp.backends.export_backend import prepare_tensors_for_export
from leapp.leapp_graph.leapp_node import LeappNode
from leapp.utils.logging import _get_logger


class TestExportInferenceMode(unittest.TestCase):
    def test_prepare_inputs_escapes_inference_mode(self):
        for ambient_mode in (False, True):
            for original_clone in (False, True):
                with self.subTest(ambient_mode=ambient_mode, original_clone=original_clone):
                    with torch.inference_mode():
                        tensor = torch.tensor([1., 2.])
                    if original_clone:
                        tensor.original_clone = tensor.clone
                    marker = object()
                    with torch.inference_mode(ambient_mode):
                        prepared, unchanged = prepare_tensors_for_export((tensor, marker))
                        self.assertFalse(prepared.is_inference())
                        self.assertEqual(torch.is_inference_mode_enabled(), ambient_mode)
                    self.assertTrue(tensor.is_inference())
                    self.assertNotEqual(prepared.data_ptr(), tensor.data_ptr())
                    torch.testing.assert_close(prepared, tensor)
                    self.assertIs(unchanged, marker)

    def test_compile_reports_inference_tensor_errors(self):
        messages = (
            "Inference tensors cannot be saved for backward.",
            "Setting requires_grad=True on inference tensor outside InferenceMode is not allowed.",
            "unrelated export failure",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            _get_logger().configure(tmpdir, verbose=False)
            for backend, target in (("onnx-dynamo", "torch.onnx.export"),
                                    ("pt2", "torch.export.export")):
                node = LeappNode("inference_test")
                node.setup_backend(backend, {})
                node.m = torch.nn.Identity()
                for message in messages:
                    for chain in (None, "__cause__", "__context__"):
                        with self.subTest(backend=backend, message=message, chain=chain):
                            original = RuntimeError(message)
                            error = original
                            if chain:
                                error = RuntimeError("export wrapper failed")
                                setattr(error, chain, original)
                            with mock.patch(target, side_effect=error) as export:
                                with self.assertLogs("leapp", level="ERROR") as logs:
                                    with self.assertRaises(RuntimeError) as caught:
                                        node.compile_model()
                            export.assert_called_once()
                            self.assertIs(caught.exception.__cause__, error)
                            diagnostic = str(caught.exception)
                            self.assertIn(node.name, diagnostic)
                            if message != "unrelated export failure":
                                self.assertIn(node.backend, diagnostic)
                                self.assertIn("torch.inference_mode()", diagnostic)
                                self.assertIn("torch.no_grad()", diagnostic)
                                self.assertIn("Recreate", diagnostic)
                                self.assertIn("FATAL:leapp:", logs.output[0])
                                self.assertIn(diagnostic, logs.output[0])
                            else:
                                self.assertEqual(diagnostic, f"Error compiling model {node.name}: {error}")
                                self.assertNotIn("torch.inference_mode()", logs.output[0])
