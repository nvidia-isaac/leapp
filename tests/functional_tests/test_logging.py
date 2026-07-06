#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile
import unittest

import torch
import leapp
from leapp.leapp import _MANAGER as annotate
from leapp.utils.logging import _get_logger

from .base import LEAPPFunctionalTestBase


class TestFatalLogging(unittest.TestCase):
    def test_fatal_logs_and_raises_requested_error_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _get_logger().configure(tmpdir, verbose=False)

            with self.assertLogs("leapp", level="ERROR") as logs:
                with self.assertRaisesRegex(ValueError, "fatal test message"):
                    _get_logger().fatal(
                        "fatal test message",
                        error_type=ValueError,
                    )

            self.assertEqual(logs.output, ["FATAL:leapp:fatal test message"])

            with self.assertRaisesRegex(RuntimeError, "fatal file message"):
                _get_logger().fatal("fatal file message")
            for handler in _get_logger().logger.handlers:
                handler.flush()
            with open(os.path.join(tmpdir, "log.txt"), "r", encoding="utf-8") as f:
                self.assertIn("[FATAL]: fatal file message", f.read())


class TestFatalLoggingInExportManager(LEAPPFunctionalTestBase):
    def test_mirror_leapp_tags_logs_mismatch_with_boundary_context(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            with self.assertLogs("leapp", level="ERROR") as logs:
                with self.assertRaisesRegex(Exception, "source and target do not match"):
                    annotate.mirror_leapp_tags(
                        torch.tensor([1.0]),
                        torch.tensor([2.0]),
                    )
        finally:
            leapp.stop()

        log_output = "\n".join(logs.output)
        self.assertIn("unexpected error mirroring LEAPP tags", log_output)
        self.assertIn("source and target do not match", log_output)


if __name__ == "__main__":
    unittest.main()
