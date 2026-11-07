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

    def test_fatal_raises_runtime_error_for_types_that_cannot_carry_the_message(
        self,
    ):
        """fatal() keeps the requested type only when the type carries the message."""

        class TwoArgumentError(Exception):
            def __init__(self, message, kind):
                super().__init__(message)

        class DropsMessageError(Exception):
            def __init__(self, message):
                super().__init__()

        class ReplacesMessageError(Exception):
            def __init__(self, message):
                super().__init__("something else")

        class FailingConstructorError(Exception):
            def __init__(self, message):
                raise ValueError("cannot build")

        class Incomparable:
            def __eq__(self, other):
                raise ValueError("cannot compare")

        class IncomparableArgumentError(Exception):
            def __init__(self, message):
                super().__init__(Incomparable(), message)

        cause = ValueError("original")
        message = "fatal 'test' message\nwith a second line"
        with tempfile.TemporaryDirectory() as tmpdir:
            _get_logger().configure(tmpdir, verbose=False)
            for error_type, expected in (
                (TwoArgumentError, RuntimeError),
                (DropsMessageError, RuntimeError),
                (ReplacesMessageError, RuntimeError),
                (FailingConstructorError, RuntimeError),
                (IncomparableArgumentError, RuntimeError),
                (KeyError, KeyError),
                (ValueError, ValueError),
            ):
                with self.subTest(error_type=error_type.__name__):
                    with self.assertLogs("leapp", level="ERROR") as logs:
                        with self.assertRaises(Exception) as caught:
                            _get_logger().fatal(
                                message, error_type=error_type, cause=cause
                            )
                    self.assertEqual(logs.output, [f"FATAL:leapp:{message}"])
                    self.assertIs(type(caught.exception), expected)
                    self.assertEqual(caught.exception.args, (message,))
                    self.assertIs(caught.exception.__cause__, cause)


class TestFatalLoggingInExportManager(LEAPPFunctionalTestBase):
    def test_mirror_leapp_tags_logs_mismatch_with_boundary_context(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            with self.assertLogs("leapp", level="ERROR") as logs:
                with self.assertRaisesRegex(
                    Exception, "source and target do not match"
                ):
                    annotate.mirror_leapp_tags(
                        torch.tensor([1.0]),
                        torch.tensor([2.0]),
                    )
        finally:
            leapp.stop()

        log_output = "\n".join(logs.output)
        self.assertIn("source and target do not match", log_output)


if __name__ == "__main__":
    unittest.main()
