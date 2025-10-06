#!/usr/bin/env python3

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

import unittest
import subprocess
import os
import sys
import shutil
from pathlib import Path


class BaseExampleTest(unittest.TestCase):
    """Base class for example script tests with common functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Get the repository root directory
        cls.repo_root = Path(__file__).parent.parent.parent.absolute()
        cls.examples_dir = cls.repo_root / "examples"

        # Ensure examples directory exists
        if not cls.examples_dir.exists():
            raise FileNotFoundError(
                f"Examples directory not found: {cls.examples_dir}")

        # Original working directory
        cls.original_cwd = os.getcwd()

    def setUp(self):
        """Set up for each test - change to repo root directory."""
        os.chdir(self.repo_root)

    def tearDown(self):
        """Clean up after each test."""
        os.chdir(self.original_cwd)

    def _run_example_script(self, script_name, expected_output_dir):
        """
        Run an example script and verify it completes successfully.

        Args:
            script_name: Name of the script file (e.g., 'wbc_plain.py')
            expected_output_dir: Expected output directory name (e.g., 'sample_wbc_graph')

        Returns:
            tuple: (subprocess_result, output_dir_path)
        """
        script_path = self.examples_dir / script_name
        output_dir_path = self.repo_root / expected_output_dir

        # Ensure script exists
        if not script_path.exists():
            raise FileNotFoundError(f"Example script not found: {script_path}")

        # Clean up any existing output directory
        if output_dir_path.exists():
            shutil.rmtree(output_dir_path)

        # Run the script
        cmd = [sys.executable, str(script_path)]
        result = subprocess.run(
            cmd,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        return result, output_dir_path

    def _verify_output_files(self, output_dir_path, expected_files):
        """
        Verify that expected output files exist in the output directory.

        Args:
            output_dir_path: Path to the output directory
            expected_files: List of expected file names or patterns
        """
        # Check that output directory was created
        self.assertTrue(
            output_dir_path.exists(),
            f"Output directory was not created: {output_dir_path}"
        )

        # Check that it's actually a directory
        self.assertTrue(
            output_dir_path.is_dir(),
            f"Output path is not a directory: {output_dir_path}"
        )

        # Get list of actual files
        actual_files = [f.name for f in output_dir_path.iterdir()
                        if f.is_file()]

        # Check each expected file
        for expected_file in expected_files:
            self.assertIn(
                expected_file,
                actual_files,
                f"Expected file '{expected_file}' not found in {output_dir_path}. "
                f"Actual files: {actual_files}"
            )

    def _run_and_verify_example(self, script_name, output_dir_name, expected_files, test_description):
        """
        Complete workflow to run an example and verify its outputs.

        Args:
            script_name: Name of the script file
            output_dir_name: Expected output directory name  
            expected_files: List of expected output files
            test_description: Description for logging
        """
        print(f"\n=== {test_description} ===")

        # Run the example
        result, output_dir = self._run_example_script(
            script_name, output_dir_name)

        # Check that script ran successfully
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")

        self.assertEqual(
            result.returncode, 0,
            f"{script_name} failed with return code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        # Verify expected output files exist
        self._verify_output_files(output_dir, expected_files)

        print(f"✓ {script_name} completed successfully")
        print(
            f"✓ Generated {len(expected_files)} expected files in {output_dir}")
