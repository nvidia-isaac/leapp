# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the public patch API through capture, export and independent replay."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_string_function_patch_example(tmp_path):
    """Cover patch precedence, reentry, registered buffers and no callback at replay.

    Raw UTF-8 units check filter semantics. This checks its integration with
    string capture, native tokenization and a serialized bundle after stop().
    Two regex registrations exercise dispatch, module buffers, a plain callable
    and static RegexFlag arguments together in one exported pipeline.
    """
    from leapp.text_runtime import resolve_library

    try:
        resolve_library("onnx")
        resolve_library("pt2")
    except FileNotFoundError as error:
        pytest.skip(str(error))
    from examples.string_function_patch import main

    artifact = main(str(tmp_path))
    # Start fresh and prohibit importing the application callback or Python
    # tokenizer. Only exported tensor operations and the native tokenizer run.
    script = """
import importlib.abc
import sys
class NoApplicationPython(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('examples', 'transformers', 'tokenizers'):
            raise AssertionError('Application Python imported during replay')
sys.meta_path.insert(0, NoApplicationPython())
import leapp
manager = leapp.InferenceManager(sys.argv[1])
try:
    result = manager.run_policy({'tokenizer/text': '123WORLD!!! HELLO456'})
    assert result['tokenizer/input_ids'].tolist() == [[3, 2]]
    assert result['tokenizer/attention_mask'].tolist() == [[1, 1]]
finally:
    manager.close()
"""
    subprocess.run(
        [sys.executable, "-c", script, str(artifact)],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2])),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
