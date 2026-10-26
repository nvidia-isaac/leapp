#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""List ONNX operators the TensorRT parser catalog marks unsupported.

Same as ``isaac_deploy_trt unsupported-ops``. No ONNX model is required.
The table is onnx-tensorrt operators.md, not a Triton or ONNX Runtime kernel list.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from isaac_deploy_trt.unsupported_cmd import main

if __name__ == "__main__":
    raise SystemExit(main())
