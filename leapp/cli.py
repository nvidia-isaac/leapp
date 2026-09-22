#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""LEAPP command-line entry point.

TensorRT ONNX rewrite, catalog check, and pipeline merge live in the sibling
``isaac_deploy_trt`` package::

    pip install -e ./packages/isaac_deploy_trt
    isaac_deploy_trt check --onnx model.onnx --strict
    isaac_deploy_trt check --onnx-dir ./exported_graph --build --strict
    isaac_deploy_trt rewrite-dir ./exported_graph -o ./exported_graph_trt
    isaac_deploy_trt merge-pipeline exported.yaml -o pipeline.onnx --rewrite
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

_MOVED = """TensorRT ONNX tools moved to isaac_deploy_trt.

  pip install -e ./packages/isaac_deploy_trt
  isaac_deploy_trt check --onnx MODEL.onnx [--build] [--strict]
  isaac_deploy_trt check --onnx-dir EXPORT_DIR [--build] [--strict]
  isaac_deploy_trt rewrite-dir INPUT_DIR -o OUTPUT_DIR
  isaac_deploy_trt merge-pipeline EXPORTED.yaml -o pipeline.onnx [--rewrite]
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    print(_MOVED, file=sys.stderr)
    _ = argv
    return 2


if __name__ == "__main__":
    sys.exit(main())
