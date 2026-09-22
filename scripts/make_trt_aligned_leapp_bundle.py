#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Write a LEAPP export dir with TensorRT-legal ONNX under the original names.

Isaac Deploy / Triton look up ``preprocess_video.onnx``, ``backbone.onnx``,
etc. next to the YAML. This copies that layout and only rewrites graphs that
need GraphSurgeon (UINT8 Cast, Resize antialias, rank-9 Reshape). Checksums
in the YAML are updated. Graph I/O names/shapes/dtypes are unchanged.

Example::

    python scripts/make_trt_aligned_leapp_bundle.py \\
        ../gr00t-leapp-export/apple_pnp_onnx \\
        -o ../gr00t-leapp-export/apple_pnp_onnx_trt
"""

from __future__ import annotations

import argparse
import sys

from leapp.backends.tensorrt_leapp_bundle import align_leapp_export_for_tensorrt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a LEAPP export dir to TensorRT-legal ONNX with aligned names "
            "and YAML checksums"
        ),
    )
    parser.add_argument("export_dir", help="Source LEAPP export directory")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Destination directory (must not be the source)",
    )
    args = parser.parse_args(argv)
    result = align_leapp_export_for_tensorrt(args.export_dir, args.output)
    print(f"Wrote aligned TensorRT LEAPP bundle to {result.output_dir}")
    for model in result.models:
        action = "rewrote" if model.rewritten else "copied"
        extra = f" ({len(model.rewritten_nodes)} nodes)" if model.rewritten else ""
        print(f"  {action} {model.filename}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
