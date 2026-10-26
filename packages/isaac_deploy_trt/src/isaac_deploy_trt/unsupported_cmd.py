#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""List ONNX operators the TensorRT parser catalog marks unsupported.

Kept free of ``onnx`` imports so it runs without the ONNX package installed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaac_deploy_trt.tensorrt_onnx_ops import (
    format_unsupported_onnx_op_report,
    load_unsupported_onnx_op_report,
    unsupported_onnx_op_report,
)


def add_unsupported_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tensorrt",
        default=None,
        help="TensorRT version for the baked-in parser table (default: 11.2)",
    )
    parser.add_argument(
        "--cuda",
        default=None,
        help="CUDA version used to pick a baked-in catalog when --tensorrt is omitted",
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="Platform catalog JSON (tensorrt_catalogs/...json) instead of the baked-in table",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON",
    )


def run_unsupported(args: argparse.Namespace) -> int:
    if args.catalog:
        path = os.path.abspath(os.path.expanduser(args.catalog))
        if not os.path.isfile(path):
            print(f"error: catalog not found: {path}", file=sys.stderr)
            return 2
        report = load_unsupported_onnx_op_report(path)
    else:
        try:
            report = unsupported_onnx_op_report(args.tensorrt, cuda=args.cuda)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_unsupported_onnx_op_report(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List ONNX operators the TensorRT parser catalog marks unsupported",
    )
    add_unsupported_arguments(parser)
    return run_unsupported(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
