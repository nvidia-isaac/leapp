#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""CLI helpers for generating TensorRT platform catalogs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaac_deploy_trt.headers import default_include_dir
from isaac_deploy_trt.platform_catalog import (
    DEFAULT_OPERATORS_MD_URL,
    DEFAULT_PLATFORMS,
    TARGET_CUDA,
    generate,
    resolve_platform_name,
)


def add_generate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--output",
        default="tensorrt_catalogs",
        help="Directory for per-platform JSON/Markdown catalogs",
    )
    parser.add_argument(
        "--platforms",
        default=",".join(DEFAULT_PLATFORMS),
        help="Comma-separated subset: noble,noble-fastos,noble-jetpack "
        "(SKU aliases: x86,orin,thor,dgx)",
    )
    parser.add_argument(
        "--operators-md",
        default=None,
        help="Local onnx-tensorrt docs/operators.md (skip network fetch)",
    )
    parser.add_argument(
        "--operators-url",
        default=DEFAULT_OPERATORS_MD_URL,
        help="operators.md URL used when --operators-md is omitted",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        default="json,md",
        help="Output formats: json,md",
    )
    parser.add_argument(
        "--write-python",
        default=None,
        help="Also write an unfiltered tensorrt_onnx_ops.py catalog",
    )
    parser.add_argument(
        "--include-dir",
        default=None,
        help=(
            "TensorRT include directory with NvInfer.h / NvInferRuntimeBase.h "
            "(or set TENSORRT_INCLUDE_DIR when --releases is omitted)"
        ),
    )
    parser.add_argument(
        "--fetch-headers",
        action="store_true",
        help="Download NvInfer*.h from NVIDIA/TensorRT GitHub when --include-dir is omitted",
    )
    parser.add_argument(
        "--releases",
        default=None,
        help=(
            "Comma-separated NVIDIA/TensorRT GitHub tags (v11.2,v10.16). "
            "Identical parsed tables share one database."
        ),
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print supported/unsupported counts to stdout",
    )


def run_generate(args: argparse.Namespace) -> int:
    platforms = [
        resolve_platform_name(name)
        for name in args.platforms.split(",")
        if name.strip()
    ]
    formats = [name.strip().lower() for name in args.formats.split(",") if name.strip()]
    releases = []
    if args.releases:
        releases = [name.strip() for name in args.releases.split(",") if name.strip()]
    include_dir = args.include_dir
    if include_dir is None and not releases:
        include_dir = default_include_dir()
    try:
        result = generate(
            platforms=platforms,
            operators_md=args.operators_md,
            operators_url=args.operators_url,
            output_dir=Path(args.output),
            formats=formats,
            write_python=Path(args.write_python) if args.write_python else None,
            include_dir=include_dir,
            fetch_headers=args.fetch_headers or bool(releases),
            releases=releases,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.print_summary:
        print(json.dumps(result["doc"], indent=2))
        for database in result.get("databases") or []:
            print(
                f"database {database['id']}: "
                f"supported_releases={database['supported_releases']}"
            )
        for catalog in result["catalogs"]:
            print(
                f"{catalog['platform']}: "
                f"{len(catalog['onnx_supported'])} supported, "
                f"{len(catalog['onnx_unsupported'])} unsupported ONNX ops, "
                f"DLA={catalog['dla']}"
            )
        for path in result["written"]:
            print(path)
        if result["python"]:
            print(result["python"])
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Generate TensorRT supported-layer catalogs for CUDA {TARGET_CUDA} "
            f"({', '.join(DEFAULT_PLATFORMS)})"
        ),
    )
    add_generate_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_generate(parse_args(argv))
