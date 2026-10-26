#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""CLI: optimize / check ONNX models for TensorRT."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from isaac_deploy_trt.catalog import build_catalog, write_catalog
from isaac_deploy_trt.generate_cmd import add_generate_arguments, run_generate
from isaac_deploy_trt.headers import default_include_dir
from isaac_deploy_trt.pipeline_merge import (
    merge_leapp_onnx_pipeline,
    save_merged_pipeline,
)
from isaac_deploy_trt.prepare import rewrite_directory, rewrite_onnx_file
from isaac_deploy_trt.minimal_layers import format_minimal_layers, minimal_supported_layers
from isaac_deploy_trt.unsupported_cmd import add_unsupported_arguments, run_unsupported
from isaac_deploy_trt.validate import validate_onnx_for_tensorrt


def _add_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")


def _add_target_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tensorrt",
        default=None,
        help=(
            "TensorRT release for the operator catalog (for example 10.16 or 11.2). "
            "Unsupported-op optimizations follow that catalog. "
            "Default: the minimal catalog, operators unsupported on every checked-in TensorRT release."
        ),
    )
    parser.add_argument(
        "--platform",
        default=None,
        help=(
            "Platform overlay for a --tensorrt release: "
            "x86, orin, thor, or dgx. Default with --tensorrt: x86. "
            "Ignored when --tensorrt is omitted."
        ),
    )


def _optimize_kwargs(args: argparse.Namespace) -> dict:
    return {"tensorrt": args.tensorrt, "platform": args.platform}


def _cmd_optimize(args: argparse.Namespace) -> int:
    try:
        result = rewrite_onnx_file(args.onnx, args.output, **_optimize_kwargs(args))
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    action = "optimized" if result.rewritten else "copied"
    extra = f" ({len(result.rewritten_nodes)} nodes)" if result.rewritten else ""
    suffix = f"  [{result.catalog_path}]" if result.catalog_path else ""
    print(f"{action} {result.filename}{extra} -> {result.output_path}{suffix}")
    return 0


def _cmd_optimize_dir(args: argparse.Namespace) -> int:
    try:
        results = rewrite_directory(args.input_dir, args.output, **_optimize_kwargs(args))
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(results)} ONNX file(s) to {os.path.abspath(args.output)}")
    for item in results:
        action = "optimized" if item.rewritten else "copied"
        extra = f" ({len(item.rewritten_nodes)} nodes)" if item.rewritten else ""
        suffix = f"  [{item.catalog_path}]" if item.catalog_path else ""
        print(f"  {action} {item.filename}{extra}{suffix}")
    return 0


def _cmd_merge_pipeline(args: argparse.Namespace) -> int:
    yaml_path = os.path.abspath(os.path.expanduser(args.yaml_path))
    if not os.path.isfile(yaml_path):
        print(f"error: YAML not found: {yaml_path}", file=sys.stderr)
        return 2
    try:
        result = merge_leapp_onnx_pipeline(
            yaml_path,
            model_dir=args.model_dir,
            optimize=args.optimize,
        )
        out_yaml = save_merged_pipeline(
            result,
            os.path.expanduser(args.output),
            yaml_path=args.yaml_out,
        )
    except Exception as exc:
        logging.getLogger(__name__).error("Failed to merge LEAPP pipeline: %s", exc)
        return 1
    if result.rewritten_models:
        print(f"optimized before merge: {', '.join(result.rewritten_models)}")
    print(f"wrote {os.path.abspath(os.path.expanduser(args.output))}")
    print(f"wrote {out_yaml}")
    print(f"TensorRTNode inputs: {result.input_tensor_names}")
    print(f"TensorRTNode outputs: {result.output_tensor_names}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    if args.catalog and (args.tensorrt or args.platform or args.cuda):
        print(
            "error: --catalog replaces --tensorrt, --platform, and --cuda",
            file=sys.stderr,
        )
        return 2
    paths = [args.onnx] if args.onnx else []
    if args.onnx_dir:
        for name in sorted(os.listdir(args.onnx_dir)):
            if name.endswith(".onnx"):
                paths.append(os.path.join(args.onnx_dir, name))
    failed = 0
    for path in paths:
        try:
            result = validate_onnx_for_tensorrt(
                path,
                build=args.build,
                strict=False,
                tensorrt=args.tensorrt,
                cuda=args.cuda,
                platform=args.platform,
                catalog=args.catalog,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        status = "ok" if result.ok else "FAIL"
        suffix = f"  [{result.catalog_path}]" if result.catalog_path else ""
        print(f"{status}  {path}{suffix}")
        if not result.ok:
            failed += 1
            for err in result.errors:
                print(f"       {err}")
    if args.strict and failed:
        return 1
    return 0


def _cmd_minimal_layers(args: argparse.Namespace) -> int:
    try:
        report = minimal_supported_layers()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.output:
        path = os.path.abspath(os.path.expanduser(args.output))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"wrote {path}")
        print(f"  layers: {report['layer_count']}")
        print(f"  excluded: {len(report['excluded'])}")
        return 0
    print(format_minimal_layers(report))
    return 0


def _cmd_catalog(args: argparse.Namespace) -> int:
    releases = []
    if args.releases:
        releases = [name.strip() for name in args.releases.split(",") if name.strip()]
    include_dir = args.include_dir
    if include_dir is None and not releases:
        include_dir = default_include_dir()
    if not include_dir and not args.fetch_headers and not args.operators_md and not releases:
        print(
            "error: pass --operators-md and/or --include-dir / --fetch-headers / --releases",
            file=sys.stderr,
        )
        return 1
    catalog = build_catalog(
        operators_md=args.operators_md,
        include_dir=include_dir,
        fetch_headers=args.fetch_headers or bool(releases),
        releases=releases or None,
    )
    path = write_catalog(catalog, os.path.abspath(args.output))
    nvinfer = catalog.get("nvinfer") or {}
    ops = catalog.get("onnx_operators") or {}
    print(f"wrote {path}")
    if catalog.get("supported_releases"):
        print(f"  supported_releases: {catalog['supported_releases']}")
    if catalog.get("databases"):
        for database in catalog["databases"]:
            print(
                f"  database supported_releases={database.get('supported_releases')}"
            )
    if nvinfer:
        print(
            f"  headers: {len(nvinfer.get('data_types') or [])} DataType, "
            f"{len(nvinfer.get('layer_type_names') or [])} LayerType, "
            f"strongly_typed={nvinfer.get('strongly_typed')}"
        )
    if ops:
        print(f"  onnx ops: {len(ops)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="isaac_deploy_trt",
        description=(
            "GraphSurgeon ONNX optimizations for TensorRT: UINT8 intermediate Casts, "
            "Resize antialias, Reshape rank > 8. Graph I/O names are preserved. "
            "Merge a LEAPP YAML DAG into one ONNX graph for TensorRTNode. "
            "Also join operators.md with NvInfer.h DataType/LayerType catalogs, "
            "and list ONNX operators that catalog marks unsupported."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    optimize = sub.add_parser("optimize", help="Optimize one ONNX file")
    optimize.add_argument("onnx", help="Input .onnx path")
    optimize.add_argument("-o", "--output", required=True, help="Output .onnx path")
    _add_target_arguments(optimize)
    optimize.set_defaults(handler=_cmd_optimize)

    optimize_dir = sub.add_parser(
        "optimize-dir", help="Optimize every .onnx in a directory"
    )
    optimize_dir.add_argument("input_dir", help="Directory of .onnx files")
    optimize_dir.add_argument("-o", "--output", required=True, help="Output directory")
    _add_target_arguments(optimize_dir)
    optimize_dir.set_defaults(handler=_cmd_optimize_dir)

    merge = sub.add_parser(
        "merge-pipeline",
        help="Merge a LEAPP YAML pipeline into one ONNX graph for TensorRTNode",
    )
    merge.add_argument(
        "yaml_path",
        help="LEAPP export YAML (models + pipeline.data_flow)",
    )
    merge.add_argument(
        "-o",
        "--output",
        required=True,
        help="Destination .onnx path (writes sibling .yaml unless --yaml-out)",
    )
    merge.add_argument(
        "--model-dir",
        default=None,
        help="Directory containing per-node ONNX files (default: YAML directory)",
    )
    merge.add_argument(
        "--yaml-out",
        default=None,
        help="Optional path for the single-model YAML / TensorRTNode name lists",
    )
    merge.add_argument(
        "--optimize",
        action="store_true",
        help=(
            "Apply TensorRT GraphSurgeon optimizations to each subgraph before merge. "
            "Uses the minimal catalog."
        ),
    )
    merge.set_defaults(handler=_cmd_merge_pipeline)

    check = sub.add_parser("check", help="Catalog-check ONNX vs TensorRT ops.md")
    src = check.add_mutually_exclusive_group(required=True)
    src.add_argument("--onnx", help="Single .onnx file")
    src.add_argument("--onnx-dir", help="Directory of .onnx files")
    check.add_argument("--build", action="store_true", help="Also TensorRT parse/build")
    check.add_argument("--strict", action="store_true", help="Exit 1 if any check fails")
    check.add_argument(
        "--tensorrt",
        default=None,
        help=(
            "TensorRT release for the operator catalog (for example 10.16 or 11.2). "
            "Default: the minimal catalog shared by every checked-in release. "
            "Other releases load tensorrt_catalogs."
        ),
    )
    check.add_argument(
        "--platform",
        default=None,
        help=(
            "Platform overlay for a --tensorrt release: "
            "x86, orin, thor, or dgx. Default with --tensorrt: x86. "
            "Ignored when --tensorrt is omitted."
        ),
    )
    check.add_argument(
        "--cuda",
        default=None,
        help="CUDA version used to pick a catalog (for example 13.2).",
    )
    check.add_argument(
        "--catalog",
        default=None,
        help="Platform catalog JSON. Replaces --tensorrt, --platform, and --cuda.",
    )
    check.set_defaults(handler=_cmd_check)

    unsupported = sub.add_parser(
        "unsupported-ops",
        help="List ONNX operators the TensorRT parser catalog marks unsupported",
    )
    add_unsupported_arguments(unsupported)
    unsupported.set_defaults(handler=run_unsupported)

    minimal = sub.add_parser(
        "minimal-layers",
        help="List native layers supported by every checked-in TensorRT catalog",
    )
    minimal.add_argument(
        "-o",
        "--output",
        default=None,
        help="Write the minimal layer catalog JSON to this path",
    )
    minimal.set_defaults(handler=_cmd_minimal_layers)

    catalog = sub.add_parser(
        "catalog",
        help="Join operators.md with TensorRT header enumerations",
    )
    catalog.add_argument(
        "--operators-md",
        help="Local onnx-tensorrt docs/operators.md",
    )
    catalog.add_argument(
        "--include-dir",
        default=None,
        help="TensorRT include dir (NvInfer.h); or set TENSORRT_INCLUDE_DIR when --releases is omitted",
    )
    catalog.add_argument(
        "--fetch-headers",
        action="store_true",
        help="Download NvInfer*.h from NVIDIA/TensorRT GitHub",
    )
    catalog.add_argument(
        "--releases",
        default=None,
        help=(
            "Comma-separated NVIDIA/TensorRT tags (v11.2,v11.1). "
            "Identical header+ops tables share one database."
        ),
    )
    catalog.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSON path",
    )
    catalog.set_defaults(handler=_cmd_catalog)

    generate = sub.add_parser(
        "generate",
        help="Build per-release TensorRT JSON/Markdown catalogs (noble/noble-fastos/noble-jetpack)",
    )
    add_generate_arguments(generate)
    generate.set_defaults(handler=run_generate)

    args = parser.parse_args(argv)
    _add_logging()
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
