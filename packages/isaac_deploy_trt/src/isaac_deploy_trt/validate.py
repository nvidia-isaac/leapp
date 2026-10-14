#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Static TensorRT ONNX op catalog checks (optional TensorRT parse/build)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import onnx

from isaac_deploy_trt.tensorrt_onnx_ops import onnx_ops_for_version, supported_onnx_ops

logger = logging.getLogger(__name__)


@dataclass
class TensorRTOnnxValidationResult:
    onnx_path: str
    used_ops: List[str] = field(default_factory=list)
    unsupported_ops: List[str] = field(default_factory=list)
    unknown_ops: List[str] = field(default_factory=list)
    parse_ok: Optional[bool] = None
    build_ok: Optional[bool] = None
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        if self.errors:
            return False
        if self.unsupported_ops or self.unknown_ops:
            return False
        if self.parse_ok is False or self.build_ok is False:
            return False
        return True


def collect_onnx_op_types(onnx_path: str) -> List[str]:
    model = onnx.load(onnx_path, load_external_data=False)
    return sorted({node.op_type for node in model.graph.node})


def check_onnx_ops_against_tensorrt(
    onnx_path: str,
    *,
    known_supported: Optional[Sequence[str]] = None,
    tensorrt: Optional[str] = None,
    cuda: Optional[str] = None,
) -> TensorRTOnnxValidationResult:
    used_ops = collect_onnx_op_types(onnx_path)
    ops_map = onnx_ops_for_version(tensorrt, cuda=cuda)
    supported = (
        set(known_supported)
        if known_supported is not None
        else supported_onnx_ops(tensorrt, cuda=cuda)
    )
    catalog = set(ops_map)
    unsupported = sorted(op for op in used_ops if op in catalog and op not in supported)
    unknown = sorted(op for op in used_ops if op not in catalog)
    result = TensorRTOnnxValidationResult(
        onnx_path=onnx_path,
        used_ops=used_ops,
        unsupported_ops=unsupported,
        unknown_ops=unknown,
    )
    if unsupported:
        result.errors.append(
            f"ONNX ops not marked supported by TensorRT catalog: {unsupported}"
        )
    if unknown:
        result.errors.append(
            f"ONNX ops not present in TensorRT catalog: {unknown}"
        )
    return result


def try_tensorrt_build(onnx_path: str, *, build: bool = True) -> TensorRTOnnxValidationResult:
    result = TensorRTOnnxValidationResult(onnx_path=onnx_path)
    try:
        import tensorrt as trt
    except ImportError as exc:
        result.parse_ok = False
        if build:
            result.build_ok = False
        result.errors.append(f"tensorrt is not installed: {exc}")
        return result

    logger_trt = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger_trt)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger_trt)
    with open(onnx_path, "rb") as handle:
        parse_ok = parser.parse(handle.read())
    result.parse_ok = bool(parse_ok)
    if not parse_ok:
        for i in range(parser.num_errors):
            result.errors.append(str(parser.get_error(i)))
        result.build_ok = False if build else None
        return result
    if not build:
        return result
    engine = builder.build_serialized_network(network, builder.create_builder_config())
    result.build_ok = engine is not None
    if engine is None:
        result.errors.append("TensorRT build_serialized_network returned None")
    return result


def validate_onnx_for_tensorrt(
    onnx_path: str,
    *,
    build: bool = False,
    strict: bool = False,
    tensorrt: Optional[str] = None,
    cuda: Optional[str] = None,
) -> TensorRTOnnxValidationResult:
    result = check_onnx_ops_against_tensorrt(
        onnx_path, tensorrt=tensorrt, cuda=cuda
    )
    if build:
        built = try_tensorrt_build(onnx_path, build=True)
        result.parse_ok = built.parse_ok
        result.build_ok = built.build_ok
        result.errors.extend(built.errors)
    if result.ok:
        logger.info("TensorRT catalog check passed for %s", onnx_path)
    else:
        message = f"TensorRT issues for {onnx_path}: " + "; ".join(result.errors)
        if strict:
            raise ValueError(message)
        logger.warning(message)
    return result
