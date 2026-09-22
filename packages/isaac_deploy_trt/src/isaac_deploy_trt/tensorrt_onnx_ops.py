#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""TensorRT ONNX operator support catalogs.

Versioned by TensorRT release (and compatible CUDA majors). ``TENSORRT_ONNX_OPS``
is the default catalog's operator map so existing callers stay unchanged.

Source of truth for the current default (TensorRT 11.2, ONNX opset 9-24):
https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md

This is a static reference for post-export checks. It is not a live
TensorRT API query and does not guarantee a successful engine build for every
attribute / dtype / shape combination.

To regenerate this table, or to emit CUDA 13.2 catalogs for x86 / Orin / Thor /
DGX (precision and DLA overlays), run::

"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict


class TensorRTOnnxOpInfo(TypedDict):
    supported: bool
    types: List[str]
    restrictions: str


class TensorRTOnnxCatalogInfo(TypedDict):
    """One TensorRT parser table, including the CUDA/TRT versions it covers."""

    tensorrt: str
    cuda: List[str]
    opset_min: int
    opset_max: int
    source: str
    ops: Dict[str, TensorRTOnnxOpInfo]


# Default catalog key (TensorRT major.minor). Add more entries to
# ``TENSORRT_ONNX_CATALOGS`` when supporting additional parser tables.
TENSORRT_ONNX_DEFAULT_VERSION = "11.2"

# Opset range documented alongside the default TensorRT 11.2 operators matrix.
TENSORRT_ONNX_OPSET_MIN = 9
TENSORRT_ONNX_OPSET_MAX = 24

TENSORRT_ONNX_OPS: Dict[str, TensorRTOnnxOpInfo] = {

    "Abs": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Acos": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Acosh": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Add": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "AffineGrid": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "And": {
        "supported": True,
        "types": ['BOOL'],
        "restrictions": '',
    },
    "ArgMax": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "ArgMin": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Asin": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Asinh": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Atan": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Atanh": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Attention": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT8', 'FP8'],
        "restrictions": 'Q, K, V and attn_mask must be 4D. past_key, past_value and nonpad_kv_seqlen inputs unsupported. present_key, present_value and qk_matmul_output outputs unsupported. qk_matmul_output_mode and softcap attributes unsupported. q_num_heads and kv_num_heads via second dim of Q and K/V shapes.',
    },
    "AveragePool": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '2D or 3D Pooling only. dilations must be empty or all ones',
    },
    "BatchNormalization": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Bernoulli": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "BitShift": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "BitwiseAnd": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "BitwiseNot": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "BitwiseOr": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "BitwiseXor": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "BlackmanWindow": {
        "supported": True,
        "types": ['FP32', 'FP16'],
        "restrictions": '',
    },
    "Cast": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'UINT8', 'BOOL'],
        "restrictions": '',
    },
    "CastLike": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'UINT8', 'BOOL'],
        "restrictions": '',
    },
    "Ceil": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Col2Im": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Celu": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "CenterCropPad": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Clip": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Compress": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Concat": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "ConcatFromSequence": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Constant": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'FP8', 'FP4', 'INT4', 'INT32', 'INT64', 'BOOL'],
        "restrictions": 'sparse_value, value_string, and value_strings attributes unsupported',
    },
    "ConstantOfShape": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'FP8', 'FP4', 'INT4', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "Conv": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "ConvInteger": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "ConvTranspose": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Cos": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Cosh": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "CumSum": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'axis must be a build-time constant',
    },
    "DFT": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'Runs on cuFFT-backed plugin; complex packed-real [..., 2]; transform axis must be -2; see TensorRT operators.md for full restrictions',
    },
    "DeformConv": {
        "supported": True,
        "types": ['FP32', 'FP16'],
        "restrictions": 'input must have 1D or 2D spatial dims; pads begin/end along each spatial axis must match',
    },
    "DepthToSpace": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "DequantizeLinear": {
        "supported": True,
        "types": ['INT8', 'FP8', 'FP4', 'INT4'],
        "restrictions": 'x_zero_point must be zero',
    },
    "Det": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Div": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Dropout": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'is_training must be an initializer and evaluate to False',
    },
    "DynamicQuantizeLinear": {
        "supported": False,
        "types": [],
        "restrictions": (
            "Not supported. TensorRT IDynamicQuantize can be composed "
            "from ONNX local functions."
        ),
    },
    "Einsum": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Elu": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Equal": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Erf": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Exp": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Expand": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "EyeLike": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": 'input must have static dimensions',
    },
    "Flatten": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "Floor": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Gather": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "GatherElements": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "GatherND": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "Gelu": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT8', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Gemm": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "GlobalAveragePool": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "GlobalLpPool": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "GlobalMaxPool": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Greater": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "GreaterOrEqual": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "GridSample": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'Input must be 4D or 5D',
    },
    "GroupNormalization": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "GRU": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'For bidirectional GRUs, activation functions must match forward and reverse',
    },
    "HammingWindow": {
        "supported": True,
        "types": ['FP32', 'FP16'],
        "restrictions": '',
    },
    "HannWindow": {
        "supported": True,
        "types": ['FP32', 'FP16'],
        "restrictions": '',
    },
    "HardSigmoid": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "HardSwish": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Hardmax": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'axis dimension of input must be a build-time constant',
    },
    "Identity": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "If": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": 'Branch outputs must have same rank and different names',
    },
    "ImageScaler": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "ImageDecoder": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "InstanceNormalization": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "IsInf": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "IsNaN": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "LayerNormalization": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'Only the first output Y is supported',
    },
    "LeakyRelu": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Less": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "LessOrEqual": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Log": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "LogSoftmax": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Loop": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": 'Scan output length cannot be dynamic; carried dependency shapes must be constant across iterations',
    },
    "LRN": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "LSTM": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'Bidirectional activations must match; input_forget must be 0; layout must be 0',
    },
    "LpNormalization": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "LpPool": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'dilations must be empty or all ones',
    },
    "MatMul": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "MatMulInteger": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Max": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "MaxPool": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '2D or 3D pooling only; Indices output unsupported; dilations empty or all ones',
    },
    "MaxRoiPool": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "MaxUnpool": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Mean": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'FP8', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "MeanVarianceNormalization": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "MelWeightMatrix": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Min": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Mish": {
        "supported": True,
        "types": ['FP32', 'FP16'],
        "restrictions": '',
    },
    "Mod": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Mul": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Multinomial": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Neg": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "NegativeLogLikelihoodLoss": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "NonMaxSuppression": {
        "supported": True,
        "types": ['FP32', 'FP16'],
        "restrictions": '',
    },
    "NonZero": {
        "supported": True,
        "types": ['FP32', 'FP16'],
        "restrictions": '',
    },
    "Not": {
        "supported": True,
        "types": ['BOOL'],
        "restrictions": '',
    },
    "OneHot": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": 'depth must be a build-time constant',
    },
    "Optional": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "OptionalGetElement": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "OptionalHasElement": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Or": {
        "supported": True,
        "types": ['BOOL'],
        "restrictions": '',
    },
    "Pad": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "ParametricSoftplus": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Pow": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "PRelu": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "QLinearConv": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "QLinearMatMul": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "QuantizeLinear": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'y_zero_point must be 0',
    },
    "RandomNormal": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'seed value is ignored by TensorRT',
    },
    "RandomNormalLike": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'seed value is ignored by TensorRT',
    },
    "RandomUniform": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'seed value is ignored by TensorRT',
    },
    "RandomUniformLike": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'seed value is ignored by TensorRT',
    },
    "Range": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Reciprocal": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "ReduceL1": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": 'axes must be an initializer',
    },
    "ReduceL2": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": 'axes must be an initializer',
    },
    "ReduceLogSum": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": 'axes must be an initializer',
    },
    "ReduceLogSumExp": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": 'axes must be an initializer',
    },
    "ReduceMax": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": 'axes must be an initializer',
    },
    "ReduceMean": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": 'axes must be an initializer',
    },
    "ReduceMin": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": 'axes must be an initializer',
    },
    "ReduceProd": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": 'axes must be an initializer',
    },
    "ReduceSum": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": 'axes must be an initializer',
    },
    "ReduceSumSquare": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": 'axes must be an initializer',
    },
    "RegexFullMatch": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Relu": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Reshape": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "Resize": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'See TensorRT operators.md for supported modes and aspect-ratio policy restrictions',
    },
    "ReverseSequence": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "RMSNormalization": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "RNN": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'For bidirectional RNNs, activation functions must match forward and reverse',
    },
    "RoiAlign": {
        "supported": True,
        "types": ['FP32', 'FP16'],
        "restrictions": '',
    },
    "RotaryEmbedding": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'position_ids must be INT64',
    },
    "Round": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "STFT": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": 'frame_step and window must be initializers; input must be real-valued',
    },
    "ScaledTanh": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Scan": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Scatter": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "ScatterElements": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "ScatterND": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": 'reduction other than none is not supported',
    },
    "Selu": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "SequenceAt": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "SequenceConstruct": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "SequenceEmpty": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "SequenceErase": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "SequenceInsert": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "SequenceLength": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "SequenceMap": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Shape": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "Shrink": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Sigmoid": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Sign": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Sin": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Sinh": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Size": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "Slice": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "Softmax": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "SoftmaxCrossEntropyLoss": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Softplus": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Softsign": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "SpaceToDepth": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Split": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "SplitToSequence": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Sqrt": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Squeeze": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": 'axes must be resolvable to a constant',
    },
    "StringConcat": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "StringNormalizer": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "StringSplit": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Sub": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Sum": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": '',
    },
    "Swish": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Tan": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Tanh": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "TensorScatter": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32'],
        "restrictions": 'past_cache and update must be 4D; axis must be -2',
    },
    "TfIdfVectorizer": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "ThresholdedRelu": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Tile": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "TopK": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64'],
        "restrictions": 'sorted must be 1; K input must be less than 3840',
    },
    "Transpose": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "Trilu": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "Unique": {
        "supported": False,
        "types": [],
        "restrictions": '',
    },
    "Unsqueeze": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": 'axes must be resolvable to a constant',
    },
    "Upsample": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16'],
        "restrictions": '',
    },
    "Where": {
        "supported": True,
        "types": ['FP32', 'FP16', 'BF16', 'INT32', 'INT64', 'BOOL'],
        "restrictions": '',
    },
    "Xor": {
        "supported": True,
        "types": ['BOOL'],
        "restrictions": '',
    },
}


# Versioned catalogs. ``ops`` for 11.2 is the ``TENSORRT_ONNX_OPS`` map above.
# Future TensorRT/CUDA tables get their own key and operator dict.
TENSORRT_ONNX_CATALOGS: Dict[str, TensorRTOnnxCatalogInfo] = {
    TENSORRT_ONNX_DEFAULT_VERSION: {
        "tensorrt": TENSORRT_ONNX_DEFAULT_VERSION,
        "cuda": ["12.x", "13.x"],
        "opset_min": TENSORRT_ONNX_OPSET_MIN,
        "opset_max": TENSORRT_ONNX_OPSET_MAX,
        "source": (
            "https://github.com/onnx/onnx-tensorrt/blob/main/docs/operators.md"
        ),
        "ops": TENSORRT_ONNX_OPS,
    },
}


def _normalize_tensorrt_version(version: str) -> str:
    parts = [part for part in str(version).strip().split(".") if part]
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    if parts:
        return parts[0]
    raise ValueError(f"Invalid TensorRT version: {version!r}")


def _cuda_matches(entry: str, requested: str) -> bool:
    """Match catalog CUDA tags like ``13.x`` or ``13.2`` against ``13.2``."""
    want = requested.strip().lower()
    have = entry.strip().lower()
    if have.endswith(".x"):
        return want == have[:-2] or want.startswith(have[:-1])
    return want == have or want.startswith(have + ".")


def list_onnx_ops_catalogs() -> List[str]:
    """Return TensorRT version keys that have an operator catalog."""
    return sorted(TENSORRT_ONNX_CATALOGS)


def onnx_ops_catalog(
    tensorrt: Optional[str] = None,
    *,
    cuda: Optional[str] = None,
) -> TensorRTOnnxCatalogInfo:
    """Return a versioned catalog.

    Args:
        tensorrt: TensorRT version (``11.2`` or ``11.2.1``). Defaults to
            ``TENSORRT_ONNX_DEFAULT_VERSION``.
        cuda: If ``tensorrt`` is omitted, pick the latest catalog that lists
            this CUDA version (``13.2`` matches ``13.x``).
    """
    if tensorrt:
        key = _normalize_tensorrt_version(tensorrt)
        if key not in TENSORRT_ONNX_CATALOGS:
            raise KeyError(
                f"No ONNX operator catalog for TensorRT {tensorrt!r} "
                f"(have {list_onnx_ops_catalogs()})"
            )
        catalog = TENSORRT_ONNX_CATALOGS[key]
        if cuda and not any(_cuda_matches(tag, cuda) for tag in catalog["cuda"]):
            raise KeyError(
                f"TensorRT {key} catalog does not list CUDA {cuda!r} "
                f"(have {catalog['cuda']})"
            )
        return catalog

    if cuda:
        matches = [
            TENSORRT_ONNX_CATALOGS[key]
            for key in sorted(TENSORRT_ONNX_CATALOGS, reverse=True)
            if any(_cuda_matches(tag, cuda) for tag in TENSORRT_ONNX_CATALOGS[key]["cuda"])
        ]
        if not matches:
            raise KeyError(
                f"No ONNX operator catalog lists CUDA {cuda!r} "
                f"(have {list_onnx_ops_catalogs()})"
            )
        return matches[0]

    return TENSORRT_ONNX_CATALOGS[TENSORRT_ONNX_DEFAULT_VERSION]


def onnx_ops_for_version(
    tensorrt: Optional[str] = None,
    *,
    cuda: Optional[str] = None,
) -> Dict[str, TensorRTOnnxOpInfo]:
    """Return the operator map for a TensorRT/CUDA catalog."""
    return onnx_ops_catalog(tensorrt, cuda=cuda)["ops"]


def supported_onnx_ops(
    tensorrt: Optional[str] = None,
    *,
    cuda: Optional[str] = None,
) -> set[str]:
    """Return ONNX op names marked supported=True in the selected catalog."""
    ops = onnx_ops_for_version(tensorrt, cuda=cuda)
    return {name for name, info in ops.items() if info["supported"]}


def unsupported_onnx_ops(
    tensorrt: Optional[str] = None,
    *,
    cuda: Optional[str] = None,
) -> set[str]:
    """Return ONNX op names marked supported=False in the selected catalog."""
    ops = onnx_ops_for_version(tensorrt, cuda=cuda)
    return {name for name, info in ops.items() if not info["supported"]}
