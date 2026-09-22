# TensorRT layers — Jetson Orin (CUDA 13.2 / JetPack 7.2)

- CUDA: `13.2`
- TensorRT: TensorRT 10.x from JetPack (TensorRT 11.2.1 does not support JetPack)
- Arch: Ampere GA10B iGPU (CC 8.7)
- Parser table: TensorRT 11.2, opset 9-24
- GPU precisions: FP32, TF32, FP16, INT8, INT32, INT64, UINT8, BOOL
- DLA: True
- Supported releases: v10.12.0, v10.13.0, v10.13.2, v10.13.3

## Notes

- Stay on JetPack TensorRT 10.x for CUDA 13.2 Orin.
- No BF16, FP8, or FP4 on Orin.
- DLA is FP16/INT8 and only for the DLA layer subset; TensorRT 11.x has no DLA.
- Orin DLA Softmax is Orin-only (not Xavier).

## TensorRT headers

- `https://raw.githubusercontent.com/NVIDIA/TensorRT/v10.12.0/include/NvInferRuntimeBase.h`
- `https://raw.githubusercontent.com/NVIDIA/TensorRT/v10.12.0/include/NvInfer.h`
- `https://raw.githubusercontent.com/NVIDIA/TensorRT/v10.12.0/include/NvInferRuntime.h`
- NetworkDefinitionCreationFlag::kSTRONGLY_TYPED (TensorRT 11: always on; the flag is retained for API compatibility). Tensor DataType is authoritative; TensorRT will not implicitly cast. Listed nvinfer_datatypes must match the ONNX/TRT tensor types.

## ONNX operators (supported)

| Operator | Types | nvinfer DataType | Restrictions |
|---|---|---|---|
| Abs | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Acos | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Acosh | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Add | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| And | BOOL | kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ArgMax | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ArgMin | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Asin | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Asinh | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Atan | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Atanh | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Attention | FP32, FP16, INT8 | kFLOAT, kHALF, kINT8 | `Q`, `K`, `V` and `attn_mask ` must be 4D. `past_key`, `past_value` and `nonpad_kv_seqlen` inputs are unsupported. `present_key`, `present_value` and `qk_matmul_output` outputs are unsupported. `qk_matmul_output_mode` and `softcap` attributes are unsupported. `q_num_heads` and `kv_num_heads` attributes are supported via being specified as the second dimension of `Q` and `K`/`V`'s shapes respectively.; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| AveragePool | FP32, FP16 | kFLOAT, kHALF | 2D or 3D Pooling only. `dilations` must be empty or all ones; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| BatchNormalization | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| BlackmanWindow | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Cast | FP32, FP16, INT32, INT64, UINT8, BOOL | kFLOAT, kHALF, kINT32, kINT64, kUINT8, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| CastLike | FP32, FP16, INT32, INT64, UINT8, BOOL | kFLOAT, kHALF, kINT32, kINT64, kUINT8, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Ceil | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Celu | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Clip | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Concat | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Constant | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | `sparse_value`, `value_string`, and `value_strings` attributes are unsupported.; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ConstantOfShape | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Conv | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ConvTranspose | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Cos | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Cosh | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| CumSum | FP32, FP16 | kFLOAT, kHALF | `axis` must be a build-time constant; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| DFT | FP32, FP16 | kFLOAT, kHALF | Runs on a cuFFT-backed plugin so cuFFT must be available at runtime. Complex tensors use the packed-real `[..., 2]` layout, and the innermost dimension must be a static 1 (real) or 2 (complex). The transform axis must be `-2`. The opset 20 `axis` input must be a build-time constant scalar. `dft_length` is required for the onesided inverse (C2R) and unsupported otherwise. FP16 and BF16 require power-of-two signal lengths.; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| DeformConv | FP32, FP16 | kFLOAT, kHALF | `input` must have 1D or 2D spatial dimensions. `pads` for the beginning and end along each spatial axis must be the same; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| DepthToSpace | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| DequantizeLinear | INT8 | kINT8 | `x_zero_point` must be zero; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Div | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Dropout | FP32, FP16 | kFLOAT, kHALF | `is_training` must be an initializer and evaluate to False.; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Einsum | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Elu | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Equal | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Erf | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Exp | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Expand | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| EyeLike | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | `input` must have static dimensions; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Flatten | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Floor | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Gather | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| GatherElements | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| GatherND | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Gelu | FP32, FP16, INT8, INT32, INT64 | kFLOAT, kHALF, kINT8, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Gemm | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| GlobalAveragePool | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| GlobalLpPool | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| GlobalMaxPool | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Greater | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| GreaterOrEqual | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| GridSample | FP32, FP16 | kFLOAT, kHALF | Input must be 4D or 5D.; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| GroupNormalization | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| GRU | FP32, FP16 | kFLOAT, kHALF | For bidirectional GRUs, activation functions must be the same for both the forward and reverse pass; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| HammingWindow | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| HannWindow | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| HardSigmoid | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| HardSwish | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Hardmax | FP32, FP16 | kFLOAT, kHALF | `axis` dimension of input must be a build-time constant; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Identity | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| If | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Output tensors of the two conditional branches must have the same rank and must have different names; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ImageScaler | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| InstanceNormalization | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| IsInf | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| IsNaN | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| LayerNormalization | FP32, FP16 | kFLOAT, kHALF | Only the first output `Y` is supported.; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| LeakyRelu | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Less | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| LessOrEqual | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Log | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| LogSoftmax | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Loop | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Scan output length cannot be dynamic. The shape of Loop carried dependencies must be the same across all loop iterations.; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| LRN | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| LSTM | FP32, FP16 | kFLOAT, kHALF | For bidirectional LSTMs, activation functions must be the same for both the forward and reverse pass. `input_forget` attribute must be 0. `layout` attribute must be 0.; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| LpNormalization | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| LpPool | FP32, FP16 | kFLOAT, kHALF | `dilations` must be empty or all ones; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| MatMul | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Max | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| MaxPool | FP32, FP16 | kFLOAT, kHALF | 2D or 3D pooling only. `Indices` output tensor unsupported. `dilations` must be empty or all ones; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Mean | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| MeanVarianceNormalization | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Min | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Mish | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Mod | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Mul | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Neg | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| NonMaxSuppression | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| NonZero | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Not | BOOL | kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| OneHot | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | `depth` must be a build-time constant; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Or | BOOL | kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Pad | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ParametricSoftplus | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Pow | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| PRelu | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| QuantizeLinear | FP32, FP16 | kFLOAT, kHALF | `y_zero_point` must be 0; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| RandomNormal | FP32, FP16 | kFLOAT, kHALF | `seed` value is ignored by TensorRT; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| RandomNormalLike | FP32, FP16 | kFLOAT, kHALF | `seed` value is ignored by TensorRT; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| RandomUniform | FP32, FP16 | kFLOAT, kHALF | `seed` value is ignored by TensorRT; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| RandomUniformLike | FP32, FP16 | kFLOAT, kHALF | `seed` value is ignored by TensorRT; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Range | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Reciprocal | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ReduceL1 | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | `axes` must be an initializer; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ReduceL2 | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | `axes` must be an initializer; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ReduceLogSum | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | `axes` must be an initializer; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ReduceLogSumExp | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | `axes` must be an initializer; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ReduceMax | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | `axes` must be an initializer; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ReduceMean | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | `axes` must be an initializer; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ReduceMin | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | `axes` must be an initializer; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ReduceProd | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | `axes` must be an initializer; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ReduceSum | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | `axes` must be an initializer; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ReduceSumSquare | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | `axes` must be an initializer; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Relu | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Reshape | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Resize | FP32, FP16 | kFLOAT, kHALF | Supported resize transformation modes: `half_pixel`, `pytorch_half_pixel`, `tf_half_pixel_for_nn`, `asymmetric`, and `align_corners`.<br />Supported resize modes: `nearest`, `linear`.<br />Supported nearest modes: `floor`, `ceil`, `round_prefer_floor`, `round_prefer_ceil`.<br />Supported aspect ratio policy: `stretch`.<br />When `scales` is a tensor input, `axes` must be an iota vector of length rank(input).<br />Antialiasing is not supported.; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ReverseSequence | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| RMSNormalization | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| RNN | FP32, FP16 | kFLOAT, kHALF | For bidirectional RNNs, activation functions must be the same for both the forward and reverse pass; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| RoiAlign | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| RotaryEmbedding | FP32, FP16 | kFLOAT, kHALF | `position_ids` must be INT64; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Round | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| STFT | FP32, FP16 | kFLOAT, kHALF | `frame_step` and `window` must be an initializer. Input must be real-valued.; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ScaledTanh | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Scan | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Scatter | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ScatterElements | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ScatterND | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | `reduction` other than `none` is not supported; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Selu | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Shape | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Shrink | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Sigmoid | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Sign | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Sin | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Sinh | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Size | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Slice | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Softmax | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Softplus | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Softsign | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| SpaceToDepth | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Split | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Sqrt | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Squeeze | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | `axes` must be resolvable to a constant.; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Sub | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Sum | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Swish | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Tan | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Tanh | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| TensorScatter | FP32, FP16, INT32 | kFLOAT, kHALF, kINT32 | `past_cache` and `update` must be 4D. `axis` must be -2.; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ThresholdedRelu | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Tile | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| TopK | FP32, FP16, INT32, INT64 | kFLOAT, kHALF, kINT32, kINT64 | `sorted` must be 1. `K` input must be less than 3840.; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Transpose | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Trilu | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Unsqueeze | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | `axes` must be resolvable to a constant.; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Upsample | FP32, FP16 | kFLOAT, kHALF | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Where | FP32, FP16, INT32, INT64, BOOL | kFLOAT, kHALF, kINT32, kINT64, kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Xor | BOOL | kBOOL | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |

## ONNX operators (unsupported)

| Operator | Restrictions |
|---|---|
| AffineGrid | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Bernoulli | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| BitShift | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| BitwiseAnd | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| BitwiseNot | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| BitwiseOr | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| BitwiseXor | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Col2Im | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| CenterCropPad | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Compress | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ConcatFromSequence | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ConvInteger | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Det | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| DynamicQuantizeLinear | Not supported. TensorRT's IDynamicQuantize can be composed from ONNX operators in the form of a model local function.; Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| ImageDecoder | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| MatMulInteger | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| MaxRoiPool | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| MaxUnpool | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| MelWeightMatrix | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Multinomial | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| NegativeLogLikelihoodLoss | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Optional | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| OptionalGetElement | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| OptionalHasElement | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| QLinearConv | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| QLinearMatMul | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| RegexFullMatch | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| SequenceAt | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| SequenceConstruct | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| SequenceEmpty | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| SequenceErase | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| SequenceInsert | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| SequenceLength | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| SequenceMap | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| SoftmaxCrossEntropyLoss | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| SplitToSequence | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| StringConcat | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| StringNormalizer | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| StringSplit | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| TfIdfVectorizer | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |
| Unique | Parser table is TensorRT 11.x; this platform ships TensorRT 10.x. Treat newly added 11.x-only ops as unverified. |

## Native TensorRT layers

| Layer | GPU | DLA | Notes |
|---|---|---|---|
| ACTIVATION | True | True | FP16, INT8; 2D; ReLU, Sigmoid, TanH, Clipped ReLU, Leaky ReLU |
| ASSERTION | True | False |  |
| ATTENTION | False | False | Documented as TensorRT 11.x native layer; not in JetPack 10.x |
| CAST | True | False |  |
| CONCATENATION | True | True | FP16, INT8; channel axis only |
| CONDITION | True | False |  |
| CONSTANT | True | False |  |
| CONVOLUTION | True | True | FP16, INT8; 2D; kernel [1,32]; stride [1,8]; channels [1,8192] |
| CUMULATIVE | True | False |  |
| DECONVOLUTION | True | True | FP16, INT8; 2D; no grouped/dilated; padding 0 |
| DEQUANTIZE | True | False |  |
| DYNAMIC_QUANTIZE | False | False | Documented as TensorRT 11.x native layer; not in JetPack 10.x |
| EINSUM | True | False |  |
| ELEMENTWISE | True | True | FP16, INT8; 2D; Sum, Sub, Product, Max, Min, Div, Pow |
| FILL | True | False |  |
| GATHER | True | False |  |
| GRID_SAMPLE | True | False |  |
| IDENTITY | True | False |  |
| LOOP | True | False |  |
| LOOP_OUTPUT | True | False |  |
| LRN | True | True | FP16 (INT8 promoted to FP16); window 3/5/7/9; ACROSS_CHANNELS |
| MATRIX_MULTIPLY | True | False |  |
| NMS | True | False |  |
| NON_ZERO | True | False |  |
| NORMALIZATION | True | False |  |
| ONE_HOT | True | False |  |
| PADDING | True | False |  |
| PARAMETRIC_RELU | True | True | FP16, INT8; slope must be a build-time constant |
| PLUGIN | True | False |  |
| PLUGIN_V2 | True | False |  |
| PLUGIN_V3 | True | False |  |
| POOLING | True | True | FP16, INT8; 2D; MAX/AVERAGE; window [1,8]; stride [1,16] |
| QUANTIZE | True | False |  |
| RAGGED_SOFTMAX | True | False |  |
| RECURRENCE | True | False |  |
| REDUCE | True | True | FP16, INT8; 4D; MAX; CHW axes |
| RESIZE | True | True | FP16, INT8; nearest [1,32], bilinear [1,4] |
| REVERSE_SEQUENCE | True | False |  |
| SCALE | True | True | FP16, INT8; Uniform, Per-Channel, ElementWise |
| SCATTER | True | False |  |
| SELECT | True | False |  |
| SHAPE | True | False |  |
| SHUFFLE | True | True | FP16, INT8; 4D; no batch transpose |
| SLICE | True | True | FP16, INT8; 4D; static slicing |
| SOFTMAX | True | True | FP16, INT8; Orin DLA only; axis dim <= 1024 in optimized mode |
| SQUEEZE | True | False |  |
| TOPK | True | False |  |
| TRIP_LIMIT | True | False |  |
| UNARY | True | True | INT8; ABS, SIN, COS, ATAN |
| UNSQUEEZE | True | False |  |
| ITERATOR | True | False |  |
| CONDITIONAL_INPUT | True | False |  |
| CONDITIONAL_OUTPUT | True | False |  |
