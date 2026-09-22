# TensorRT layers — Linux x86_64 (CUDA 13.2)

- CUDA: `13.2`
- TensorRT: TensorRT 10.13+ or 11.x (11.2.1 packages built against CUDA 13.3, compatible with CUDA 13.x)
- Arch: Ampere / Ada / Hopper / Blackwell dGPU (CC 8.0–12.x (SKU-dependent))
- Parser table: TensorRT 11.2, opset 9-24
- GPU precisions: FP32, TF32, FP16, BF16, FP8, FP4, INT8, INT4, INT32, INT64, UINT8, BOOL, FP64
- DLA: False
- Supported releases: v10.8.0

## Notes

- ONNX parser coverage matches onnx-tensorrt operators.md.
- FP8 requires Hopper or newer; FP4 requires Blackwell.
- BF16 is not available on every Ampere SKU (for example some GA10x).
- No DLA.

## TensorRT headers

- `https://raw.githubusercontent.com/NVIDIA/TensorRT/v10.8.0/include/NvInferRuntimeBase.h`
- `https://raw.githubusercontent.com/NVIDIA/TensorRT/v10.8.0/include/NvInfer.h`
- `https://raw.githubusercontent.com/NVIDIA/TensorRT/v10.8.0/include/NvInferRuntime.h`
- NetworkDefinitionCreationFlag::kSTRONGLY_TYPED (TensorRT 11: always on; the flag is retained for API compatibility). Tensor DataType is authoritative; TensorRT will not implicitly cast. Listed nvinfer_datatypes must match the ONNX/TRT tensor types.

## ONNX operators (supported)

| Operator | Types | nvinfer DataType | Restrictions |
|---|---|---|---|
| Abs | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| Acos | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Acosh | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Add | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| And | BOOL | kBOOL |  |
| ArgMax | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| ArgMin | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| Asin | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Asinh | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Atan | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Atanh | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Attention | FP32, FP16, BF16, INT8, FP8 | kFLOAT, kHALF, kBF16, kINT8, kFP8 | `Q`, `K`, `V` and `attn_mask ` must be 4D. `past_key`, `past_value` and `nonpad_kv_seqlen` inputs are unsupported. `present_key`, `present_value` and `qk_matmul_output` outputs are unsupported. `qk_matmul_output_mode` and `softcap` attributes are unsupported. `q_num_heads` and `kv_num_heads` attributes are supported via being specified as the second dimension of `Q` and `K`/`V`'s shapes respectively. |
| AveragePool | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | 2D or 3D Pooling only. `dilations` must be empty or all ones |
| BatchNormalization | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| BlackmanWindow | FP32, FP16 | kFLOAT, kHALF |  |
| Cast | FP32, FP16, BF16, INT32, INT64, UINT8, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kUINT8, kBOOL |  |
| CastLike | FP32, FP16, BF16, INT32, INT64, UINT8, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kUINT8, kBOOL |  |
| Ceil | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Celu | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Clip | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Concat | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| Constant | FP32, FP16, BF16, FP8, FP4, INT4, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kFP8, kFP4, kINT4, kINT32, kINT64, kBOOL | `sparse_value`, `value_string`, and `value_strings` attributes are unsupported. |
| ConstantOfShape | FP32, FP16, BF16, FP8, FP4, INT4, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kFP8, kFP4, kINT4, kINT32, kINT64, kBOOL |  |
| Conv | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| ConvTranspose | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Cos | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Cosh | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| CumSum | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | `axis` must be a build-time constant |
| DFT | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | Runs on a cuFFT-backed plugin so cuFFT must be available at runtime. Complex tensors use the packed-real `[..., 2]` layout, and the innermost dimension must be a static 1 (real) or 2 (complex). The transform axis must be `-2`. The opset 20 `axis` input must be a build-time constant scalar. `dft_length` is required for the onesided inverse (C2R) and unsupported otherwise. FP16 and BF16 require power-of-two signal lengths. |
| DeformConv | FP32, FP16 | kFLOAT, kHALF | `input` must have 1D or 2D spatial dimensions. `pads` for the beginning and end along each spatial axis must be the same |
| DepthToSpace | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| DequantizeLinear | INT8, FP8, FP4, INT4 | kINT8, kFP8, kFP4, kINT4 | `x_zero_point` must be zero |
| Div | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| Dropout | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | `is_training` must be an initializer and evaluate to False. |
| Einsum | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Elu | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Equal | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| Erf | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Exp | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Expand | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| EyeLike | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL | `input` must have static dimensions |
| Flatten | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| Floor | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Gather | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| GatherElements | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| GatherND | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| Gelu | FP32, FP16, BF16, INT8, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT8, kINT32, kINT64 |  |
| Gemm | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| GlobalAveragePool | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| GlobalLpPool | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| GlobalMaxPool | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Greater | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| GreaterOrEqual | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| GridSample | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | Input must be 4D or 5D. |
| GroupNormalization | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| GRU | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | For bidirectional GRUs, activation functions must be the same for both the forward and reverse pass |
| HammingWindow | FP32, FP16 | kFLOAT, kHALF |  |
| HannWindow | FP32, FP16 | kFLOAT, kHALF |  |
| HardSigmoid | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| HardSwish | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Hardmax | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | `axis` dimension of input must be a build-time constant |
| Identity | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| If | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL | Output tensors of the two conditional branches must have the same rank and must have different names |
| ImageScaler | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| InstanceNormalization | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| IsInf | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| IsNaN | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| LayerNormalization | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | Only the first output `Y` is supported. |
| LeakyRelu | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Less | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| LessOrEqual | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| Log | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| LogSoftmax | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Loop | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL | Scan output length cannot be dynamic. The shape of Loop carried dependencies must be the same across all loop iterations. |
| LRN | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| LSTM | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | For bidirectional LSTMs, activation functions must be the same for both the forward and reverse pass. `input_forget` attribute must be 0. `layout` attribute must be 0. |
| LpNormalization | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| LpPool | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | `dilations` must be empty or all ones |
| MatMul | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Max | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| MaxPool | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | 2D or 3D pooling only. `Indices` output tensor unsupported. `dilations` must be empty or all ones |
| Mean | FP32, FP16, BF16, FP8, INT32, INT64 | kFLOAT, kHALF, kBF16, kFP8, kINT32, kINT64 |  |
| MeanVarianceNormalization | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Min | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| Mish | FP32, FP16 | kFLOAT, kHALF |  |
| Mod | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| Mul | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| Neg | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| NonMaxSuppression | FP32, FP16 | kFLOAT, kHALF |  |
| NonZero | FP32, FP16 | kFLOAT, kHALF |  |
| Not | BOOL | kBOOL |  |
| OneHot | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL | `depth` must be a build-time constant |
| Or | BOOL | kBOOL |  |
| Pad | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| ParametricSoftplus | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Pow | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| PRelu | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| QuantizeLinear | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | `y_zero_point` must be 0 |
| RandomNormal | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | `seed` value is ignored by TensorRT |
| RandomNormalLike | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | `seed` value is ignored by TensorRT |
| RandomUniform | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | `seed` value is ignored by TensorRT |
| RandomUniformLike | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | `seed` value is ignored by TensorRT |
| Range | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| Reciprocal | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| ReduceL1 | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 | `axes` must be an initializer |
| ReduceL2 | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 | `axes` must be an initializer |
| ReduceLogSum | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 | `axes` must be an initializer |
| ReduceLogSumExp | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 | `axes` must be an initializer |
| ReduceMax | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 | `axes` must be an initializer |
| ReduceMean | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 | `axes` must be an initializer |
| ReduceMin | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 | `axes` must be an initializer |
| ReduceProd | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 | `axes` must be an initializer |
| ReduceSum | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 | `axes` must be an initializer |
| ReduceSumSquare | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 | `axes` must be an initializer |
| Relu | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| Reshape | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| Resize | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | Supported resize transformation modes: `half_pixel`, `pytorch_half_pixel`, `tf_half_pixel_for_nn`, `asymmetric`, and `align_corners`.<br />Supported resize modes: `nearest`, `linear`.<br />Supported nearest modes: `floor`, `ceil`, `round_prefer_floor`, `round_prefer_ceil`.<br />Supported aspect ratio policy: `stretch`.<br />When `scales` is a tensor input, `axes` must be an iota vector of length rank(input).<br />Antialiasing is not supported. |
| ReverseSequence | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| RMSNormalization | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| RNN | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | For bidirectional RNNs, activation functions must be the same for both the forward and reverse pass |
| RoiAlign | FP32, FP16 | kFLOAT, kHALF |  |
| RotaryEmbedding | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | `position_ids` must be INT64 |
| Round | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| STFT | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 | `frame_step` and `window` must be an initializer. Input must be real-valued. |
| ScaledTanh | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Scan | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Scatter | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| ScatterElements | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| ScatterND | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 | `reduction` other than `none` is not supported |
| Selu | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Shape | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| Shrink | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| Sigmoid | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Sign | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| Sin | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Sinh | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Size | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| Slice | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| Softmax | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Softplus | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Softsign | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| SpaceToDepth | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| Split | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| Sqrt | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Squeeze | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL | `axes` must be resolvable to a constant. |
| Sub | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| Sum | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 |  |
| Swish | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Tan | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Tanh | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| TensorScatter | FP32, FP16, BF16, INT32 | kFLOAT, kHALF, kBF16, kINT32 | `past_cache` and `update` must be 4D. `axis` must be -2. |
| ThresholdedRelu | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Tile | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| TopK | FP32, FP16, BF16, INT32, INT64 | kFLOAT, kHALF, kBF16, kINT32, kINT64 | `sorted` must be 1. `K` input must be less than 3840. |
| Transpose | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| Trilu | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| Unsqueeze | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL | `axes` must be resolvable to a constant. |
| Upsample | FP32, FP16, BF16 | kFLOAT, kHALF, kBF16 |  |
| Where | FP32, FP16, BF16, INT32, INT64, BOOL | kFLOAT, kHALF, kBF16, kINT32, kINT64, kBOOL |  |
| Xor | BOOL | kBOOL |  |

## ONNX operators (unsupported)

| Operator | Restrictions |
|---|---|
| AffineGrid |  |
| Bernoulli |  |
| BitShift |  |
| BitwiseAnd |  |
| BitwiseNot |  |
| BitwiseOr |  |
| BitwiseXor |  |
| Col2Im |  |
| CenterCropPad |  |
| Compress |  |
| ConcatFromSequence |  |
| ConvInteger |  |
| Det |  |
| DynamicQuantizeLinear | Not supported. TensorRT's IDynamicQuantize can be composed from ONNX operators in the form of a model local function. |
| ImageDecoder |  |
| MatMulInteger |  |
| MaxRoiPool |  |
| MaxUnpool |  |
| MelWeightMatrix |  |
| Multinomial |  |
| NegativeLogLikelihoodLoss |  |
| Optional |  |
| OptionalGetElement |  |
| OptionalHasElement |  |
| QLinearConv |  |
| QLinearMatMul |  |
| RegexFullMatch |  |
| SequenceAt |  |
| SequenceConstruct |  |
| SequenceEmpty |  |
| SequenceErase |  |
| SequenceInsert |  |
| SequenceLength |  |
| SequenceMap |  |
| SoftmaxCrossEntropyLoss |  |
| SplitToSequence |  |
| StringConcat |  |
| StringNormalizer |  |
| StringSplit |  |
| TfIdfVectorizer |  |
| Unique |  |

## Native TensorRT layers

| Layer | GPU | DLA | Notes |
|---|---|---|---|
| ACTIVATION | True | False |  |
| ASSERTION | True | False |  |
| ATTENTION | True | False |  |
| CAST | True | False |  |
| CONCATENATION | True | False |  |
| CONDITION | True | False |  |
| CONSTANT | True | False |  |
| CONVOLUTION | True | False |  |
| CUMULATIVE | True | False |  |
| DECONVOLUTION | True | False |  |
| DEQUANTIZE | True | False |  |
| DYNAMIC_QUANTIZE | True | False |  |
| EINSUM | True | False |  |
| ELEMENTWISE | True | False |  |
| FILL | True | False |  |
| GATHER | True | False |  |
| GRID_SAMPLE | True | False |  |
| IDENTITY | True | False |  |
| LOOP | True | False |  |
| LOOP_OUTPUT | True | False |  |
| LRN | True | False |  |
| MATRIX_MULTIPLY | True | False |  |
| NMS | True | False |  |
| NON_ZERO | True | False |  |
| NORMALIZATION | True | False |  |
| ONE_HOT | True | False |  |
| PADDING | True | False |  |
| PARAMETRIC_RELU | True | False |  |
| PLUGIN | True | False |  |
| PLUGIN_V2 | True | False |  |
| PLUGIN_V3 | True | False |  |
| POOLING | True | False |  |
| QUANTIZE | True | False |  |
| RAGGED_SOFTMAX | True | False |  |
| RECURRENCE | True | False |  |
| REDUCE | True | False |  |
| RESIZE | True | False |  |
| REVERSE_SEQUENCE | True | False |  |
| SCALE | True | False |  |
| SCATTER | True | False |  |
| SELECT | True | False |  |
| SHAPE | True | False |  |
| SHUFFLE | True | False |  |
| SLICE | True | False |  |
| SOFTMAX | True | False |  |
| SQUEEZE | True | False |  |
| TOPK | True | False |  |
| TRIP_LIMIT | True | False |  |
| UNARY | True | False |  |
| UNSQUEEZE | True | False |  |
| ITERATOR | True | False |  |
| CONDITIONAL_INPUT | True | False |  |
| CONDITIONAL_OUTPUT | True | False |  |
