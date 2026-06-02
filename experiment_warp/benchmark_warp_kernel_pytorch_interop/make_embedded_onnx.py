# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a single ONNX graph with embedded Warp APIC bundles and dense weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from benchmark_pipeline import DENSE_OUT_DIM, FEATURE_DIM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "onnx_embedded_wrp"))
from wrp_bundle import pack_bundle  # noqa: E402


DOMAIN = "com.nvidia.warp"


def _wrp_node(
    node_name: str,
    wrp_path: Path,
    data_inputs: list[str],
    input_param_names: list[str],
    output_name: str,
    output_param_name: str,
    output_shape: tuple[int, int],
) -> tuple[onnx.NodeProto, onnx.TensorProto]:
    archive, wrp_name = pack_bundle(wrp_path)
    bundle_name = f"{node_name}.bundle"
    bundle = helper.make_tensor(
        name=bundle_name,
        data_type=TensorProto.UINT8,
        dims=[len(archive)],
        vals=archive,
        raw=True,
    )
    node = helper.make_node(
        "WrpRunner",
        inputs=[*data_inputs, bundle_name],
        outputs=[output_name],
        domain=DOMAIN,
        name=node_name,
        wrp_name=wrp_name,
        input_names=",".join(input_param_names),
        output_names=output_param_name,
        output_shape=f"{output_shape[0]},{output_shape[1]}",
    )
    return node, bundle


def make_model(
    artifacts_dir: Path,
    output_path: Path,
    external_data: bool,
    size_threshold: int,
) -> None:
    dense = onnx.load(artifacts_dir / "dense.onnx")
    dense_input_shape = dense.graph.input[0].type.tensor_type.shape.dim
    batch_size = dense_input_shape[0].dim_value

    dummy_feature = numpy_helper.from_array(
        np.zeros((batch_size, FEATURE_DIM), dtype=np.float32),
        name="dummy_feature",
    )
    dummy_dense = numpy_helper.from_array(
        np.zeros((batch_size, DENSE_OUT_DIM), dtype=np.float32),
        name="dummy_dense",
    )

    branch_wave, branch_wave_bundle = _wrp_node(
        "branch_wave_wrp",
        artifacts_dir / "branch_wave_features.wrp",
        ["input", "dummy_feature"],
        ["input", "dummy"],
        "branch_wave",
        "output",
        (batch_size, FEATURE_DIM),
    )
    branch_stencil, branch_stencil_bundle = _wrp_node(
        "branch_stencil_wrp",
        artifacts_dir / "branch_stencil_features.wrp",
        ["input", "dummy_feature"],
        ["input", "dummy"],
        "branch_stencil",
        "output",
        (batch_size, FEATURE_DIM),
    )
    merge, merge_bundle = _wrp_node(
        "merge_wrp",
        artifacts_dir / "merge_parallel_features.wrp",
        ["branch_wave", "branch_stencil"],
        ["branch_a", "branch_b"],
        "merged",
        "merged",
        (batch_size, FEATURE_DIM),
    )
    final, final_bundle = _wrp_node(
        "final_postprocess_wrp",
        artifacts_dir / "final_postprocess.wrp",
        ["dense_out", "dummy_dense"],
        ["dense_out", "dummy"],
        "final",
        "final",
        (batch_size, DENSE_OUT_DIM),
    )

    graph = helper.make_graph(
        [branch_wave, branch_stencil, merge, *dense.graph.node, final],
        "warp_onnx_vs_cpp_embedded",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch_size, FEATURE_DIM])],
        outputs=[helper.make_tensor_value_info("final", TensorProto.FLOAT, [batch_size, DENSE_OUT_DIM])],
        initializer=[
            *dense.graph.initializer,
            dummy_feature,
            dummy_dense,
            branch_wave_bundle,
            branch_stencil_bundle,
            merge_bundle,
            final_bundle,
        ],
        value_info=[
            helper.make_tensor_value_info("branch_wave", TensorProto.FLOAT, [batch_size, FEATURE_DIM]),
            helper.make_tensor_value_info("branch_stencil", TensorProto.FLOAT, [batch_size, FEATURE_DIM]),
            helper.make_tensor_value_info("merged", TensorProto.FLOAT, [batch_size, FEATURE_DIM]),
            helper.make_tensor_value_info("dense_out", TensorProto.FLOAT, [batch_size, DENSE_OUT_DIM]),
        ],
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid(DOMAIN, 1)],
        producer_name="warp_onnx_vs_cpp_benchmark",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_file = output_path.with_name(output_path.name + ".data")
    if data_file.exists():
        data_file.unlink()

    if external_data:
        onnx.save_model(
            model,
            str(output_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file.name,
            size_threshold=size_threshold,
            convert_attribute=False,
        )
    else:
        onnx.save(model, str(output_path))

    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument("--artifacts-dir", type=Path, default=root / "artifacts")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "embedded_pipeline.onnx")
    parser.add_argument("--external-data", action="store_true")
    parser.add_argument("--size-threshold", type=int, default=1024)
    args = parser.parse_args()
    make_model(args.artifacts_dir, args.output, args.external_data, args.size_threshold)


if __name__ == "__main__":
    main()
