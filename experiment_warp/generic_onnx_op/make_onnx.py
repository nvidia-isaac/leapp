"""Create an ONNX model containing two WrpRunner custom-op nodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from generate_wrp import HEIGHT, SMALL_HEIGHT, SMALL_WIDTH, WIDTH


DOMAIN = "com.nvidia.warp"


def make_model(
    kernel_chain_wrp: Path,
    subtract_square_wrp: Path,
    output_path: Path,
) -> None:
    kernel_chain_node = helper.make_node(
        "WrpRunner",
        inputs=["a", "b"],
        outputs=["averaged"],
        domain=DOMAIN,
        name="run_kernel_chain_wrp",
        wrp_path=str(kernel_chain_wrp.resolve()),
        input_names="a,b",
        output_names="averaged",
        output_shape="4,5",
    )
    subtract_square_node = helper.make_node(
        "WrpRunner",
        inputs=["x", "y"],
        outputs=["squared"],
        domain=DOMAIN,
        name="run_subtract_square_wrp",
        wrp_path=str(subtract_square_wrp.resolve()),
        input_names="x,y",
        output_names="squared",
        output_shape="2,3",
    )

    graph = helper.make_graph(
        [kernel_chain_node, subtract_square_node],
        "wrp_runner_two_file_demo",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [HEIGHT, WIDTH]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [HEIGHT, WIDTH]),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [SMALL_HEIGHT, SMALL_WIDTH]),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [SMALL_HEIGHT, SMALL_WIDTH]),
        ],
        outputs=[
            helper.make_tensor_value_info("averaged", TensorProto.FLOAT, [HEIGHT, WIDTH]),
            helper.make_tensor_value_info("squared", TensorProto.FLOAT, [SMALL_HEIGHT, SMALL_WIDTH]),
        ],
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid(DOMAIN, 1),
        ],
        producer_name="generic_wrp_onnx_op_prototype",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = Path(__file__).resolve().parent / "artifacts"
    parser.add_argument(
        "--kernel-chain-wrp",
        type=Path,
        default=default_dir / "kernel_chain.wrp",
    )
    parser.add_argument(
        "--subtract-square-wrp",
        type=Path,
        default=default_dir / "subtract_square.wrp",
    )
    parser.add_argument("--output", type=Path, default=default_dir / "wrp_runner.onnx")
    args = parser.parse_args()
    make_model(args.kernel_chain_wrp, args.subtract_square_wrp, args.output)


if __name__ == "__main__":
    main()
