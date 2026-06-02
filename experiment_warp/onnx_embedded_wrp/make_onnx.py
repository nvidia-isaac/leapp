"""Create an ONNX model with two WrpRunner nodes that embed their .wrp bundles.

Each node's APIC bundle (.wrp + _modules/) is packed into a `WRPB` archive and
stored as a `uint8` tensor initializer wired in as the node's last input. The op
reads those bytes at runtime, so the model is self-contained: no `wrp_path`, no
sibling `.wrp` files, no environment variables.

Because the bundle lives in an initializer (not an attribute), ONNX's
external-data mechanism applies. With `--external-data`, large bundles spill into
a sibling `<model>.onnx.data` file (the LEAPP pattern); small ones can stay
inline via `--size-threshold`. ONNX Runtime resolves the external file relative
to the model path, so portability is preserved either way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from generate_wrp import HEIGHT, SMALL_HEIGHT, SMALL_WIDTH, WIDTH
from wrp_bundle import pack_bundle


DOMAIN = "com.nvidia.warp"


def _make_node_and_initializer(
    node_name: str,
    wrp_path: Path,
    data_inputs: list[str],
    outputs: list[str],
    output_shape: str,
) -> tuple[onnx.NodeProto, onnx.TensorProto]:
    archive, wrp_name = pack_bundle(wrp_path)
    bundle_name = f"{node_name}.bundle"

    bundle_initializer = helper.make_tensor(
        name=bundle_name,
        data_type=TensorProto.UINT8,
        dims=[len(archive)],
        vals=archive,
        raw=True,
    )

    # Bundle is the LAST input so the data inputs keep clean 0-based indices.
    node = helper.make_node(
        "WrpRunner",
        inputs=[*data_inputs, bundle_name],
        outputs=outputs,
        domain=DOMAIN,
        name=node_name,
        wrp_name=wrp_name,
        input_names=",".join(data_inputs),
        output_names=",".join(outputs),
        output_shape=output_shape,
    )
    return node, bundle_initializer


def make_model(
    kernel_chain_wrp: Path,
    subtract_square_wrp: Path,
    output_path: Path,
    external_data: bool,
    size_threshold: int,
) -> None:
    kernel_chain_node, kernel_chain_bundle = _make_node_and_initializer(
        "run_kernel_chain_wrp",
        kernel_chain_wrp,
        data_inputs=["a", "b"],
        outputs=["averaged"],
        output_shape=f"{HEIGHT},{WIDTH}",
    )
    subtract_square_node, subtract_square_bundle = _make_node_and_initializer(
        "run_subtract_square_wrp",
        subtract_square_wrp,
        data_inputs=["x", "y"],
        outputs=["squared"],
        output_shape=f"{SMALL_HEIGHT},{SMALL_WIDTH}",
    )

    graph = helper.make_graph(
        [kernel_chain_node, subtract_square_node],
        "wrp_runner_embedded_demo",
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
        initializer=[kernel_chain_bundle, subtract_square_bundle],
    )

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid(DOMAIN, 1),
        ],
        producer_name="onnx_embedded_wrp_prototype",
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
        onnx_mb = output_path.stat().st_size / (1024 * 1024)
        if data_file.exists():
            data_mb = data_file.stat().st_size / (1024 * 1024)
            print(
                f"wrote {output_path} ({onnx_mb:.2f} MB) + {data_file.name} "
                f"({data_mb:.2f} MB, bundles external)"
            )
        else:
            print(
                f"wrote {output_path} ({onnx_mb:.2f} MB); all bundles stayed inline "
                f"(below size_threshold={size_threshold})"
            )
    else:
        onnx.save(model, str(output_path))
        onnx_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"wrote {output_path} ({onnx_mb:.2f} MB, bundles inline)")


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
    parser.add_argument("--output", type=Path, default=default_dir / "wrp_runner_embedded.onnx")
    parser.add_argument(
        "--external-data",
        action="store_true",
        help="Spill bundles into a sibling <model>.onnx.data file (LEAPP pattern).",
    )
    parser.add_argument(
        "--size-threshold",
        type=int,
        default=1024,
        help="Initializers larger than this many bytes go external (with --external-data).",
    )
    args = parser.parse_args()
    make_model(
        args.kernel_chain_wrp,
        args.subtract_square_wrp,
        args.output,
        external_data=args.external_data,
        size_threshold=args.size_threshold,
    )


if __name__ == "__main__":
    main()
