#!/usr/bin/env python3

# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Create a Triton ensemble model repository from a LEAPP YAML config.

This creates a directory layout like this:

    model_repo/
      <model_a>/
        config.pbtxt          (onnxruntime_onnx)
        1/model.onnx
      <model_b>/
        ...
      ensemble/
        config.pbtxt          (ensemble scheduling)
        1/
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import shutil

import onnx
import yaml

# Mapping from LEAPP dtype strings to Triton data type strings.
DTYPE_TO_TRITON = {
    "float32": "TYPE_FP32",
    "float64": "TYPE_FP64",
    "float16": "TYPE_FP16",
    "int8": "TYPE_INT8",
    "int16": "TYPE_INT16",
    "int32": "TYPE_INT32",
    "int64": "TYPE_INT64",
    "uint8": "TYPE_UINT8",
    "uint16": "TYPE_UINT16",
    "uint32": "TYPE_UINT32",
    "uint64": "TYPE_UINT64",
    "bool": "TYPE_BOOL",
}

_TENSOR_BLOCK_TEMPLATE = """\
{direction} [
  {{
    name: "{name}"
    data_type: {data_type}
    dims: [ {dims} ]
  }}
]
"""

_MAP_ENTRY_TEMPLATE = """\
      {map_type} {{
        key: "{key}"
        value: "{value}"
      }}"""

_STEP_TEMPLATE = """\
    {{
      model_name: "{model_name}"
      model_version: -1
{mappings}
    }}"""


def _generate_ensemble_config(
    inputs: list[dict],
    outputs: list[dict],
    steps: list[dict],
    dynamic_batch: bool,
) -> str:
    """Generate a Triton config.pbtxt for an ensemble model."""
    def _format_dims(shape: list[int]) -> str:
        dims = list(shape)
        if dynamic_batch:
            dims[0] = -1
        return ", ".join(str(d) for d in dims)

    def _blocks(direction: str, tensors: list[dict]) -> str:
        return "".join(
            _TENSOR_BLOCK_TEMPLATE.format(
                direction=direction,
                name=t["name"],
                data_type=DTYPE_TO_TRITON.get(t.get("dtype", "float32"), "TYPE_FP32"),
                dims=_format_dims(t["shape"]),
            )
            for t in tensors
        )

    header = (
        'name: "ensemble"\n'
        'platform: "ensemble"\n'
        "max_batch_size: 0\n\n"
        + _blocks("input", inputs)
        + _blocks("output", outputs)
    )

    step_strs = []
    for step in steps:
        mappings = []
        for key, value in step["input_map"].items():
            mappings.append(
                _MAP_ENTRY_TEMPLATE.format(map_type="input_map", key=key, value=value)
            )
        for key, value in step["output_map"].items():
            mappings.append(
                _MAP_ENTRY_TEMPLATE.format(map_type="output_map", key=key, value=value)
            )
        step_strs.append(
            _STEP_TEMPLATE.format(
                model_name=step["model_name"],
                mappings="\n".join(mappings),
            )
        )

    # Comma-separate step entries (protobuf text format requires this).
    steps_body = ",\n".join(step_strs)
    return header + f"ensemble_scheduling {{\n  step [\n{steps_body}\n  ]\n}}\n"


def _has_dynamic_batch(onnx_path: Path) -> bool:
    """
    Check if an ONNX model uses a dynamic (symbolic) batch dimension.

    Inspects the first dimension of the first model input. If it has a
    symbolic name (e.g. 'batch_size'), the model uses dynamic batching.
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    for inp in model.graph.input:
        shape = inp.type.tensor_type.shape
        if shape and shape.dim:
            return bool(shape.dim[0].dim_param)
    return False


def _generate_model_config(
    model_name: str,
    inputs: list[dict],
    outputs: list[dict],
    dynamic_batch: bool,
    device: str = "cuda",
) -> str:
    """
    Generate a Triton config.pbtxt for an ONNX model.

    When ``dynamic_batch`` is True, the first dimension (batch) of every
    tensor is replaced with -1 so the config matches the ONNX model's
    dynamic batch dimension.

    When ``device`` is ``"cpu"``, the model is placed on CPU via
    ``instance_group``.  This is useful for small models (e.g. AGILE) that
    are fast enough on CPU and should not compete for GPU with heavier models.
    """
    def _format_dims(shape: list[int]) -> str:
        dims = list(shape)
        if dynamic_batch:
            dims[0] = -1
        return ", ".join(str(d) for d in dims)

    def _blocks(direction: str, tensors: list[dict]) -> str:
        return "".join(
            _TENSOR_BLOCK_TEMPLATE.format(
                direction=direction,
                name=t["name"],
                data_type=DTYPE_TO_TRITON.get(t.get("dtype", "float32"), "TYPE_FP32"),
                dims=_format_dims(t["shape"]),
            )
            for t in tensors
        )

    instance_group = ""
    if device.lower() == "cpu":
        instance_group = "instance_group [{ kind: KIND_CPU }]\n\n"

    return (
        f'name: "{model_name}"\n'
        f'platform: "onnxruntime_onnx"\n'
        f"max_batch_size: 0\n\n"
        + instance_group
        + _blocks("input", inputs)
        + _blocks("output", outputs)
    )


def _resolve_model_path(config_path: Path, model_cfg: dict) -> Path:
    """
    Resolve the ONNX model path from a model config entry.

    If the model_path is relative, it is resolved relative to the config file.
    """
    model_path_str = model_cfg.get("parameters", {}).get("model_path", "")
    model_path = Path(model_path_str)
    if model_path_str and not model_path.is_absolute():
        model_path = (config_path.parent / model_path).resolve()
    return model_path


def _create_model_dir(
    config_path: Path,
    model_name: str,
    output_dir: Path,
    config: dict,
) -> bool:
    """
    Create a single Triton model directory inside a repo.

    Generates the config.pbtxt and copies the ONNX model file. If the ONNX
    model uses a dynamic batch dimension, the first dim in config.pbtxt is
    set to -1 (using YAML shapes for the non-batch dimensions).

    Returns True if the ONNX model uses dynamic batch dimensions.
    """
    model_cfg = config["models"][model_name]

    onnx_model_path = _resolve_model_path(config_path, model_cfg)
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

    model_dir = output_dir / model_name
    version_dir = model_dir / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_model_path, version_dir / "model.onnx")

    # Symlink any ONNX external data files (e.g. model.onnx.data) into the
    # version directory so that the ONNX runtime can find them next to the
    # copied model.onnx.
    for data_file in onnx_model_path.parent.glob(f"{onnx_model_path.name}.*"):
        dest = version_dir / data_file.name
        if dest.is_symlink():
            dest.unlink()
        if not dest.exists():
            dest.symlink_to(data_file.resolve())

    dynamic_batch = _has_dynamic_batch(onnx_model_path)
    device = model_cfg.get("parameters", {}).get("device", "cuda")
    config_pbtxt = _generate_model_config(
        model_name,
        model_cfg.get("inputs", []),
        model_cfg.get("outputs", []),
        dynamic_batch,
        device=device,
    )
    (model_dir / "config.pbtxt").write_text(config_pbtxt)

    return dynamic_batch


# Directory holding the warp python-backend templates (model.py + warp_apic_runtime.py),
# copied verbatim into each warp node's version dir. Shipped as part of this package.
_WARP_TEMPLATE_DIR = Path(__file__).parent / "warp_node"


def _generate_warp_model_config(
    model_name: str,
    inputs: list[dict],
    outputs: list[dict],
) -> str:
    """Generate a Triton config.pbtxt for a Warp (APIC) python-backend node.

    Uses the ``python`` backend (shipped in the Triton bundle) with a ``KIND_GPU`` instance and
    ``FORCE_CPU_ONLY_INPUT_TENSORS:"no"`` so warp↔onnx tensors stay GPU-resident across the
    ensemble (the python backend's DLPack path then hands them to warp zero-copy). Shapes/dtypes
    come from the LEAPP YAML (a ``.wrp`` is not introspectable like an ``.onnx``); static shapes.
    """
    # H3: fail at GENERATION time (not opaquely at serve time) for dtypes the warp runtime
    # (warp_apic_runtime.py) does not support. Keep this set in sync with that module.
    _WARP_SUPPORTED_DTYPES = {
        "float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool",
    }
    for t in list(inputs) + list(outputs):
        dt = t.get("dtype", "float32")
        if dt not in _WARP_SUPPORTED_DTYPES:
            raise ValueError(
                f"Warp node '{model_name}' tensor '{t['name']}' has dtype '{dt}' which the warp "
                f"runtime does not support (supported: {sorted(_WARP_SUPPORTED_DTYPES)})")

    def _blocks(direction: str, tensors: list[dict]) -> str:
        return "".join(
            _TENSOR_BLOCK_TEMPLATE.format(
                direction=direction,
                name=t["name"],
                data_type=DTYPE_TO_TRITON.get(t.get("dtype", "float32"), "TYPE_FP32"),
                dims=", ".join(str(d) for d in t["shape"]),
            )
            for t in tensors
        )

    return (
        f'name: "{model_name}"\n'
        'backend: "python"\n'
        "max_batch_size: 0\n\n"
        + _blocks("input", inputs)
        + _blocks("output", outputs)
        + "instance_group [ { kind: KIND_GPU } ]\n\n"
        'parameters: { key: "FORCE_CPU_ONLY_INPUT_TENSORS" value: { string_value: "no" } }\n'
    )


def _create_warp_model_dir(
    config_path: Path,
    model_name: str,
    output_dir: Path,
    config: dict,
) -> None:
    """Create a Triton python-backend model dir for a Warp (APIC) node.

    Copies the ``.wrp`` + its compiled ``<stem>_modules/`` dir + the ``.warpmeta.json`` sidecar
    (written by leapp's export-side ``save_warp_node``) into ``<node>/1/``, drops the warp
    python-backend templates alongside, and writes a python-backend ``config.pbtxt``. The
    downstream ensemble step builder wires this node by tensor name like any other node.
    """
    model_cfg = config["models"][model_name]
    wrp_path = _resolve_model_path(config_path, model_cfg)
    if not wrp_path.exists():
        raise FileNotFoundError(f"Warp .wrp artifact not found: {wrp_path}")

    version_dir = output_dir / model_name / "1"
    version_dir.mkdir(parents=True, exist_ok=True)

    # The APIC artifact set: <name>.wrp + <name>_modules/ + <name>.warpmeta.json.
    shutil.copy2(wrp_path, version_dir / wrp_path.name)
    modules_dir = wrp_path.with_name(wrp_path.stem + "_modules")
    if modules_dir.is_dir():
        shutil.copytree(modules_dir, version_dir / modules_dir.name, dirs_exist_ok=True)
    meta_path = wrp_path.with_suffix(".warpmeta.json")
    if meta_path.exists():
        shutil.copy2(meta_path, version_dir / meta_path.name)

    # Python-backend model + self-contained warp runtime.
    shutil.copy2(_WARP_TEMPLATE_DIR / "model.py", version_dir / "model.py")
    shutil.copy2(_WARP_TEMPLATE_DIR / "warp_apic_runtime.py", version_dir / "warp_apic_runtime.py")

    config_pbtxt = _generate_warp_model_config(
        model_name, model_cfg.get("inputs", []), model_cfg.get("outputs", []),
    )
    (output_dir / model_name / "config.pbtxt").write_text(config_pbtxt)


@dataclass
class TritonRepoResult:
    """
    Result of creating a Triton ensemble model repository.

    Fields:
    - model_name: Name of the ensemble model (always "ensemble").
    - input_tensor_names: Tensor names as they appear in the ROS TensorList
      (matching InputBuilderNode output). These are the original YAML model
      input names.
    - input_binding_names: Tensor names as they appear in the Triton ensemble
      config.pbtxt. Usually the same as input_tensor_names, but may differ
      when an input name collides with an output name (Triton requires unique
      names across inputs and outputs).
    - output_tensor_names: Tensor names for outputs (same for ROS and Triton).
    - output_binding_names: Output binding names (same as output_tensor_names).
    """

    model_name: str
    input_tensor_names: list[str] = field(default_factory=list)
    input_binding_names: list[str] = field(default_factory=list)
    output_tensor_names: list[str] = field(default_factory=list)
    output_binding_names: list[str] = field(default_factory=list)


def create_triton_model_repo(
    config_path: Path,
    output_dir: Path,
) -> TritonRepoResult:
    """
    Create a Triton ensemble model repository from a YAML config.

    Creates per-model directories with ONNX backends and an ensemble directory
    that chains them together. Works for any number of models (including one).

    ``config_path`` is the path to the YAML configuration file.
    ``output_dir`` is the directory to create the model repository in.

    Returns a TritonRepoResult with model name, tensor names, and binding
    names.
    """
    config = yaml.safe_load(config_path.read_text())
    if "models" not in config:
        raise ValueError(
            f"Config file {config_path} is missing required 'models' section"
        )
    models = config["models"]
    pipeline = config.get("pipeline", {})

    # Create per-model repos with config.pbtxt (ONNX, or Warp/APIC python-backend).
    dynamic_batch = False
    for model_name in models:
        backend = models[model_name].get("parameters", {}).get("backend", "")
        if backend == "warp":
            # Warp (APIC) peer node-kind: emit a python-backend model dir from the .wrp.
            _create_warp_model_dir(config_path, model_name, output_dir, config)
        elif _create_model_dir(config_path, model_name, output_dir, config):
            dynamic_batch = True

    # Parse pipeline connectivity.
    dangling_inputs = pipeline.get("inputs", {})
    dangling_outputs = pipeline.get("outputs", {})
    feedback_connections = pipeline.get("feedback_connections", {})
    data_flow = pipeline.get("data_flow", {})

    # Build name sets.
    dangling_input_names = {
        name for names in dangling_inputs.values() for name in names
    }
    dangling_output_names = {
        name for names in dangling_outputs.values() for name in names
    }
    feedback_target_names = {
        t.split("/", 1)[1]
        for targets in feedback_connections.values()
        for t in targets
    }
    feedback_source_names = {
        key.split("/", 1)[1] for key in feedback_connections
    }

    # Determine ensemble tensor names for data-flow connections.
    # If the source output is also externally visible (dangling or feedback),
    # use its own name; otherwise prefix with _internal_.
    data_flow_tensor_names: dict[str, str] = {}
    data_flow_target_to_tensor: dict[str, str] = {}
    for source_key, targets in data_flow.items():
        source_name = source_key.split("/", 1)[1]
        if source_name in dangling_output_names or source_name in feedback_source_names:
            tensor_name = source_name
        else:
            tensor_name = f"_internal_{source_name}"
        data_flow_tensor_names[source_key] = tensor_name
        for t in targets:
            data_flow_target_to_tensor[t] = tensor_name

    # Compute ensemble I/O (preserving YAML order).
    ensemble_inputs: list[dict] = []
    ensemble_input_names: list[str] = []
    ensemble_outputs: list[dict] = []
    ensemble_output_names: list[str] = []

    for model_name, model_cfg in models.items():
        for inp in model_cfg.get("inputs", []):
            name = inp["name"]
            if name in dangling_input_names or name in feedback_target_names:
                if name not in ensemble_input_names:
                    ensemble_inputs.append(inp)
                    ensemble_input_names.append(name)

        for out in model_cfg.get("outputs", []):
            name = out["name"]
            if name in dangling_output_names or name in feedback_source_names:
                if name not in ensemble_output_names:
                    ensemble_outputs.append(out)
                    ensemble_output_names.append(name)

    # Resolve name collisions between ensemble inputs and outputs.
    # Triton requires unique tensor names across inputs and outputs. When the
    # same name appears in both (e.g. "left_hand" as a state input [1,7] and
    # an action output [1,30,7]), prefix the ensemble input binding name.
    collision_names = set(ensemble_input_names) & set(ensemble_output_names)
    input_rename: dict[str, str] = {
        name: f"_in_{name}" for name in collision_names
    }

    if input_rename:
        # Rename ensemble input tensors in the config (for config.pbtxt).
        # Use shallow copies to avoid mutating the original model config dicts.
        ensemble_inputs = [dict(inp) for inp in ensemble_inputs]
        for inp_tensor in ensemble_inputs:
            if inp_tensor["name"] in input_rename:
                inp_tensor["name"] = input_rename[inp_tensor["name"]]

    # Build ensemble steps.
    # Triton ensemble input_map/output_map convention:
    #   key = model tensor name, value = ensemble tensor name
    steps: list[dict] = []
    for model_name, model_cfg in models.items():
        input_map = {}
        for inp in model_cfg.get("inputs", []):
            ensemble_tensor = data_flow_target_to_tensor.get(
                f"{model_name}/{inp['name']}", inp["name"],
            )
            # Apply input rename if the ensemble tensor was a colliding input.
            ensemble_tensor = input_rename.get(ensemble_tensor, ensemble_tensor)
            input_map[inp["name"]] = ensemble_tensor

        output_map = {
            out["name"]: data_flow_tensor_names.get(
                f"{model_name}/{out['name']}", out["name"],
            )
            for out in model_cfg.get("outputs", [])
        }
        steps.append({
            "model_name": model_name,
            "input_map": input_map,
            "output_map": output_map,
        })

    # Generate and write ensemble config.pbtxt.
    config_pbtxt = _generate_ensemble_config(
        ensemble_inputs, ensemble_outputs, steps, dynamic_batch,
    )
    ensemble_dir = output_dir / "ensemble"
    (ensemble_dir / "1").mkdir(parents=True, exist_ok=True)
    (ensemble_dir / "config.pbtxt").write_text(config_pbtxt)

    # Build binding names (with renames applied for collisions).
    input_binding_names = [
        input_rename.get(name, name) for name in ensemble_input_names
    ]

    return TritonRepoResult(
        model_name="ensemble",
        input_tensor_names=ensemble_input_names,
        input_binding_names=input_binding_names,
        output_tensor_names=ensemble_output_names,
        output_binding_names=ensemble_output_names,
    )


def main():
    """CLI entry point for creating a Triton ensemble model repository."""
    parser = argparse.ArgumentParser(
        description="Create a Triton ensemble model repository from a YAML config.",
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Directory for the model repository. If not specified, creates next to the config.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.config_path.parent / "model_repo"
    result = create_triton_model_repo(args.config_path, output_dir)

    print(f"Created Triton model repository at: {output_dir}")
    print(f"  Model: {result.model_name}")
    print(f"  Input tensors: {result.input_tensor_names}")
    print(f"  Input bindings: {result.input_binding_names}")
    print(f"  Output tensors: {result.output_tensor_names}")
    print(f"  Output bindings: {result.output_binding_names}")


if __name__ == "__main__":
    main()
