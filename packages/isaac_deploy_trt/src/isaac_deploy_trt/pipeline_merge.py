#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Merge a LEAPP ONNX pipeline into one graph for TensorRTNode (no Triton).

LEAPP export is one ONNX file per node plus YAML ``pipeline.data_flow``.
Isaac ROS ``TensorRTNode`` loads a single ``.plan`` with one TensorList in and
out, so the DAG has to be inlined first. This module prefixes each subgraph,
concatenates them, then rewires ``data_flow`` edges (including fan-out).

``feedback_flow`` is not inlined: those tensors stay as extra graph inputs
(the previous tick's values), matching how Triton exposes them on the ensemble
boundary.

Name collisions between pipeline inputs and outputs use the same convention as
Triton model repos: the input binding is renamed ``_in_<name>``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import onnx
import yaml
from onnx import compose, helper

from isaac_deploy_trt.prepare import needs_tensorrt_rewrite
from isaac_deploy_trt.surgery import rewrite_onnx_for_tensorrt

logger = logging.getLogger(__name__)


def _split_ref(ref: str) -> Tuple[str, str]:
    model, sep, tensor = ref.partition("/")
    if not sep or not model or not tensor:
        raise ValueError(f"Expected 'model/tensor' reference, got {ref!r}")
    return model, tensor


def _prefixed(model: str, tensor: str) -> str:
    return f"{model}__{tensor}"


def _topo_models(model_names: Sequence[str], data_flow: Mapping[str, Any]) -> List[str]:
    """Stable topological order of LEAPP model names."""
    incoming: Dict[str, set[str]] = {name: set() for name in model_names}
    for source, targets in (data_flow or {}).items():
        src_model, _ = _split_ref(source)
        for target in targets or []:
            dst_model, _ = _split_ref(target)
            if src_model == dst_model:
                continue
            if dst_model in incoming:
                incoming[dst_model].add(src_model)
    ordered: List[str] = []
    remaining = list(model_names)
    while remaining:
        ready = [name for name in remaining if not incoming[name] - set(ordered)]
        if not ready:
            raise ValueError(
                "Pipeline data_flow has a cycle; cannot merge into one ONNX graph"
            )
        name = ready[0]
        remaining.remove(name)
        ordered.append(name)
        for other in remaining:
            incoming[other].discard(name)
    return ordered


def _load_model(path: str) -> onnx.ModelProto:
    """Load only the ONNX protobuf; keep large initializers external."""
    return onnx.load(path, load_external_data=False)


def _iter_graphs(graph: onnx.GraphProto):
    yield graph
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                yield from _iter_graphs(attr.g)
            for subgraph in attr.graphs:
                yield from _iter_graphs(subgraph)


def _iter_tensors(model: onnx.ModelProto):
    for graph in _iter_graphs(model.graph):
        yield from graph.initializer
        for node in graph.node:
            for attr in node.attribute:
                if attr.HasField("t"):
                    yield attr.t
                yield from attr.tensors


def _external_locations(model: onnx.ModelProto) -> set[str]:
    locations = set()
    for tensor in _iter_tensors(model):
        for entry in tensor.external_data:
            if entry.key == "location":
                locations.add(entry.value)
    return locations


def _copy_graph_fields(dst: onnx.GraphProto, src: onnx.GraphProto) -> None:
    dst.node.extend(src.node)
    dst.initializer.extend(src.initializer)
    dst.sparse_initializer.extend(src.sparse_initializer)
    dst.input.extend(src.input)
    dst.output.extend(src.output)
    dst.value_info.extend(src.value_info)


def _replace_tensor_name(graph: onnx.GraphProto, old: str, new: str) -> None:
    if old == new:
        return
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new
        for i, name in enumerate(node.output):
            if name == old:
                node.output[i] = new
    for collection in (
        graph.input,
        graph.output,
        graph.value_info,
        graph.initializer,
        graph.sparse_initializer,
    ):
        for item in collection:
            if item.name == old:
                item.name = new


def _remove_graph_input(graph: onnx.GraphProto, name: str) -> None:
    keep = [item for item in graph.input if item.name != name]
    del graph.input[:]
    graph.input.extend(keep)


def _remove_graph_output(graph: onnx.GraphProto, name: str) -> None:
    keep = [item for item in graph.output if item.name != name]
    del graph.output[:]
    graph.output.extend(keep)


def _value_info_by_name(graph: onnx.GraphProto, name: str):
    for item in list(graph.input) + list(graph.output) + list(graph.value_info):
        if item.name == name:
            return item
    return None


@dataclass
class MergedPipelineResult:
    """Result of ``merge_leapp_onnx_pipeline``."""

    model: onnx.ModelProto
    input_tensor_names: List[str] = field(default_factory=list)
    output_tensor_names: List[str] = field(default_factory=list)
    input_rename: Dict[str, str] = field(default_factory=dict)
    external_data_sources: Dict[str, str] = field(default_factory=dict)
    yaml_spec: Dict[str, Any] = field(default_factory=dict)
    rewritten_models: List[str] = field(default_factory=list)


def merge_leapp_onnx_pipeline(
    yaml_path: str,
    *,
    model_dir: Optional[str] = None,
    rewrite: bool = False,
) -> MergedPipelineResult:
    """Build one ONNX graph from a LEAPP export YAML and sibling ``.onnx`` files.

    ``model_dir`` defaults to the YAML's directory. Relative ``parameters.model_path``
    values are resolved against that directory.

    If ``rewrite`` is true, apply TensorRT GraphSurgeon edits (UINT8 Cast,
    Resize antialias, rank>8 Reshape) to each subgraph before inlining.
    """
    yaml_path = os.path.abspath(yaml_path)
    model_dir = os.path.abspath(model_dir or os.path.dirname(yaml_path))
    with open(yaml_path, encoding="utf-8") as handle:
        spec = yaml.safe_load(handle) or {}
    models_cfg = spec.get("models")
    pipeline = spec.get("pipeline") or {}
    if not isinstance(models_cfg, dict) or not models_cfg:
        raise ValueError(f"{yaml_path} has no models section")

    data_flow = pipeline.get("data_flow") or {}
    feedback_flow = pipeline.get("feedback_flow") or {}
    graph_inputs = pipeline.get("inputs") or {}
    graph_outputs = pipeline.get("outputs") or {}

    model_names = list(models_cfg.keys())
    order = _topo_models(model_names, data_flow)

    prefixed: Dict[str, onnx.ModelProto] = {}
    opsets: List[int] = []
    ir_version = 0
    external_data_sources: Dict[str, str] = {}
    rewritten_models: List[str] = []
    for name, cfg in models_cfg.items():
        rel = ((cfg or {}).get("parameters") or {}).get("model_path", "")
        if not rel:
            raise ValueError(f"Model {name!r} has no parameters.model_path")
        path = rel if os.path.isabs(rel) else os.path.join(model_dir, rel)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ONNX for model {name!r} not found: {path}")
        logger.info("Loading %s from %s", name, path)
        raw = _load_model(path)
        if rewrite and needs_tensorrt_rewrite(raw):
            logger.info("Rewriting %s for TensorRT before merge", name)
            raw = rewrite_onnx_for_tensorrt(raw)
            rewritten_models.append(name)
        for location in _external_locations(raw):
            source = os.path.abspath(os.path.join(os.path.dirname(path), location))
            previous = external_data_sources.get(location)
            if previous is not None and previous != source:
                raise ValueError(
                    f"External-data location {location!r} is ambiguous: "
                    f"{previous} vs {source}. Rename one source artifact first."
                )
            if not os.path.isfile(source):
                raise FileNotFoundError(
                    f"External data for model {name!r} not found: {source}"
                )
            external_data_sources[location] = source
        ir_version = max(ir_version, raw.ir_version)
        for imp in raw.opset_import:
            if imp.domain in ("", "ai.onnx"):
                opsets.append(imp.version)
        prefix = f"{name}__"
        try:
            prefixed[name] = compose.add_prefix(raw, prefix=prefix, inplace=False)
        except TypeError:
            prefixed[name] = compose.add_prefix(raw, prefix)

    combined = helper.make_graph(
        nodes=[],
        name="leapp_merged_pipeline",
        inputs=[],
        outputs=[],
    )
    for name in order:
        _copy_graph_fields(combined, prefixed[name].graph)

    for source, targets in data_flow.items():
        src_model, src_tensor = _split_ref(source)
        producer = _prefixed(src_model, src_tensor)
        if _value_info_by_name(combined, producer) is None:
            raise ValueError(f"data_flow source {source} not in merged graph")
        for target in targets or []:
            dst_model, dst_tensor = _split_ref(target)
            consumer = _prefixed(dst_model, dst_tensor)
            _replace_tensor_name(combined, consumer, producer)
            _remove_graph_input(combined, producer)

    for source, targets in feedback_flow.items():
        src_model, src_tensor = _split_ref(source)
        producer = _prefixed(src_model, src_tensor)
        for target in targets or []:
            dst_model, dst_tensor = _split_ref(target)
            consumer = _prefixed(dst_model, dst_tensor)
            if _value_info_by_name(combined, consumer) is None:
                raise ValueError(f"feedback_flow target {target} not in merged graph")
            if _value_info_by_name(combined, producer) is None:
                raise ValueError(f"feedback_flow source {source} not in merged graph")

    pipeline_input_names: List[str] = []
    for model_name, tensors in graph_inputs.items():
        for tensor in tensors:
            pipeline_input_names.append(_prefixed(model_name, tensor))

    for targets in feedback_flow.values():
        for target in targets or []:
            dst_model, dst_tensor = _split_ref(target)
            name = _prefixed(dst_model, dst_tensor)
            if name not in pipeline_input_names:
                pipeline_input_names.append(name)

    pipeline_output_names: List[str] = []
    for model_name, tensors in graph_outputs.items():
        for tensor in tensors:
            pipeline_output_names.append(_prefixed(model_name, tensor))
    for source in feedback_flow:
        src_model, src_tensor = _split_ref(source)
        name = _prefixed(src_model, src_tensor)
        if name not in pipeline_output_names:
            pipeline_output_names.append(name)

    wanted_inputs = set(pipeline_input_names)
    wanted_outputs = set(pipeline_output_names)
    for item in list(combined.input):
        if item.name not in wanted_inputs:
            _remove_graph_input(combined, item.name)
    for item in list(combined.output):
        if item.name not in wanted_outputs:
            _remove_graph_output(combined, item.name)

    input_rename: Dict[str, str] = {}
    output_originals = {name.split("__", 1)[1] for name in pipeline_output_names}
    final_inputs: List[str] = []
    for prefixed_name in pipeline_input_names:
        original = prefixed_name.split("__", 1)[1]
        if original in output_originals:
            public = f"_in_{original}"
            input_rename[original] = public
        else:
            public = original
        _replace_tensor_name(combined, prefixed_name, public)
        final_inputs.append(public)

    final_outputs: List[str] = []
    for prefixed_name in pipeline_output_names:
        original = prefixed_name.split("__", 1)[1]
        current = prefixed_name
        if _value_info_by_name(combined, current) is None:
            current = original if _value_info_by_name(combined, original) else current
        if current != original and _value_info_by_name(combined, current) is not None:
            _replace_tensor_name(combined, current, original)
        final_outputs.append(original)

    info = {
        item.name: item
        for item in list(combined.input) + list(combined.output) + list(combined.value_info)
    }
    del combined.input[:]
    del combined.output[:]
    for name in final_inputs:
        item = info.get(name)
        if item is None:
            raise ValueError(f"Merged graph missing pipeline input {name}")
        combined.input.append(item)
    for name in final_outputs:
        item = info.get(name)
        if item is None:
            raise ValueError(f"Merged graph missing pipeline output {name}")
        combined.output.append(item)

    opset = max(opsets) if opsets else 17
    model = helper.make_model(
        combined,
        opset_imports=[helper.make_opsetid("", opset)],
        ir_version=ir_version or 10,
    )
    model.producer_name = "isaac_deploy_trt.merge_leapp_onnx_pipeline"

    yaml_out = {
        "models": {
            "pipeline": {
                "inputs": [
                    {"name": name, "dtype": "float32", "shape": [], "type": "tensor"}
                    for name in final_inputs
                ],
                "outputs": [
                    {"name": name, "dtype": "float32", "shape": [], "type": "tensor"}
                    for name in final_outputs
                ],
                "parameters": {
                    "model_path": "pipeline.onnx",
                    "backend": "onnx",
                },
            }
        },
        "pipeline": {
            "data_flow": {},
            "feedback_flow": {},
            "inputs": {"pipeline": list(final_inputs)},
            "outputs": {"pipeline": list(final_outputs)},
        },
        "tensor_rt_node": {
            "input_tensor_names": list(final_inputs),
            "output_tensor_names": list(final_outputs),
            "note": (
                "Single-engine graph for TensorRTNode. Colliding input names "
                "are _in_<name>. Compile pipeline.onnx with trtexec, then set "
                "engine_file_path / input_tensor_names / output_tensor_names."
            ),
        },
    }
    meta_in = {}
    meta_out = {}
    for model_name, cfg in models_cfg.items():
        for item in (cfg or {}).get("inputs") or []:
            meta_in[(model_name, item.get("name"))] = item
        for item in (cfg or {}).get("outputs") or []:
            meta_out[(model_name, item.get("name"))] = item

    def _enrich(entries: List[dict], kind: str) -> None:
        for entry in entries:
            public = entry["name"]
            original = public[4:] if public.startswith("_in_") else public
            source = meta_in if kind == "inputs" else meta_out
            for (_mn, tname), item in source.items():
                if tname == original:
                    entry["dtype"] = item.get("dtype", entry["dtype"])
                    entry["shape"] = list(item.get("shape") or [])
                    if item.get("kind"):
                        entry["kind"] = item["kind"]
                    if item.get("element_names"):
                        entry["element_names"] = item["element_names"]
                    break

    _enrich(yaml_out["models"]["pipeline"]["inputs"], "inputs")
    _enrich(yaml_out["models"]["pipeline"]["outputs"], "outputs")

    return MergedPipelineResult(
        model=model,
        input_tensor_names=final_inputs,
        output_tensor_names=final_outputs,
        input_rename=input_rename,
        external_data_sources=external_data_sources,
        yaml_spec=yaml_out,
        rewritten_models=rewritten_models,
    )


def save_merged_pipeline(
    result: MergedPipelineResult,
    onnx_path: str,
    *,
    yaml_path: Optional[str] = None,
) -> str:
    """Write ``pipeline.onnx`` and sibling YAML. Returns the YAML path."""
    onnx_path = os.path.abspath(onnx_path)
    parent = os.path.dirname(onnx_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    onnx.save(result.model, onnx_path)
    for location, source in result.external_data_sources.items():
        destination = os.path.abspath(os.path.join(parent, location))
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if os.path.lexists(destination):
            if os.path.realpath(destination) == os.path.realpath(source):
                continue
            os.remove(destination)
        os.symlink(source, destination)
    result.yaml_spec["models"]["pipeline"]["parameters"]["model_path"] = os.path.basename(
        onnx_path
    )
    out_yaml = yaml_path or os.path.splitext(onnx_path)[0] + ".yaml"
    with open(out_yaml, "w", encoding="utf-8") as handle:
        yaml.safe_dump(result.yaml_spec, handle, sort_keys=False)
    logger.info(
        "Wrote merged ONNX %s and YAML %s; inputs=%s outputs=%s",
        onnx_path,
        out_yaml,
        result.input_tensor_names,
        result.output_tensor_names,
    )
    return out_yaml
