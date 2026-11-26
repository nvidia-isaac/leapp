# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Patched capture through native export and the ordinary bundle scheduler."""

from contextlib import nullcontext
import json
import re

import pytest
import torch

import leapp
from leapp.backends.string import StringCaptureError


@pytest.fixture
def tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3, "你好": 4},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        padding_side="left",
        model_input_names=["input_ids", "attention_mask"],
    )


@pytest.fixture
def native_runtime():
    from leapp.text_runtime import resolve_library

    try:
        resolve_library("onnx")
        resolve_library("pt2")
    except FileNotFoundError as error:
        pytest.skip(str(error))


@pytest.fixture
def native_tokenizer():
    from leapp.text_runtime import resolve_library

    try:
        resolve_library("pt2")
    except FileNotFoundError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("backend", ["onnx", "pt2"])
def test_module_level_capture_exports_once_and_replays_changed_inputs(
    tmp_path, tokenizer, backend, native_runtime
):
    """Serialize distinct tokenizer instances and call settings through both backends.

    Equal capture operands must stay independently dynamic. The example test owns
    fresh-process coverage; re-entry and individual operations have separate units.
    """
    from copy import deepcopy
    from tokenizers import models
    from transformers.tokenization_utils_base import PaddingStrategy, TruncationStrategy
    from transformers.utils import TensorType

    alternate = deepcopy(tokenizer)
    alternate.backend_tokenizer.model = models.WordLevel(
        {"[UNK]": 0, "[PAD]": 1, "hello": 3, "world": 2, "你好": 4}, unk_token="[UNK]"
    )
    bounded = dict(
        padding=PaddingStrategy.MAX_LENGTH,
        truncation=TruncationStrategy.LONGEST_FIRST,
        max_length=4,
        pad_to_multiple_of=2,
        return_tensors=TensorType.PYTORCH,
        return_attention_mask=False,
    )

    def pipeline(left, right):
        text = (
            "".join([left, right])
            .strip(None)
            .lstrip("!")
            .rstrip("?")
            .upper()
            .replace("AA", "HELLO")
            .replace("", " ", 1)
            .lower()
        )
        return {
            **tokenizer(text, padding=True, return_tensors="pt"),
            "alternate_ids": alternate(text, padding=True, return_tensors="pt")[
                "input_ids"
            ],
            "bounded_ids": tokenizer(text=text, text_pair=None, **bounded)["input_ids"],
        }

    originals = (str.join, str.lower, type(tokenizer).__call__)
    leapp.start("patch_text", save_path=str(tmp_path))
    try:
        left, right = leapp.annotate.input_tensors(
            "tokenizer", {"left": "HELLO", "right": "HELLO"}
        )
        left.title()
        re.sub(r"[^\w\s]", "", left)  # Unsupported unused calls must be pruned.
        leapp.annotate.output_tensors(
            "tokenizer",
            pipeline(left, right),
            export_with=backend,
            backend_params={"utf8_capacity": 32},
        )
    finally:
        leapp.stop()
    assert (str.join, str.lower, type(tokenizer).__call__) == originals
    assert all(leapp.compile_graph(visualize=False, strict=True).values())
    manager = leapp.InferenceManager(str(tmp_path / "patch_text/patch_text.yaml"))
    try:
        assert manager.inputs == ["tokenizer/left", "tokenizer/right"]
        for a, b in [
            ("WORLD", " HELLO"),
            ("", "你好"),
            ("", ""),
            ("!!!AAA", " world???"),
            ("hello world hello", " world hello"),
        ]:
            actual = manager.run_policy({"tokenizer/left": a, "tokenizer/right": b})
            for name, expected in pipeline(a, b).items():
                torch.testing.assert_close(
                    actual["tokenizer/" + name], expected, rtol=0, atol=0
                )
    finally:
        manager.close()


@pytest.mark.parametrize("backend", ["pt2", "onnx"])
def test_tokenizer_output_connects_to_existing_tensor_backend(
    tmp_path, tokenizer, backend, native_runtime
):
    """Cover interleaved capture and native outputs inheriting variable-axis metadata.

    Contract units validate bounds; this checks actual native-tokenizer lengths
    reach a serialized tensor consumer instead of freezing the capture width.
    """
    axis = leapp.Axis(variable=leapp.Variable("tokens", min=2, max=16))
    leapp.start("bridge", save_path=str(tmp_path), global_patching=False)
    pixels = torch.arange(3.0)
    try:
        text = leapp.annotate.input_tensors("tokenizer", {"text": "hello world"})
        # GR00T performs image tensor work between prompt creation and tokenization.
        image = leapp.annotate.input_tensors("image", {"pixels": pixels})
        image = leapp.annotate.output_tensors(
            "image", {"pixels": image + 1}, export_with="jit"
        )
        encoded = tokenizer(text, padding=True, return_tensors="pt")
        ids, mask = leapp.annotate.output_tensors(
            "tokenizer",
            [
                leapp.TensorSemantics(name, value, axes=[None, axis])
                for name, value in encoded.items()
            ],
            export_with=backend,
            backend_params={"utf8_capacity": 128},
        )
        ids, mask, image = leapp.annotate.input_tensors(
            "tensor", {"ids": ids, "mask": mask, "pixels": image}
        )
        result = (ids * mask).sum(dim=1).float() / ids.shape[1] + image.sum()
        leapp.annotate.output_tensors("tensor", {"result": result}, export_with=backend)
    finally:
        leapp.stop()
    assert all(leapp.compile_graph(visualize=False, strict=True).values())
    manager = leapp.InferenceManager(str(tmp_path / "bridge/bridge.yaml"))
    try:
        assert set(manager.inputs) == {"tokenizer/text", "image/pixels"}
        for text in ("world hello", "你好 hello world", "hello " * 16):
            actual = manager.run_policy(
                {"tokenizer/text": text, "image/pixels": pixels}
            )
            ids = tokenizer(text, padding=True, return_tensors="pt")["input_ids"]
            expected = ids.sum(dim=1) / ids.shape[1] + (pixels + 1).sum()
            torch.testing.assert_close(
                actual["tensor/result"], expected.to(actual["tensor/result"].device)
            )
        with pytest.raises(ValueError, match="tokens"):
            manager.run_policy(
                {"tokenizer/text": "hello " * 17, "image/pixels": pixels}
            )
    finally:
        manager.close()


@pytest.mark.parametrize("mutate", [False, True], ids=["direct", "in-place"])
def test_reentry_replays_tokenizer_outputs_through_mutation(
    tmp_path, tokenizer, native_tokenizer, mutate
):
    """Re-entry runs eagerly; the first capture replays changed input through FX."""
    from leapp.backends.string.capture import plain
    from leapp.export_manager import ExportManager

    samples = ("hello", "world hello")
    expected = [tokenizer(text, return_tensors="pt")["input_ids"] for text in samples]
    leapp.start("origins", save_path=str(tmp_path), global_patching=False)
    try:
        for index, sample in enumerate(samples):
            text = leapp.annotate.input_tensors("tokens", {"text": sample})
            assert (type(text) is str) == bool(index)
            value = tokenizer(text, return_tensors="pt")["input_ids"]
            if mutate:
                value.add_(10)
            torch.testing.assert_close(
                plain(value), expected[index] + (10 if mutate else 0)
            )
            leapp.annotate.output_tensors(
                "tokens", {"ids": value}, backend_params={"utf8_capacity": 32}
            )
    finally:
        leapp.stop()
    node = ExportManager().nodes["tokens"]
    fresh = "你好 world hello"
    actual = node.m(*node.encode_inputs({"text": fresh}))
    expected = tokenizer(fresh, return_tensors="pt")["input_ids"]
    torch.testing.assert_close(actual, expected + (10 if mutate else 0))


@pytest.mark.parametrize("change", ["dtype", "rank", "array-kind", "extent"])
def test_reentry_validates_unused_numeric_inputs(tmp_path, change):
    """Pruning an unused input must not remove its capture-time type contract.

    Public re-entry checks the original declaration even after FX has removed
    the unused string and numeric placeholders; same-rank extents may vary.
    """
    import numpy as np

    changed = {
        "dtype": torch.ones(2, dtype=torch.float64),
        "rank": torch.ones(1, 2),
        "array-kind": np.ones(2, dtype=np.float32),
        "extent": torch.ones(3),
    }[change]

    def record(unused):
        _, _, value = leapp.annotate.input_tensors(
            "mixed", {"text": "unused", "unused": unused, "value": torch.ones(2)}
        )
        return leapp.annotate.output_tensors("mixed", {"result": value + 1})

    leapp.start("input_schema", save_path=str(tmp_path), global_patching=False)
    try:
        record(torch.ones(2))
        with (
            nullcontext()
            if change == "extent"
            else pytest.raises(StringCaptureError, match="changed its input schema")
        ):
            torch.testing.assert_close(record(changed), torch.full((2,), 2.0))
    finally:
        leapp.stop()


def test_stop_restores_methods_after_application_failure(tmp_path):
    """Cover inferred dispatch rejection/recovery and lifecycle patch restoration.

    Unit patch tests do not exercise the public manager's input-type selection.
    """
    original = str.join
    leapp.start("incomplete", save_path=str(tmp_path), global_patching=False)
    try:
        with pytest.raises(StringCaptureError, match="scalar strings or tensors"):
            leapp.annotate.input_tensors("invalid", {"image": 1, "text": "hello"})
        assert str.join is original
        text = leapp.annotate.input_tensors("text", {"text": "hello"})
        assert str.join is not original
        with pytest.raises(StringCaptureError):
            bool(text)
    finally:
        leapp.stop()
    assert str.join is original
    assert leapp.annotate.input_tensors("passthrough", {"text": ""}) == ""
    image = torch.ones(1)
    raw_image, raw_text = leapp.annotate.input_tensors(
        "passthrough_mixed", {"image": image, "text": "hello"}
    )
    assert raw_image is image and raw_text == "hello"
    with pytest.raises(Exception, match="never completed"):
        leapp.compile_graph(visualize=False)


@pytest.mark.parametrize("backend", ["onnx", "pt2"])
def test_unused_string_operations_need_no_native_runtime(
    tmp_path, monkeypatch, backend
):
    """Pruning must remove dead text and native dependencies before backend loading.

    Unlike capture-only units, this checks automatic string dispatch from the
    public method decorator, its exporter selection, and serialized replay with
    neither native adapter installed.
    """
    import leapp.text_runtime

    def unavailable(*args, **kwargs):
        raise AssertionError("Pruned text operations requested a native runtime")

    monkeypatch.setattr(leapp.text_runtime, "resolve_library", unavailable)

    @leapp.annotate.method(node_name="mixed", export_with=backend)
    def compute(text, value):
        text.title()
        return {"result": value + 2}

    leapp.start("pruned_text", save_path=str(tmp_path), global_patching=False)
    try:
        compute("hello", torch.arange(3.0))
    finally:
        leapp.stop()
    assert all(
        leapp.compile_graph(visualize=False, validate=True, strict=True).values()
    )
    manager = leapp.InferenceManager(str(tmp_path / "pruned_text/pruned_text.yaml"))
    try:
        assert manager.inputs == ["mixed/value"]
        actual = manager.run_policy({"mixed/value": torch.tensor([5.0, 7.0, 11.0])})
        torch.testing.assert_close(
            actual["mixed/result"], torch.tensor([7.0, 9.0, 13.0])
        )
    finally:
        manager.close()


def test_string_tensor_input_output_name_overlap_rejects_before_export(tmp_path):
    """Reject an ambiguous bundle interface through public capture and compilation.

    A shared ONNX rename must not leave text boundary samples and metadata with
    different port names; this failure needs neither native tokenization nor export.
    """
    leapp.start("overlap", save_path=str(tmp_path), global_patching=False)
    try:
        _, value = leapp.annotate.input_tensors(
            "mixed", {"text": "unused", "value": torch.ones(1)}
        )
        leapp.annotate.output_tensors("mixed", {"value": value + 1}, export_with="onnx")
    finally:
        leapp.stop()
    with pytest.raises(ValueError, match="input/output names must be distinct"):
        leapp.compile_graph(visualize=False, validate=True, strict=True)


@pytest.mark.parametrize(
    "operation,use",
    [
        (lambda text: text.title(), "direct"),
        (lambda text: re.sub("unregistered", "", text), "unused"),
        (lambda text: re.sub("unregistered", "", text), "direct"),
        (lambda text: re.sub("unregistered", "", text), "downstream"),
        (lambda text: re.sub("!", "", text, count=1), "unused"),
        (lambda text: re.sub("!", "", text, count=1), "direct"),
        (lambda text: re.sub("!", "", text, count=1), "downstream"),
        (json.dumps, "unused"),
        (json.dumps, "direct"),
        (json.dumps, "downstream"),
        (lambda text: text.split(), "unused"),
        (lambda text: text.split(), "direct"),
        (lambda text: text.split()[0], "downstream"),
        (lambda text: re.compile("!").sub("", text), "direct"),
        (lambda text: text.strip(text), "direct"),
        (lambda text: text.replace(text, ""), "direct"),
        (lambda text: "hello".replace("h", text), "direct"),
    ],
    ids=[
        "method",
        "unused-unregistered-regex",
        "direct-unregistered-regex",
        "downstream-unregistered-regex",
        "unused-rejected-options",
        "direct-rejected-options",
        "downstream-rejected-options",
        "unused-json",
        "json",
        "downstream-json",
        "unused-split",
        "split",
        "downstream-split",
        "compiled-regex",
        "dynamic-strip-set",
        "dynamic-replace-pattern",
        "dynamic-replacement",
    ],
)
def test_unsupported_dependencies_are_pruned_or_rejected_by_fx(
    tmp_path, tokenizer, operation, use, native_tokenizer
):
    """Check string/nonstring failures across capture, pruning and the FX bridge.

    Unused calls must allow changed-input replay; direct and downstream uses
    must reach lowering rejection rather than becoming capture-time constants.
    """
    from leapp.leapp import _MANAGER

    def remove_bang(text, *, replacement, count, flags):
        if replacement or count or flags:
            raise ValueError("unsupported regex options")
        codes, valid = text.codepoints()
        return text.filter(valid & (codes != ord("!")))

    rejection = (
        nullcontext()
        if use == "unused"
        else pytest.raises(
            (ValueError, StringCaptureError, NotImplementedError),
            match="unsupported|Unregistered",
        )
    )
    with rejection:
        patch = leapp.regex_sub_patch({"!": remove_bang})
        leapp.start("unsupported", save_path=str(tmp_path), patching=[patch])
        try:
            text = leapp.annotate.input_tensors("tokenizer", {"text": "hello"})
            result = operation(text)
            if use == "unused":
                result = text.lower()
            elif use == "downstream":
                result = result.lower()
            encoded = tokenizer(result, padding=True, return_tensors="pt")
            leapp.annotate.output_tensors("tokenizer", dict(encoded.items()))
        finally:
            leapp.stop()
        node = _MANAGER.nodes["tokenizer"]
        ids, mask = node.m(*node.encode_inputs({"text": "WORLD"}))
        assert ids.tolist() == [[3]] and mask.tolist() == [[1]]


def test_user_string_lowering_rejects_wrong_exported_values(
    tmp_path, tokenizer, native_runtime
):
    """Check callback contracts and exported parity against the eager reference.

    Unit tests cannot catch a valid tensor lowering that exports successfully
    but changes the application's token IDs or sequence length.
    """

    def incorrect_lowering(string, *, replacement, count, flags):
        return string

    original = re.sub
    patch = leapp.regex_sub_patch({r"[^\w\s]": incorrect_lowering})
    leapp.start("bad_patch", save_path=str(tmp_path), patching=[patch])
    try:
        text = leapp.annotate.input_tensors("tokenizer", {"text": "hello!"})
        encoded = tokenizer(
            re.sub(r"[^\w\s]", "", text), padding=True, return_tensors="pt"
        )
        leapp.annotate.output_tensors(
            "tokenizer",
            dict(encoded.items()),
            export_with="onnx",
            backend_params={"utf8_capacity": 16},
        )
    finally:
        leapp.stop()
    assert re.sub is original
    with pytest.raises(Exception, match="Model validation failed"):
        leapp.compile_graph(visualize=False, validate=True, strict=True)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_mixed_node_traces_tokenizer_and_tensor_computation(
    tmp_path, tokenizer, device, monkeypatch, native_runtime
):
    """Exercise one mixed FX graph and serialization across both numeric edges.

    Unlike patch units, this catches frozen tokenizer sizes, lost model weights,
    and confusing connected numeric ports with externally supplied UTF-8 ports.
    """
    import onnxruntime as ort

    provider = "CPUExecutionProvider"  # String models always run on CPU.
    original_session = ort.InferenceSession

    def profiled_session(path, options, **kwargs):
        options.enable_profiling = True
        options.profile_file_prefix = str(tmp_path / "string-runtime-profile")
        return original_session(path, options, **kwargs)

    monkeypatch.setattr(ort, "InferenceSession", profiled_session)

    projection = torch.nn.Linear(2, 2).eval().to(device)
    with torch.no_grad():
        projection.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        projection.bias.copy_(torch.tensor([0.25, -0.5]))

    def compute(text, pixels, scale):
        visual = pixels * scale
        visual.add_(1)
        visual = visual.sum().reshape(1)
        # Unused unsupported tokenizer dependencies must also be pruned.
        tokenizer(text.title(), padding=True, return_tensors="pt")
        tokens = tokenizer(text.strip().lower(), padding=True, return_tensors="pt")
        ids = tokens["input_ids"]
        features = torch.stack(
            (ids.sum(dim=1).float().to(pixels.device) / (ids.shape[1] + 1), visual),
            dim=1,
        )
        return projection(features), ids

    variable_pixels = leapp.Axis(variable=leapp.Variable("pixels", min=2, max=8))
    leapp.start("mixed", save_path=str(tmp_path), global_patching=False)
    try:
        for text, number in [(" HELLO WORLD ", 1.0), ("你好", 2.0)]:
            pixels = leapp.annotate.input_tensors(
                "image",
                leapp.TensorSemantics(
                    "pixels", torch.arange(3.0, device=device), axes=[variable_pixels]
                ),
            )
            pixels = leapp.annotate.output_tensors(
                "image", {"pixels": pixels + 1}, export_with="jit"
            )
            text, pixels, scale = leapp.annotate.input_tensors(
                "fusion",
                {
                    "text": text,
                    "pixels": pixels,
                    "__leapp_utf8_0_bytes": torch.tensor(number, device=device),
                },
            )
            result, ids = compute(text, pixels, scale)
            result, ids = leapp.annotate.output_tensors(
                "fusion",
                {"result": result, "ids": ids},
                backend_params={"utf8_capacity": 64},
            )
            result = leapp.annotate.input_tensors("consumer", {"result": result})
            leapp.annotate.output_tensors(
                "consumer", {"result": result * 2}, export_with="jit"
            )
    finally:
        leapp.stop()
    assert all(
        leapp.compile_graph(visualize=False, validate=True, strict=True).values()
    )
    manager = leapp.InferenceManager(str(tmp_path / "mixed/mixed.yaml"))
    try:
        runtime = manager.nodes["fusion"].model
        assert runtime.session.get_providers()[0] == provider
        assert set(manager.inputs) == {
            "image/pixels",
            "fusion/text",
            "fusion/__leapp_utf8_0_bytes",
        }
        for text, number in [("world hello 你好", -2.0), ("", 0.0), ("你好", 3.0)]:
            pixels = torch.arange(2.0 if number < 0 else 5.0, device=device) + number
            scale = torch.tensor(number, device=device)
            expected, ids = compute(text, pixels + 1, scale)
            actual = manager.run_policy(
                {
                    "image/pixels": pixels,
                    "fusion/text": text,
                    "fusion/__leapp_utf8_0_bytes": scale,
                }
            )
            torch.testing.assert_close(
                actual["consumer/result"].cpu(), expected.cpu() * 2
            )
            torch.testing.assert_close(
                actual["fusion/ids"].cpu(), ids.cpu(), rtol=0, atol=0
            )
        with open(runtime.session.end_profiling()) as profile:
            events = json.load(profile)
        assert any(
            event.get("args", {}).get("op_name") == "HfTokenizerUtf8"
            and event["args"].get("provider") == "CPUExecutionProvider"
            for event in events
        )
        assert any(
            event.get("args", {}).get("op_name") in ("MatMul", "Gemm", "FusedGemm")
            and event["args"].get("provider") == provider
            for event in events
        )
    finally:
        manager.close()


def test_regex_options_are_captured_per_call(tmp_path, tokenizer, native_runtime):
    """Forward each call's options through capture, export and changed-input replay.

    Unlike direct callback units, this detects lost/defaulted static arguments or
    accidental reuse of the first call's options in the serialized graph.
    """

    def replace_a(text, *, replacement, count, flags):
        codes, valid = text.codepoints()
        selected = valid & (
            (codes == ord("a")) | (bool(flags & re.IGNORECASE) & (codes == ord("A")))
        )
        if count:
            selected = selected & (selected.to(torch.int64).cumsum(1) <= count)
        if not replacement:
            return text.filter(~selected)
        return leapp.Utf8Text(
            torch.where(selected, ord(replacement), text.data),
            text.lengths,
            text.capacity,
            text.scalar,
        )

    def pipeline(text):
        texts = {
            "first": re.sub("a", "h", text, 1, re.IGNORECASE),
            "all": re.sub(pattern="a", repl="w", string=text, flags=re.ASCII),
            "delete": re.sub("a", "", text),
        }
        return {
            name: tokenizer(value, padding=True, return_tensors="pt")["input_ids"]
            for name, value in texts.items()
        }

    patch = leapp.regex_sub_patch({"a": replace_a})
    leapp.start("regex_options", save_path=str(tmp_path), patching=[patch])
    try:
        text = leapp.annotate.input_tensors("tokenizer", {"text": "Aello aorld a"})
        leapp.annotate.output_tensors(
            "tokenizer",
            pipeline(text),
            export_with="pt2",
            backend_params={"utf8_capacity": 32},
        )
    finally:
        leapp.stop()
    assert all(
        leapp.compile_graph(visualize=False, validate=True, strict=True).values()
    )
    manager = leapp.InferenceManager(str(tmp_path / "regex_options/regex_options.yaml"))
    try:
        for sample in ("aello aello Aello", "aorld aorld", "ahello aworld", "你好", ""):
            actual = manager.run_policy({"tokenizer/text": sample})
            for name, value in pipeline(sample).items():
                torch.testing.assert_close(
                    actual[f"tokenizer/{name}"], value, rtol=0, atol=0
                )
    finally:
        manager.close()
