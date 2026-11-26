# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""String capture: CPython patches, tokenizer discovery, and native tokenizer parity."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import re

import numpy as np
import pytest
import torch

from leapp.backends.string import StringCaptureError, utf8
from leapp.backends.string.tokenizer import snapshot, validate_options
from leapp.backends.string.capture import (
    ACTIVE,
    Patches,
    finish_capture,
    observes,
    plain,
    reference,
)


@pytest.mark.parametrize(
    "backend,environment",
    [("onnx", "LEAPP_TEXT_RUNTIME_LIBRARY"), ("pt2", "LEAPP_TEXT_PT2_RUNTIME_LIBRARY")],
)
def test_native_runtime_is_resolved_from_the_deployment_environment(
    tmp_path, monkeypatch, backend, environment
):
    from leapp.text_runtime import library_name, resolve_library

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv(environment, raising=False)
    installed = tmp_path / "leapp/text-runtime/release" / library_name(backend)
    with pytest.raises(FileNotFoundError, match="text_runtime --build"):
        resolve_library(backend)
    installed.parent.mkdir(parents=True)
    installed.touch()
    assert resolve_library(backend) == installed.resolve()

    override = tmp_path / library_name(backend)
    monkeypatch.setenv(environment, str(override))
    with pytest.raises(FileNotFoundError, match=environment):
        resolve_library(backend)
    override.touch()
    assert resolve_library(backend) == override.resolve()


def test_external_weights_preserve_text_metadata(tmp_path, monkeypatch):
    """Exercise the ordinary large-model save path without embedded libraries."""
    import numpy as np
    import onnx
    from leapp.backends.export_backend import SimplifiedONNXProgram
    from leapp.backends.string import FORMAT

    weights = np.arange(512, dtype=np.float32)
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [], "external", [], [], [onnx.numpy_helper.from_array(weights, "weights")]
        )
    )
    definition = {"format": FORMAT, "text_inputs": []}
    onnx.helper.set_model_props(model, {"leapp.string": json.dumps(definition)})
    source = tmp_path / "source.onnx"
    onnx.save(model, source)
    destination = tmp_path / "saved" / "text.onnx"
    program = SimplifiedONNXProgram(source)
    monkeypatch.setattr(program, "_get_source_size", lambda: 3 * 1024**3)
    try:
        program.save(destination)
    finally:
        program.close()
    restored = onnx.load(destination, load_external_data=False)
    tensors = {tensor.name: tensor for tensor in restored.graph.initializer}
    assert set(tensors) == {"weights"}
    assert tensors["weights"].data_location == onnx.TensorProto.EXTERNAL
    assert not tensors["weights"].raw_data
    assert (destination.parent / "text.onnx.data").stat().st_size == weights.nbytes
    properties = {item.key: item.value for item in restored.metadata_props}
    assert json.loads(properties["leapp.string"]) == definition
    loaded = onnx.load(destination)
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(loaded.graph.initializer[0]), weights
    )


@contextmanager
def capture(*, resources=None, node=None, **inputs):
    from leapp.backends.string.node import StringNode

    node = StringNode("text") if node is None else node
    node.begin(inputs)
    patches = Patches(node, {} if resources is None else resources)
    originals = dict(str.__dict__)
    original_sub = re.sub
    original_pattern_sub = re.Pattern.sub
    original_encoder = json.JSONEncoder.encode
    patches.install()
    token = ACTIVE.set(node)
    try:
        yield node
    finally:
        ACTIVE.reset(token)
        node.end()
        patches.restore()
        assert re.sub is original_sub
        assert re.Pattern.sub is original_pattern_sub
        assert json.JSONEncoder.encode is original_encoder
        assert all(str.__dict__[name] is value for name, value in originals.items())


@pytest.mark.parametrize("sample", ["", "\x00你好🤖"], ids=["empty", "multibyte-nul"])
def test_string_uses_shared_carrier_and_structure_helpers(sample):
    from leapp.leapp_graph.datatypes import TracedData
    from leapp.backends.string import utf8

    with capture(text=sample) as node:
        value = node._text_inputs["text"]
        assert isinstance(value, TracedData) and value.context_obj is node
        nested = {value: [value, (value, 7)]}
        assert TracedData.unwrap_traced_data(nested) == {value: [sample, (sample, 7)]}
        assert TracedData.unwrap_traced_data(nested, map_keys=True) == {
            sample: [sample, (sample, 7)]
        }
        assert TracedData.extract_proxy(nested, map_keys=True) == {
            value.proxy: [value.proxy, (value.proxy, 7)]
        }
        packed = value.tensor
        text = utf8.unpack(packed)
        assert utf8.decode(text.data, text.lengths) == [sample]
        packed.fill_(255)
        assert not torch.equal(value.tensor, packed)
        assert reference(node, value._new("independent", proxy=None)) is None


@pytest.mark.parametrize("same_node", [True, False], ids=["same-node", "other-node"])
def test_reusing_string_in_another_capture_rejects(same_node):
    with capture(text="hello") as first:
        saved = first._text_inputs["text"]
    with capture(text="world", node=first if same_node else None):
        with pytest.raises(
            StringCaptureError, match="another capture|reusing captured"
        ):
            saved.lower()


@pytest.mark.parametrize(
    "operation,expected",
    [
        (lambda s: s.strip().lower(), "hello world"),
        (lambda s: str.lower(s), " hello world "),
        (lambda s: s.replace("HELLO", "你好", 1), " 你好 WORLD "),
        (lambda s: "|".join([s, "🤖"]), " HELLO WORLD |🤖"),
        (lambda s: s.join(["a", "b"]), "a HELLO WORLD b"),
        (lambda s: "prefix" + s + "suffix", "prefix HELLO WORLD suffix"),
        (lambda s: re.sub(r"[^\w\s]", "", s), " HELLO WORLD "),
    ],
)
def test_methods_and_literal_receivers_preserve_eager_behavior(operation, expected):
    with capture(text=" HELLO WORLD ") as node:
        result = operation(node._text_inputs["text"])
        assert plain(result) == expected
        assert observes(node, result)
        finish_capture(node, {"result": result})


def test_user_regex_replacement_takes_precedence_over_automatic_capture():
    from leapp.leapp_graph.datatypes.patching import FunctionPatch, TracingPatcher

    original = re.sub
    patcher = TracingPatcher()
    patcher.install(
        patching=[FunctionPatch(re, "sub", lambda p, r, s: s.replace(p, r))]
    )
    try:
        with capture(text="a.b") as node:
            # Python's re.sub would treat "." as a wildcard and return "___".
            result = re.sub(".", "_", node._text_inputs["text"])
            assert plain(result) == "a_b"
            required = finish_capture(node, {"result": result})
            assert reference(node, node._text_inputs["text"]).node in required
            assert reference(node, result).node in required
    finally:
        patcher.uninstall()
    assert re.sub is original


def test_user_string_lowering_records_a_generic_module_function():
    from types import ModuleType
    import leapp
    from leapp.backends.string.utf8 import StringLowering, decode, text
    from leapp.leapp_graph.datatypes.patching import TracingPatcher

    application = ModuleType("user_application")
    application.clean = lambda *, text: text.replace("!", "")
    original = application.clean

    def lowering(*, text):
        assert isinstance(text, leapp.Utf8Text)
        code, valid = text.codepoints()
        return text.filter(valid & (code != ord("!")))

    patcher = TracingPatcher()
    patcher.install(
        patching=[
            leapp.FunctionPatch(
                application,
                "clean",
                original,
                string_lowering=lowering,
            )
        ]
    )
    try:
        with capture(text="hello!") as node:
            result = application.clean(text=node._text_inputs["text"])
            assert plain(result) == "hello"
            required = finish_capture(node, {"result": result})
            assert reference(node, node._text_inputs["text"]).node in required
            recorded = reference(node, result).node
            assert recorded in required
            # The call is one module, not the str.replace inside the original.
            assert recorded.op == "call_module"
            assert recorded.meta["leapp_text_factory"] == (StringLowering, (lowering,))
    finally:
        patcher.uninstall()
    assert application.clean is original

    actual = StringLowering(lowering)(text=text("你好!"))
    assert decode(actual.data, actual.lengths) == ["你好"]
    with pytest.raises(StringCaptureError, match="must return Utf8Text"):
        StringLowering(lambda text: 7)(text("hello"))


def test_saved_method_and_calls_from_other_threads():
    with capture(text=" HELLO ") as node:
        method = node._text_inputs["text"].lower
        result = method()
        assert observes(node, result) and plain(result) == " hello "
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(lambda: " NORMAL ".strip()).result() == "NORMAL"


@pytest.mark.parametrize(
    "operation",
    [
        bool,
        len,
        lambda s: s[0],
        lambda s: s == "x",
        repr,
        ascii,
        lambda s: f"prefix{s!r}",
        lambda s: "prefix%r" % s,
        lambda s: bool(s.startswith("h")),
        lambda s: int(s.find("h")),
        lambda s: list(s.split()),
    ],
)
def test_control_rejection_survives_application_exception_handler(operation):
    with capture(text="hello") as node:
        with pytest.raises(StringCaptureError, match="unsupported"):
            operation(node._text_inputs["text"])
        with pytest.raises(StringCaptureError, match="unsupported"):
            finish_capture(node, {"result": node._text_inputs["text"]})


def test_overlapping_capture_rejects_without_restoring_active_patches():
    with capture(text="HELLO") as node:
        from leapp.backends.string.node import StringNode

        other_node = StringNode("other")
        other_node.begin({"text": "world"})
        other = Patches(other_node, {})
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="Concurrent"):
                pool.submit(other.install).result()
        other.restore()
        result = node._text_inputs["text"].lower()
        assert observes(node, result) and plain(result) == "hello"
    with capture(text="WORLD") as node:
        assert plain(node._text_inputs["text"].lower()) == "world"


def test_partial_installation_restores_already_replaced_methods(monkeypatch):
    from leapp.backends.string.node import StringNode

    node = StringNode("text")
    node.begin({"text": "hello"})
    patches = Patches(node, {})
    original_replace = patches._replace
    originals = dict(str.__dict__)
    replaced = []

    def replace(owner, name, wrapper):
        if name == "upper":
            raise RuntimeError("installation failed")
        original_replace(owner, name, wrapper)
        replaced.append(name)

    monkeypatch.setattr(patches, "_replace", replace)
    with pytest.raises(RuntimeError, match="installation failed"):
        patches.install()
    assert replaced, "the failure must follow at least one replacement"
    assert all(str.__dict__[name] is value for name, value in originals.items())


def test_already_executed_literal_call_uses_replacement_and_restores():
    def combine(left, right):
        return "".join([left, right])

    for _ in range(1000):
        assert combine("a", "b") == "ab"
    with capture(left="same", right="same") as node:
        result = combine(*node._text_inputs.values())
        assert observes(node, result)
        assert plain(result) == "samesame"
    assert type(combine("c", "d")) is str


def test_generator_argument_is_rejected_before_execution():
    consumed = []
    with capture(text="hello") as node:

        def parts():
            consumed.append(True)
            yield node._text_inputs["text"]

        with pytest.raises(StringCaptureError, match="generator"):
            # The literal receiver is not tracked, so a generator would hide
            # its dependencies. A traced separator makes the rejection explicit.
            node._text_inputs["text"].join(parts())
    assert not consumed


@pytest.fixture
def tokenizer_factory():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    def make(wrapper="PreTrainedTokenizerFast", hello_id=2):
        backend = tokenizers.Tokenizer(
            tokenizers.models.WordLevel(
                {"[UNK]": 0, "[PAD]": 1, "hello": hello_id, "world": 5 - hello_id},
                unk_token="[UNK]",
            )
        )
        backend.normalizer = tokenizers.normalizers.BertNormalizer(lowercase=False)
        backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        wrapper = (
            getattr(transformers, wrapper) if isinstance(wrapper, str) else wrapper
        )
        return wrapper(
            tokenizer_object=backend,
            unk_token="[UNK]",
            pad_token="[PAD]",
            model_input_names=["input_ids", "attention_mask"],
        )

    return make


# Base dispatch and an inherited library wrapper share discovery semantics.
@pytest.mark.parametrize("wrapper", ["PreTrainedTokenizerFast", "BertTokenizerFast"])
def test_tokenizer_discovered_only_on_traced_use_and_hooks_restored(
    tokenizer_factory, wrapper
):
    from leapp.backends.string.tokenizer import snapshot

    tokenizer = tokenizer_factory(wrapper)
    expected = tokenizer("hello world", padding=True, return_tensors="pt")["input_ids"]
    single = tokenizer("hello", padding=True, return_tensors="pt")["input_ids"]
    original = type(tokenizer).__call__
    resources = {}
    with capture(text="hello world", resources=resources) as node:
        torch.testing.assert_close(
            tokenizer("hello", padding=True, return_tensors="pt")["input_ids"], single
        )
        assert not resources
        encoded = tokenizer(
            [node._text_inputs["text"]], padding=True, return_tensors="pt"
        )
        torch.testing.assert_close(encoded["input_ids"], expected)
        assert observes(node, encoded["input_ids"])
        # Repeated use and keyword text reuse the same frozen instance.
        encoded = tokenizer(
            text=node._text_inputs["text"], padding=True, return_tensors="pt"
        )
        finish_capture(node, {"ids": encoded["input_ids"]})
    assert type(tokenizer).__call__ is original
    assert len(resources) == 1
    label, (known, descriptor) = next(iter(resources.items()))
    assert known is tokenizer and descriptor == snapshot(tokenizer)


def test_discovery_separates_instances_and_avoids_explicit_label_collision(
    tokenizer_factory,
):
    from leapp.backends.string.tokenizer import snapshot

    first, second = tokenizer_factory(), tokenizer_factory(hello_id=3)
    resources = {"tokenizer_1": (first, snapshot(first))}
    with capture(text="hello world", resources=resources) as node:
        left = first(node._text_inputs["text"], padding=True, return_tensors="pt")
        right = second(node._text_inputs["text"], padding=True, return_tensors="pt")
        finish_capture(node, {"left": left["input_ids"], "right": right["input_ids"]})
    assert left["input_ids"].tolist() == [[2, 3]]
    assert right["input_ids"].tolist() == [[3, 2]]
    assert len(resources) == 2
    assert resources["tokenizer_1"][0] is first
    assert {id(value[0]) for value in resources.values()} == {id(first), id(second)}
    assert len({value[1] for value in resources.values()}) == 2


@pytest.mark.parametrize(
    "method", [None, "__call__", "_batch_encode_plus", "_convert_encoding", "decorated"]
)
def test_discovery_accepts_inherited_wrappers_and_rejects_python_overrides(
    tokenizer_factory, method
):
    from transformers import PreTrainedTokenizerFast

    class ApplicationTokenizer(PreTrainedTokenizerFast):
        pass

    if method:
        from functools import wraps

        target_method = "_batch_encode_plus" if method == "decorated" else method

        def override(self, *args, **kwargs):
            return getattr(super(ApplicationTokenizer, self), target_method)(
                *args, **kwargs
            )

        if method == "decorated":
            override = wraps(getattr(PreTrainedTokenizerFast, target_method))(override)
        setattr(ApplicationTokenizer, target_method, override)

    original = PreTrainedTokenizerFast.__call__
    tokenizer = tokenizer_factory(ApplicationTokenizer)
    application_call = ApplicationTokenizer.__call__
    if method:
        with pytest.raises(StringCaptureError, match="explicit adapter"):
            with capture(text="hello") as node:
                tokenizer(node._text_inputs["text"], padding=True, return_tensors="pt")
    else:
        with capture(text="hello") as node:
            encoded = tokenizer(
                node._text_inputs["text"], padding=True, return_tensors="pt"
            )
            assert observes(node, encoded["input_ids"])
    assert PreTrainedTokenizerFast.__call__ is original
    assert ApplicationTokenizer.__call__ is application_call


def test_official_wrapper_with_python_preprocessing_requires_adapter(tokenizer_factory):
    tokenizer = tokenizer_factory("CodeLlamaTokenizerFast")
    # A plain sample would not exercise Code Llama's Python infilling branch.
    # Reject its override even when captured samples happen to match backend JSON.
    with pytest.raises(StringCaptureError, match="explicit adapter"):
        with capture(text="hello") as node:
            tokenizer(node._text_inputs["text"], return_tensors="pt")


def test_slow_tokenizer_requires_adapter(tmp_path):
    transformers = pytest.importorskip("transformers")
    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text("[UNK]\n[PAD]\n[CLS]\n[SEP]\n[MASK]\nhello\n")
    tokenizer = transformers.BertTokenizer(vocab_file=str(vocabulary))
    with pytest.raises(StringCaptureError, match="slow/Python"):
        with capture(text="hello") as node:
            tokenizer(node._text_inputs["text"], return_tensors="pt")


# Native tokenizer parity across algorithms and independent call options.


@pytest.fixture
def algorithm_tokenizer_factory():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    def make(algorithm="wordpiece", **defaults):
        if algorithm == "wordpiece":
            words = [
                "[UNK]",
                "[PAD]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "hello",
                "world",
                "##s",
                "cafe",
                "你",
                "好",
                "robot",
                "!",
                "long",
            ]
            backend = tokenizers.Tokenizer(
                tokenizers.models.WordPiece(
                    dict(zip(words, range(len(words)))), unk_token="[UNK]"
                )
            )
            backend.normalizer = tokenizers.normalizers.BertNormalizer(lowercase=True)
            backend.pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
            backend.post_processor = tokenizers.processors.TemplateProcessing(
                single="[CLS] $A [SEP]",
                special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
            )
            tokenizer = transformers.BertTokenizerFast(
                tokenizer_object=backend, **defaults
            )
        elif algorithm == "bpe":
            backend = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
            backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
                add_prefix_space=True
            )
            backend.train_from_iterator(
                ["hello world café", "机器人 🤖", "hello worlds!", "long robot"] * 3,
                tokenizers.trainers.BpeTrainer(
                    vocab_size=300,
                    show_progress=False,
                    initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
                    special_tokens=["<unk>", "<pad>", "<s>", "</s>", "<mask>"],
                ),
            )
            backend.post_processor = tokenizers.processors.RobertaProcessing(
                sep=("</s>", 3),
                cls=("<s>", 2),
                add_prefix_space=True,
            )
            tokenizer = transformers.RobertaTokenizerFast(
                tokenizer_object=backend, add_prefix_space=True, **defaults
            )
        else:
            vocabulary = [("<unk>", 0.0), ("<pad>", 0.0), ("</s>", 0.0)]
            vocabulary += [
                (char, -3.0) for char in sorted(set("▁hello worldscafé!你好robotlong"))
            ]
            vocabulary += [("hello", -1.0), ("world", -1.0)]
            backend = tokenizers.Tokenizer(
                tokenizers.models.Unigram(vocabulary, unk_id=0)
            )
            backend.normalizer = tokenizers.normalizers.NFKC()
            backend.pre_tokenizer = tokenizers.pre_tokenizers.Metaspace()
            backend.post_processor = tokenizers.processors.TemplateProcessing(
                single="$A </s>", special_tokens=[("</s>", 2)]
            )
            tokenizer = transformers.T5TokenizerFast(
                tokenizer_object=backend, extra_ids=0, **defaults
            )
        tokenizer.add_tokens(
            [
                tokenizers.AddedToken(
                    "<robot>", lstrip=True, rstrip=True, normalized=False
                )
            ]
        )
        return tokenizer

    return make


@pytest.fixture
def native_tokenize(monkeypatch):
    from leapp.backends.string.tokenizer import load_runtime
    from leapp.text_runtime import resolve_library

    try:
        resolve_library("pt2")
    except FileNotFoundError as error:
        pytest.skip(str(error))
    operation = load_runtime()
    # FX replay must execute the native Torch kernel without an ONNX session.
    import onnxruntime as ort

    def no_onnx_session(*args, **kwargs):
        raise AssertionError("Native Torch replay constructed an ONNX session")

    monkeypatch.setattr(ort, "InferenceSession", no_onnx_session)
    return operation


def compare(tokenizer, native_tokenize, options, samples):
    descriptor = snapshot(tokenizer)
    # Freeze native settings from the first real application call, then replay
    # every later input with that same configuration.
    configuration = None
    for text in samples:
        expected = tokenizer(text, **options)
        if configuration is None:
            configuration = tokenizer.backend_tokenizer.to_str()
            special = validate_options(options)
            names = tuple(expected)
            graph = torch.fx.Graph()
            data = graph.placeholder("data")
            lengths = graph.placeholder("lengths")
            graph.output(
                graph.call_function(
                    native_tokenize, (data, lengths, configuration, special)
                )
            )
            replay = torch.fx.GraphModule(torch.nn.Module(), graph)
        data, lengths = utf8.encode(text, 256)
        results = replay(torch.from_numpy(data), torch.from_numpy(lengths))
        actual = dict(zip(("input_ids", "attention_mask", "token_type_ids"), results))
        assert set(names) == set(expected)
        for name in names:
            reference = expected[name]
            if isinstance(reference, np.ndarray):
                reference = torch.from_numpy(reference)
            torch.testing.assert_close(actual[name], reference, rtol=0, atol=0)
        # HF's per-call mutation of backend padding/truncation is not drift.
        assert snapshot(tokenizer) == descriptor


@pytest.mark.parametrize("algorithm", ["wordpiece", "bpe", "unigram"])
def test_native_algorithms_preserve_normalizers_added_tokens_and_processors(
    algorithm_tokenizer_factory, native_tokenize, algorithm
):
    tokenizer = algorithm_tokenizer_factory(algorithm)
    compare(
        tokenizer,
        native_tokenize,
        {"padding": True, "return_tensors": "pt"},
        [
            ["HELLO worlds!", "café 你好", ""],
            ["hello  <robot>   world", "ＨＥＬＬＯ cafe\u0301 🤖", "long robot!"],
        ],
    )


@pytest.mark.parametrize(
    "options,defaults",
    [
        pytest.param({"padding": False}, {}, id="unpadded"),
        pytest.param(
            {"padding": "do_not_pad", "truncation": "do_not_truncate"},
            {},
            id="disabled-strategies",
        ),
        pytest.param(
            {"padding": "max_length", "max_length": 8}, {}, id="fixed-padding"
        ),
        pytest.param(
            {"padding": "max_length", "truncation": True, "max_length": 6},
            {},
            id="bounded",
        ),
        pytest.param(
            {"padding": "max_length", "truncation": True},
            {"model_max_length": 6},
            id="model-default",
        ),
        pytest.param(
            {"padding": "max_length", "truncation": True},
            {},
            id="unbounded-model-default",
        ),
        pytest.param(
            {"truncation": "only_first", "max_length": 5}, {}, id="only-first"
        ),
        pytest.param({"max_length": 5}, {}, id="implicit-truncation"),
        pytest.param(
            {"padding": True, "truncation": True, "max_length": 5},
            {"truncation_side": "left"},
            id="left-truncation",
        ),
        pytest.param({"padding": True}, {"padding_side": "left"}, id="left-padding"),
        pytest.param(
            {"padding": True, "padding_side": "right"},
            {"padding_side": "left"},
            id="call-padding-side",
        ),
        pytest.param(
            {"padding": True, "pad_to_multiple_of": 8}, {}, id="padding-multiple"
        ),
        pytest.param(
            {
                "padding": "max_length",
                "truncation": True,
                "max_length": 8,
                "pad_to_multiple_of": 4,
            },
            {},
            id="aligned-truncation",
        ),
        pytest.param(
            {"truncation": True, "max_length": 6, "stride": 1},
            {},
            id="stride-first-encoding",
        ),
        pytest.param(
            {"padding": True, "add_special_tokens": False}, {}, id="no-specials"
        ),
        pytest.param(
            {
                "padding": True,
                "return_token_type_ids": False,
                "return_attention_mask": False,
            },
            {},
            id="ids-only",
        ),
        pytest.param(
            {
                "padding": True,
                "return_token_type_ids": True,
                "return_attention_mask": None,
                "return_tensors": "np",
            },
            {},
            id="numpy-outputs",
        ),
    ],
)
def test_native_call_options(
    algorithm_tokenizer_factory, native_tokenize, options, defaults
):
    # Options are independent of vocabulary algorithms; exercise them once with
    # a processor that inserts special tokens, which affect truncation lengths.
    tokenizer = algorithm_tokenizer_factory(**defaults)
    samples = ["hello", "hello world long robot hello worlds!", ""]
    if options.get("padding") is True or (
        options.get("padding") == "max_length"
        and options.get("truncation")
        and (defaults.get("model_max_length") or options.get("max_length"))
    ):
        samples = [["hello", samples[1]], ["", "worlds! <robot>"]]
    compare(tokenizer, native_tokenize, {"return_tensors": "pt", **options}, samples)


def test_no_pad_token_needed_without_padding(
    algorithm_tokenizer_factory, native_tokenize
):
    tokenizer = algorithm_tokenizer_factory()
    tokenizer.pad_token = None
    compare(tokenizer, native_tokenize, {"return_tensors": "pt"}, ["hello", ""])


@pytest.mark.parametrize(
    "option",
    [
        "is_split_into_words",
        "return_overflowing_tokens",
        "return_offsets_mapping",
        "return_special_tokens_mask",
        "return_length",
        "split_special_tokens",
    ],
)
def test_unimplemented_call_semantics_rejected(option):
    with pytest.raises(StringCaptureError, match=option):
        validate_options({"return_tensors": "pt", option: True})


@pytest.mark.parametrize("options", [{}, {"padding": "max_length", "max_length": 3}])
def test_ragged_outputs_require_padding_or_truncation(
    algorithm_tokenizer_factory, native_tokenize, options
):
    tokenizer = algorithm_tokenizer_factory()
    text = ["hello", "hello world long robot"]
    configuration = tokenizer.backend_tokenizer.to_str()
    special = validate_options({"return_tensors": "pt", **options})
    data, lengths = utf8.encode(text, 64)
    with pytest.raises(RuntimeError, match="padding"):
        native_tokenize(
            torch.from_numpy(data), torch.from_numpy(lengths), configuration, special
        )
