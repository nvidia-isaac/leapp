# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""UTF-8 tensor operations and regex callbacks, tested apart from tokenization.

Tokenization could hide their errors as unknown tokens.
"""

import re

import numpy as np
import pytest
import torch

import leapp
from leapp.backends.string import utf8
from leapp import regex_sub_patch
from leapp.leapp_graph.datatypes.patching import TracingPatcher
from examples.string_function_patch import CharacterFilter, remove_digits


@pytest.fixture(scope="module")
def unicode_ops():
    return utf8.Unicode()


def value(strings, capacity=48):
    data, lengths = utf8.encode(strings, capacity)
    return utf8.Text(torch.from_numpy(data), torch.from_numpy(lengths), capacity)


@pytest.mark.parametrize("mode", ["lower", "upper", "strip", "lstrip", "rstrip"])
def test_unicode_transform_covers_mapping_and_property_boundaries(unicode_ops, mode):
    description = utf8._unicode_tables()
    characters = set(description["lower"]) | set(description["upper"])
    for name in ("space", "cased", "ignorable"):
        for start, end in description[name]:
            characters.update(
                chr(code)
                for boundary in (start, end)
                for code in (boundary - 1, boundary, boundary + 1)
                if 0 <= code <= 0x10FFFF and not 0xD800 <= code <= 0xDFFF
            )
    samples = sorted(characters) + [
        "ΟΣ",
        "ΟΣΑ",
        "AΣ\u0301.",
        "AΣ\u0345A",
        "Σ",
        "  İ CAFÉ 机器人 🤖\u00a0",
        "\x00a\x00",
        "",
        "\u001c x\u00a0",
        "ß ﬃ ΐ ᾀ",
        " \tleft",
        "right\u00a0",
    ]
    result = unicode_ops.transform(value(samples), mode)
    expected = [getattr(text, mode)() for text in samples]
    assert utf8.decode(result.data, result.lengths) == expected


@pytest.mark.parametrize("mode", ["strip", "lstrip", "rstrip"])
@pytest.mark.parametrize("chars", [None, "", "ab", "🤖\x00éé\u0301\ud800"])
def test_strip_character_sets_and_sides(unicode_ops, mode, chars):
    # The set and side interact at both boundaries; interior code points and
    # UTF-8 continuation bytes must survive. A surrogate in chars cannot match.
    samples = [
        "",
        "abba",
        "ba middle ab",
        "a b",
        " 🤖 ",
        "\t x\u00a0",
        "🤖é\x00x\u0301🤖",
        "🤖" * 12,
        "a" * 48,
        "é" * 24,
    ]
    result = unicode_ops.transform(value(samples), mode, chars)
    assert utf8.decode(result.data, result.lengths) == [
        getattr(sample, mode)(chars) for sample in samples
    ]


@pytest.mark.parametrize(
    "old,new,count",
    [
        ("ab", "机器人", -1),
        ("🤖", "x", -1),
        ("ab", "q", 0),
        ("<image>", "<patch>" * 128, -1),
        ("aa", "机器人", 1),
        ("ab", "q", 2**63 - 1),
        ("aa", "机器人", -1),
        ("aba", "", -1),
        ("🤖🤖", "é", -2),
        ("\x00\x00", "🤖", -1),
        ("", "🤖", -1),
        ("", "-", 2),
        ("", "a", 0),
        ("", "", -1),
    ],
)
def test_replacement_and_concatenation_preserve_bytes(old, new, count):
    samples = [
        "ababa",
        "xxabxxab",
        "🤖ab🤖",
        "",
        "\x00ab",
        "<image><image>",
        "a" * 48,
        "ab" * 24,
        "🤖" * 12,
        "é\u0301𐐨",
        "\x00" * 48,
    ]
    result = utf8.replace(value(samples), old, new, count)
    prefix = value(["前缀:"], 12)
    result = utf8.concatenate(prefix, result)
    assert utf8.decode(result.data, result.lengths) == [
        "前缀:" + text.replace(old, new, count) for text in samples
    ]


@pytest.mark.parametrize("encoder", [utf8.encode, utf8.pack])
@pytest.mark.parametrize("strings,capacity", [(["🤖"], 3), (["\ud800"], 8), ([], 8)])
def test_boundary_rejects_overflow_surrogates_and_empty_batches(
    strings, capacity, encoder
):
    with pytest.raises((ValueError, UnicodeEncodeError)):
        encoder(strings, capacity)


def test_packed_boundary_preserves_byte_lengths_and_callback_fields():
    samples = ["", "é\x00🤖", "a" * 257, "é" * 150]
    packed = utf8.pack(samples, 300)
    assert packed.dtype == np.uint8 and packed.shape == (4, 308)
    assert packed[2, :8].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    # Padding bytes are not terminators and must never extend the logical text.
    packed[0, 8:] = ord("x")
    text = utf8.unpack(torch.from_numpy(packed))
    assert text.data.dtype == torch.uint8 and text.lengths.dtype == torch.int64
    assert text.capacity == 300 and not text.scalar
    assert utf8.decode(text.data, text.lengths) == samples
    scalar = utf8.unpack(torch.from_numpy(utf8.pack("你好", 6)), scalar=True)
    assert scalar.scalar and utf8.decode(scalar.data, scalar.lengths) == ["你好"]


@pytest.mark.parametrize("pattern", ["aa", "aba", "abab", "aaa", "aabaa"])
def test_overlapping_replacement_exhaustive_short_inputs(pattern):
    # Enumerate match gaps and competing overlaps. This checks greedy selection
    # against Python without repeating the exporter for each input or count.
    from itertools import product

    samples = [
        "".join(chars) for size in range(10) for chars in product("ab", repeat=size)
    ]
    for count in (-1, 1, 2, 3):
        result = utf8.replace(value(samples, 9), pattern, "x", count)
        assert utf8.decode(result.data, result.lengths) == [
            sample.replace(pattern, "x", count) for sample in samples
        ]


@pytest.mark.parametrize("mode,capacity", [("lower", 72), ("upper", 144)])
def test_case_capacity_bounds_expansion(unicode_ops, mode, capacity):
    # U+023A expands from two to three UTF-8 bytes: exercise a completely
    # occupied input buffer, not only short strings with spare capacity.
    samples = ["Ⱥ" * 24, "İ" * 24, "ΐ" * 24, "A" * 48, "ΟΣ", ""]
    result = unicode_ops.transform(value(samples), mode)
    assert result.capacity == capacity
    result = unicode_ops.transform(unicode_ops.transform(result, "strip"), mode)
    assert utf8.decode(result.data, result.lengths) == [
        getattr(sample, mode)() for sample in samples
    ]
    other = "upper" if mode == "lower" else "lower"
    result = unicode_ops.transform(result, other)
    assert utf8.decode(result.data, result.lengths) == [
        getattr(getattr(sample, mode)(), other)() for sample in samples
    ]


def test_fx_and_onnx_preprocessing_replay_changed_text(tmp_path, unicode_ops):
    """Compile once and check raw UTF-8 through both graph runtimes.

    Independent outputs keep tokenization or later cleanup from hiding errors
    in case expansion, side-specific trimming, and replacement boundaries.
    """
    import onnxruntime as ort

    operations = [
        ("upper", ()),
        ("lstrip", ("a🤖",)),
        ("rstrip", ("b🤖",)),
        ("strip", (None,)),
        ("replace", ("aba", "🤖", -1)),
        ("replace", ("aa", "", 2)),
        ("replace", ("", "·", -1)),
        ("replace", ("", "·", 2)),
    ]

    class Preprocess(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unicode = unicode_ops
            self.user_filter = regex_sub_patch(
                {r"[^\w\s]": CharacterFilter()}
            ).string_lowering

        def forward(self, packed):
            original = utf8.unpack(packed)
            text = original
            for mode in ("lower", "strip", "lower"):
                text = self.unicode.transform(text, mode)
            text = self.user_filter(r"[^\w\s]", "", text)
            text = utf8.replace(text, " ", "___")
            text = utf8.replace(text, "__", "x", 1)
            outputs = [text.data, text.lengths]
            for mode, args in operations:
                result = (
                    utf8.replace(original, *args)
                    if mode == "replace"
                    else self.unicode.transform(original, mode, *args)
                )
                outputs.extend([result.data, result.lengths])
            return tuple(outputs)

    example = torch.from_numpy(utf8.pack(["CAPTURE!", "SECOND"], 48))
    batch = torch.export.Dim("batch", min=1, max=128)
    program = torch.export.export(
        Preprocess(),
        (example,),
        dynamic_shapes=({0: batch},),
        strict=False,
    )
    path = tmp_path / "preprocess.pt2"
    torch.export.save(program, path)
    replay = torch.export.load(path).module()
    model = torch.onnx.export(
        program, (), dynamo=True, opset_version=20, optimize=True
    ).model_proto
    runtime = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    for samples in (
        [
            "  İ 🤖 CAFÉ!  ",
            "ΟΣ",
            "\x00A",
            "",
            "🤖aababaaab🤖",
            "ß ﬃ ΐ",
            "ΐ" * 24,
            "a" * 48,
        ],
        ["single"],
    ):
        packed = utf8.pack(samples, 48)
        expected = [
            re.sub(r"[^\w\s]", "", item.lower().strip())
            .replace(" ", "___")
            .replace("__", "x", 1)
            for item in samples
        ]
        expected = [expected] + [
            [getattr(item, mode)(*args) for item in samples]
            for mode, args in operations
        ]
        actual_fx = replay(torch.from_numpy(packed))
        actual_onnx = runtime.run(
            None,
            {
                runtime.get_inputs()[0].name: packed,
            },
        )
        for actual in (actual_fx, actual_onnx):
            assert [
                utf8.decode(*actual[index : index + 2])
                for index in range(0, len(actual), 2)
            ] == expected


@pytest.mark.parametrize(
    "pattern,flags",
    [
        (r"(?<=a)b", re.IGNORECASE),
        (r"(?i:(?<=a)b)", 0),
        (r"(?<=a) b # (", re.VERBOSE),
    ],
)
def test_pattern_grammar_and_capture_references_belong_to_python(pattern, flags):
    # LEAPP has no regex engine: the pattern is only a dispatch key and the
    # template reaches the callback unexpanded. Python computes eager values.
    received = []

    def callback(text, **options):
        received.append(options["replacement"])
        return text

    patch = leapp.regex_sub_patch({pattern: callback})
    source = value(["ababAB"], 64)
    template = r"<\g<0>\n>"
    assert patch.string_lowering(pattern, template, source, flags=flags) is source
    assert received == [template]
    assert patch.replacement is re.sub


def test_multiple_patterns_dispatch_and_register_module_buffers():
    module = CharacterFilter()
    merged = leapp.regex_sub_patch({r"[^\w\s]": module, r"[0-9]+": remove_digits})
    samples = ["Hello!123 🤖你好", "café ４٢4", "", "1" * 64]
    source = value(samples, 64)
    for pattern in (r"[^\w\s]", r"[0-9]+"):
        actual = merged.string_lowering(pattern, "", source)
        assert utf8.decode(actual.data, actual.lengths) == [
            re.sub(pattern, "", value) for value in samples
        ]
    assert any(buffer is module.keep for buffer in merged.string_lowering.buffers())


@pytest.mark.parametrize("conflict", ["second-regex-patch", "generic-patch"])
def test_ambiguous_registration_rejected_before_patching(conflict):
    original = re.sub
    first = leapp.regex_sub_patch({"a": lambda text, **options: text})
    second = (
        leapp.regex_sub_patch({"b": lambda text, **options: text})
        if conflict == "second-regex-patch"
        else leapp.FunctionPatch(re, "sub", re.sub)
    )
    patcher = TracingPatcher()
    try:
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            patcher.install(patching=[first, second])
        assert re.sub is original and not patcher.installed
    finally:
        patcher.uninstall()


@pytest.mark.parametrize(
    "implementations",
    [
        {b"a": remove_digits},
        {re.compile("a"): remove_digits},
        {"a": None},
        {},
        [("a", remove_digits)],
    ],
    ids=["bytes", "compiled", "not-callable", "empty", "not-mapping"],
)
def test_invalid_registration_rejected(implementations):
    with pytest.raises(TypeError):
        leapp.regex_sub_patch(implementations)


@pytest.mark.parametrize(
    "options",
    [
        {"pattern": re.compile("a")},
        {"repl": lambda match: ""},
        {"count": 1.5},
        {"flags": True},
        {"string": "ab"},
    ],
)
def test_invalid_call_rejected(options):
    patch = leapp.regex_sub_patch({"a": lambda text, **options: text})
    arguments = {"pattern": "a", "repl": "", "string": value(["ab"], 64)}
    arguments.update(options)
    with pytest.raises(TypeError):
        patch.string_lowering(**arguments)


def test_unregistered_pattern_rejected():
    patch = leapp.regex_sub_patch({"a": lambda text, **options: text})
    with pytest.raises(ValueError, match="Unregistered re.sub call"):
        patch.string_lowering("b", "", value(["ab"], 64))


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"count": -1},
        {"flags": re.UNICODE},
        {"repl": "x", "count": 2, "flags": re.ASCII},
    ],
)
def test_callback_receives_exact_call_options(options):
    received = []

    def callback(text, **keywords):
        received.append(keywords)
        return text

    patch = leapp.regex_sub_patch({"a": callback})
    source = value(["aé"], 64)
    arguments = {"pattern": "a", "repl": "", "string": source, **options}
    assert patch.string_lowering(**arguments) is source
    assert received == [
        {
            "replacement": arguments["repl"],
            "count": arguments.get("count", 0),
            "flags": int(arguments.get("flags", 0)),
        }
    ]
    assert type(received[0]["flags"]) is int
