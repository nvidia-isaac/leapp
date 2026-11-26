# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""User-owned regex implementations, exported without a regex runtime.

Run ``python examples/string_function_patch.py`` with the native text runtime
installed. LEAPP binds each regex pattern to a user implementation; Python's
original re.sub supplies the capture reference for validation.
"""

import re

import torch

import leapp


class CharacterFilter(torch.nn.Module):
    """Implement two application-selected single-character deletion patterns."""

    def __init__(self, pattern=r"[^\w\s]"):
        super().__init__()
        if pattern not in (r"[^\w\s]", r"[^0-9]"):
            raise ValueError("This adapter only supports its two character classes")
        self.pattern = pattern
        matches = re.compile(pattern).fullmatch
        # Freeze this Python interpreter's Unicode semantics before capture.
        self.register_buffer(
            "keep", torch.tensor(
                [matches(chr(code)) is None for code in range(0x110000)],
                dtype=torch.bool,
            ),
        )

    def forward(self, text: leapp.Utf8Text, *, replacement, count, flags):
        if replacement or count != 0 or flags & ~int(re.UNICODE):
            raise ValueError("CharacterFilter only supports unlimited Unicode deletion")
        codes, valid = text.codepoints()
        return text.filter(valid & self.keep[codes])


def remove_digits(text: leapp.Utf8Text, *, replacement, count, flags):
    """Implement re.sub(r'[0-9]+', '', text) using tensor operations."""
    if replacement or count != 0:
        raise ValueError("remove_digits only supports unlimited deletion")
    # Python's valid string-regex flags do not change the explicit [0-9] class.
    codes, valid = text.codepoints()
    return text.filter(valid & ((codes < ord("0")) | (codes > ord("9"))))


def main(save_path="."):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel(
        {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3, "你好": 4},
        unk_token="[UNK]",
    ))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        model_input_names=["input_ids", "attention_mask"],
    )
    original_sub = re.sub
    patch = leapp.regex_sub_patch(
        {r"[^\w\s]": CharacterFilter(), r"[0-9]+": remove_digits}
    )
    leapp.start("string_patch_example", save_path=save_path, patching=[patch])
    try:
        for sample in ("HELLO!12 WORLD", "你好 🤖2026"):
            text = leapp.annotate.input_tensors("tokenizer", {"text": sample})
            # Existing application code, including GR00T's normalization.
            text = re.sub(r"[^\w\s]", "", text.lower(), flags=re.UNICODE)
            text = re.sub(r"[0-9]+", "", text)
            encoded = tokenizer(text, padding=True, return_tensors="pt")
            leapp.annotate.output_tensors(
                "tokenizer", dict(encoded.items()), export_with="onnx",
                backend_params={"utf8_capacity": 64},
            )
    finally:
        leapp.stop()
    assert re.sub is original_sub
    assert all(leapp.compile_graph(visualize=False, validate=True, strict=True).values())

    from pathlib import Path
    artifact = Path(save_path) / "string_patch_example/string_patch_example.yaml"
    manager = leapp.InferenceManager(str(artifact))
    try:
        for sample in ("123WORLD!!! HELLO", "你好🤖456", "\x00HELLO", "", "  İ CAFÉ\u00a0ΟΣ  "):
            expected = tokenizer(
                original_sub(r"[0-9]+", "", original_sub(r"[^\w\s]", "", sample.lower())),
                padding=True, return_tensors="pt",
            )
            actual = manager.run_policy({"tokenizer/text": sample})
            for name, value in expected.items():
                torch.testing.assert_close(actual["tokenizer/" + name], value, rtol=0, atol=0)
    finally:
        manager.close()
    print("String patch: exported and replayed five fresh inputs")
    return artifact


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
