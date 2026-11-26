# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture HF tokenizer calls and lower them to shared native execution."""

import functools
import json

import torch
from onnxscript import INT64, UINT8

from . import StringCaptureError
from .utf8 import text


def snapshot(tokenizer):
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError as error:
        raise StringCaptureError(
            "Install Transformers to capture an HF tokenizer"
        ) from error

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise StringCaptureError(
            "Native string resources require a Hugging Face fast tokenizer; "
            "slow/Python tokenizers require an explicit adapter"
        )
    # Accept inherited implementations regardless of the subclass name. These
    # reviewed HF overrides only guard pretokenized inputs (which we reject).
    # Other overrides can transform text in Python, even in official HF models
    # such as CPM and Code Llama; their behavior is not captured by backend JSON.
    guarded_models = {
        "bart",
        "blenderbot",
        "bloom",
        "codegen",
        "cohere",
        "deberta",
        "gpt2",
        "led",
        "longformer",
        "mvp",
        "roberta",
        "whisper",
    }

    def original(method):
        method = getattr(method, "__func__", method)
        return getattr(method, "_leapp_tokenizer_original", method)

    for name in (
        "__call__",
        "_call_one",
        "encode_plus",
        "batch_encode_plus",
        "_encode_plus",
        "_batch_encode_plus",
        "_convert_encoding",
        "_get_padding_truncation_strategies",
        "set_truncation_and_padding",
        "_switch_to_input_mode",
    ):
        method = original(getattr(tokenizer, name))
        if method is original(getattr(PreTrainedTokenizerFast, name)):
            continue
        module = getattr(method, "__module__", "")
        if name in ("_encode_plus", "_batch_encode_plus") and any(
            module == f"transformers.models.{model}.tokenization_{model}_fast"
            for model in guarded_models
        ):
            # Compare identities as well: functools.wraps on an application
            # override can copy the official module and qualified name.
            import sys

            owner = getattr(sys.modules[module], method.__qualname__.split(".")[0])
            if method is original(getattr(owner, name)):
                continue
        raise StringCaptureError(
            f"Tokenizer has an unsupported Python override of {name}; "
            "supply an explicit adapter"
        )
    if tokenizer._in_target_context_manager:
        raise StringCaptureError(
            "Target-mode tokenization requires an explicit adapter"
        )
    if tokenizer.split_special_tokens:
        raise StringCaptureError(
            "Native tokenizer export requires split_special_tokens=False"
        )
    if set(tokenizer.model_input_names) - {
        "input_ids",
        "attention_mask",
        "token_type_ids",
    }:
        raise StringCaptureError(
            "Native tokenizer export only supports input_ids, attention_mask and token_type_ids"
        )
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # HF calls mutate these backend settings. Freeze the configured defaults
    # separately so calls with different options do not appear to change vocab.
    backend["padding"] = None
    backend["truncation"] = None
    return json.dumps(
        {
            "backend": backend,
            "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
            "model_max_length": tokenizer.model_max_length,
            "pad_token": tokenizer.pad_token,
            "pad_token_type_id": tokenizer.pad_token_type_id,
            "model_input_names": list(tokenizer.model_input_names),
        },
        sort_keys=True,
    )


def validate_options(options):
    """Validate supported call options and return the special-token policy."""
    options = dict(options)
    tensors = options.pop("return_tensors", None)
    special = options.pop("add_special_tokens", True)
    for key in ("return_attention_mask", "return_token_type_ids"):
        enabled = options.pop(key, None)
        if enabled is not None and type(enabled) is not bool:
            raise StringCaptureError(
                f"Tokenizer {key} must be a constant boolean or None"
            )
    for key in (
        "is_split_into_words",
        "return_overflowing_tokens",
        "return_special_tokens_mask",
        "return_offsets_mapping",
        "return_length",
        "split_special_tokens",
    ):
        if options.pop(key, False) is not False:
            raise StringCaptureError(
                f"Tokenizer option {key}=True requires an additional lowering"
            )
    options.pop("verbose", True)
    if tensors not in ("pt", "np") or type(special) is not bool:
        raise StringCaptureError(
            "Tokenizer requires return_tensors='pt'/'np' and add_special_tokens=bool"
        )
    if options.keys() - {
        "padding",
        "truncation",
        "max_length",
        "stride",
        "pad_to_multiple_of",
        "padding_side",
    }:
        raise StringCaptureError(f"Unsupported tokenizer options: {sorted(options)}")

    return special


def discover(resources, tokenizer):
    """Freeze an unregistered tokenizer on its first traced use."""
    descriptor = snapshot(tokenizer)
    index = len(resources)
    while (label := f"tokenizer_{index}") in resources:
        index += 1
    resources[label] = (tokenizer, descriptor)
    return label


class TokenizerCall(torch.nn.Module):
    def __init__(self, configuration, names, keywords):
        super().__init__()
        options = {}
        for key, value in keywords.items():
            if key == "text":
                continue
            if type(value) not in (str, int, bool, type(None)):
                raise StringCaptureError(f"Tokenizer option {key} must be constant")
            if (
                key in ("text_pair", "text_target", "text_pair_target")
                and value is None
            ):
                continue
            options[key] = value
        self.configuration = configuration
        self.special = validate_options(options)
        self.outputs = tuple(names)
        self.tokenizer_op = load_runtime()

    def forward(self, *args, **kwargs):
        if len(args) == 2 and args[1] is None:
            args = args[:1]
        if len(args) == 1 and "text" not in kwargs:
            value = args[0]
        elif not args and "text" in kwargs:
            value = kwargs["text"]
        else:
            raise StringCaptureError(
                "Tokenizer requires one text argument and no text pair"
            )
        value = text(value)
        results = self.tokenizer_op(
            value.data, value.lengths, self.configuration, self.special
        )
        return {
            name: result
            for name, result in zip(
                ("input_ids", "attention_mask", "token_type_ids"), results
            )
            if name in self.outputs
        }


@functools.lru_cache(maxsize=1)
def register_operator():
    """Register graph/schema support without requiring a native runtime."""
    torch.library.define(
        "leapp_text::tokenize",
        "(Tensor data, Tensor lengths, str configuration, bool special=True) -> (Tensor, Tensor, Tensor)",
    )

    @torch.library.register_fake("leapp_text::tokenize")
    def fake(data, lengths, configuration, special=True):
        torch._check(data.dtype == torch.uint8)
        torch._check(lengths.dtype == torch.int64)
        torch._check(data.device.type == "cpu" and lengths.device.type == "cpu")
        torch._check(data.dim() == 2)
        torch._check(lengths.dim() == 1)
        torch._check(data.shape[0] == lengths.shape[0])
        torch._check(data.shape[0] > 0)
        torch._check(data.shape[1] > 0)
        width = torch.library.get_ctx().new_dynamic_size(min=0)
        return tuple(
            data.new_empty((data.shape[0], width), dtype=torch.int64) for _ in range(3)
        )

    return torch.ops.leapp_text.tokenize.default


@functools.lru_cache(maxsize=1)
def load_runtime():
    """Load the native CPU implementation used by eager FX and PT2 replay."""
    from leapp.text_runtime import resolve_library

    operation = register_operator()
    torch.ops.load_library(str(resolve_library("pt2")))
    return operation


def onnx_translation(
    data: UINT8, lengths: INT64, configuration: str, special: bool = True
):
    from onnx.defs import OpSchema
    from onnxscript.values import Op, Opset

    schema = OpSchema(
        "HfTokenizerUtf8",
        "com.leapp.text",
        1,
        inputs=[
            OpSchema.FormalParameter("data", "tensor(uint8)"),
            OpSchema.FormalParameter("lengths", "tensor(int64)"),
        ],
        outputs=[
            OpSchema.FormalParameter(name, "tensor(int64)")
            for name in ("ids", "mask", "types")
        ],
        attributes=[
            OpSchema.Attribute("tokenizer_json", OpSchema.AttrType.STRING),
            OpSchema.Attribute("add_special_tokens", OpSchema.AttrType.INT),
        ],
    )
    op = Op(Opset("com.leapp.text", 1), "HfTokenizerUtf8", schema)
    return op(
        data, lengths, tokenizer_json=configuration, add_special_tokens=int(special)
    )
