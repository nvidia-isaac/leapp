# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded UTF-8 values implemented with ordinary tensor operations.

Lengths count bytes, never code points. Padding is not a terminator. Kernels
only operate on valid UTF-8 inputs; encode() enforces this at Python boundaries
and the native tokenizer checks its tensor inputs again.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import re

import numpy as np
import torch

from . import StringCaptureError


@dataclass
class Text:
    """Bounded UTF-8 tensor value passed to user string lowerings.

    ``data`` is uint8 [batch, capacity]; ``lengths`` is int64 [batch] in bytes.
    ``scalar`` distinguishes one logical string from a batch of strings.
    """

    data: object
    lengths: object
    capacity: int
    scalar: bool = False

    def codepoints(self):
        """Return (codes, valid), both [batch, capacity], at UTF-8 start bytes.

        Only positions selected by ``valid`` represent characters. Padding and
        continuation-byte positions must not contribute to a computation.
        """
        code, valid, _ = characters(self)
        return code, valid

    def filter(self, keep):
        """Retain whole characters selected by a bool mask at start bytes."""
        _, valid = self.codepoints()
        starts = torch.cumsum(valid.to(torch.int64), dim=1)
        retained = (
            torch.zeros_like(starts[:, :1])
            .expand(-1, self.capacity + 1)
            .clone()
            .scatter_add(1, starts, (keep & valid).to(torch.int64))
        )
        byte_keep = (torch.gather(retained, 1, starts) > 0) & (
            torch.arange(self.capacity)[None, :] < self.lengths[:, None]
        )
        data, lengths = compact(self.data.to(torch.int64), byte_keep, self.capacity)
        return Text(data, lengths, self.capacity, self.scalar)


def encode(value, capacity):
    values = [value] if type(value) is str else value
    if not values or any(type(item) is not str for item in values):
        raise ValueError("UTF-8 inputs require a nonempty batch of strings")
    encoded = [item.encode("utf-8", errors="strict") for item in values]
    if any(len(item) > capacity for item in encoded):
        raise ValueError(f"UTF-8 input exceeds its {capacity}-byte capacity")
    data = np.zeros((len(values), capacity), dtype=np.uint8)
    for index, item in enumerate(encoded):
        data[index, : len(item)] = np.frombuffer(item, dtype=np.uint8)
    return data, np.asarray([len(item) for item in encoded], dtype=np.int64)


def pack(value, capacity):
    """Encode each boundary row as an eight-byte little-endian length and bytes."""
    data, lengths = encode(value, capacity)
    header = lengths.astype("<i8").view(np.uint8).reshape(-1, 8)
    return np.concatenate((header, data), axis=1)


def unpack(tensor, scalar=False):
    """Restore callback fields using portable tensor arithmetic inside the graph."""
    weights = tensor.new_tensor([256**index for index in range(8)], dtype=torch.int64)
    lengths = (tensor[:, :8].to(torch.int64) * weights).sum(dim=1)
    return Text(tensor[:, 8:], lengths, tensor.shape[1] - 8, scalar)


def decode(data, lengths):
    """Boundary/debug helper; never emitted into a graph."""
    return [
        bytes(row[: int(length)].tolist()).decode("utf-8", errors="strict")
        for row, length in zip(data, lengths)
    ]


def compact(values, valid, capacity):
    """Stable byte compaction into a statically bounded buffer."""
    positions = torch.cumsum(valid.to(torch.int64), dim=1) - 1
    output = torch.zeros_like(values[:, :1]).expand(-1, capacity).clone()
    output = torch.scatter_add(
        output,
        1,
        torch.clamp(positions, min=0, max=capacity - 1),
        torch.where(valid, values, 0),
    )
    return output.to(torch.uint8), valid.sum(dim=1)


def concatenate(left, right):
    capacity = left.capacity + right.capacity
    batch = (left.lengths + right.lengths).shape[0]
    values = torch.cat(
        (left.data.expand(batch, -1), right.data.expand(batch, -1)), dim=1
    ).to(torch.int64)
    valid = torch.cat(
        (
            (torch.arange(left.capacity)[None, :] < left.lengths[:, None]).expand(
                batch, -1
            ),
            (torch.arange(right.capacity)[None, :] < right.lengths[:, None]).expand(
                batch, -1
            ),
        ),
        dim=1,
    )
    data, lengths = compact(values, valid, capacity)
    return Text(data, lengths, capacity, left.scalar and right.scalar)


def characters(text):
    data = text.data.to(torch.int64)
    positions = torch.arange(text.capacity)[None, :]
    valid = (positions < text.lengths[:, None]) & ((data & 0xC0) != 0x80)
    width = torch.where(
        data < 0x80, 1, torch.where(data < 0xE0, 2, torch.where(data < 0xF0, 3, 4))
    )
    code = data & torch.where(
        width == 1,
        0x7F,
        torch.where(width == 2, 0x1F, torch.where(width == 3, 0x0F, 0x07)),
    )
    for offset in range(1, 4):
        following = (
            data[
                :,
                torch.clamp(
                    torch.arange(text.capacity) + offset, max=text.capacity - 1
                ),
            ]
            & 0x3F
        )
        code = torch.where(width > offset, code * 64 + following, code)
    return code.clamp(0, 0x10FFFF), valid, width


def _ranges(values):
    result = []
    for value in values:
        if result and result[-1][1] + 1 == value:
            result[-1][1] = value
        else:
            result.append([value, value])
    return result


@lru_cache(maxsize=1)
def _unicode_tables():
    # Freeze the capture interpreter's case conversion and whitespace behavior.
    lower, upper, cased, ignorable, space = {}, {}, [], [], []
    for code in range(0x110000):
        if 0xD800 <= code <= 0xDFFF:
            continue
        char = chr(code)
        if char.lower() != char:
            lower[char] = char.lower()
        if char.upper() != char:
            upper[char] = char.upper()
        if char.islower() or char.isupper() or char.istitle():
            cased.append(code)
        # Final_Sigma is Python lower()'s contextual mapping. These probes
        # distinguish case-ignorable characters, including cased marks.
        before_boundary = ("AΣ" + char + ".").lower()[1]
        before_letter = ("AΣ" + char + "A").lower()[1]
        if before_boundary == "ς" and before_letter == "σ":
            ignorable.append(code)
        if char.isspace():
            space.append(code)
    return {
        "lower": lower,
        "upper": upper,
        "cased": _ranges(cased),
        "ignorable": _ranges(ignorable),
        "space": _ranges(space),
    }


@lru_cache(maxsize=1)
def _tables():
    description = _unicode_tables()
    flags = np.zeros((0x110000, 1), dtype=np.uint8)
    for bit, name in enumerate(("cased", "ignorable", "space")):
        for start, end in description[name]:
            flags[start : end + 1, 0] |= 1 << bit

    def pages(array):
        unique, indices = np.unique(
            array.reshape(-1, 256 * array.shape[1]), axis=0, return_inverse=True
        )
        return torch.from_numpy(indices.copy()), torch.from_numpy(
            unique.reshape(-1, array.shape[1]).copy()
        ).to(torch.int64)

    mappings = {}
    for mode in ("lower", "upper"):
        mapping = description[mode]
        maximum = max(4, max(len(value.encode("utf-8")) for value in mapping.values()))
        array = np.full((0x110000, maximum + 1), -1, dtype=np.int16)
        for key, value in mapping.items():
            encoded = value.encode("utf-8")
            array[ord(key)] = [
                len(encoded),
                *encoded,
                *([0] * (maximum - len(encoded))),
            ]
        expansion = max(
            Fraction(len(value.encode("utf-8")), len(key.encode("utf-8")))
            for key, value in mapping.items()
        )
        mappings[mode] = (
            *pages(array),
            maximum,
            (expansion.numerator, expansion.denominator),
        )
    return mappings, pages(flags)


class Unicode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mappings, (flag_pages, flags) = _tables()
        self.maximum, self.expansion = {}, {}
        for mode, (pages, values, maximum, expansion) in mappings.items():
            self.register_buffer(mode + "_pages", pages)
            self.register_buffer(mode + "_bytes", values)
            self.maximum[mode], self.expansion[mode] = maximum, expansion
        self.register_buffer("flag_pages", flag_pages)
        self.register_buffer("flags", flags)

    def lookup(self, code, name):
        pages = self.flag_pages if name == "flags" else getattr(self, name + "_pages")
        values = self.flags if name == "flags" else getattr(self, name + "_bytes")
        return values[pages[code // 256] * 256 + code % 256]

    def transform(self, text, mode, chars=None):
        code, valid, width = characters(text)
        if mode in ("strip", "lstrip", "rstrip"):
            if chars is None:
                nonspace = valid & ((self.lookup(code, "flags")[..., 0] & 4) == 0)
            else:
                # Constants describe a set of code points, not a substring.
                # Keep intermediate storage linear in the input capacity.
                nonspace = valid
                for char in dict.fromkeys(chars):
                    nonspace = nonspace & (code != ord(char))
            keep = valid
            if mode != "rstrip":
                keep = keep & (torch.cumsum(nonspace.to(torch.int64), 1) > 0)
            if mode != "lstrip":
                keep = keep & (
                    torch.flip(
                        torch.cumsum(torch.flip(nonspace.to(torch.int64), [1]), 1),
                        [1],
                    )
                    > 0
                )
            return text.filter(keep)
        if mode not in ("lower", "upper"):
            raise ValueError(f"Unsupported Unicode transformation: {mode}")

        mapped = self.lookup(code, mode)
        maximum = self.maximum[mode]
        sigma = None
        # Final_Sigma: find adjacent non-ignorable code points by segment ID,
        # using prefix sums/gather rather than a host loop or quadratic matrix.
        if mode == "lower":
            flags = self.lookup(code, "flags")[..., 0]
            significant = valid & ((flags & 2) == 0)
            segments = torch.cumsum(significant.to(torch.int64), 1)
            cased = ((flags & 1) != 0) & significant
            buckets = (
                torch.zeros_like(segments[:, :1])
                .expand(-1, text.capacity + 2)
                .clone()
                .scatter_add(1, segments, cased.to(torch.int64))
            )
            before = (
                torch.gather(buckets, 1, segments - significant.to(torch.int64)) > 0
            )
            after = torch.gather(buckets, 1, segments + 1) > 0
            sigma = (code == 0x03A3) & before & ~after
        # Unchanged scalars retain their exact original UTF-8 byte sequence.
        original = torch.stack(
            [
                text.data[
                    :,
                    torch.clamp(
                        torch.arange(text.capacity) + offset, max=text.capacity - 1
                    ),
                ].to(torch.int64)
                for offset in range(maximum)
            ],
            dim=2,
        )
        use_mapping = mapped[..., 0] >= 0
        output_width = torch.where(use_mapping, mapped[..., 0], width)
        values = torch.where(use_mapping[..., None], mapped[..., 1:], original)
        if sigma is not None:
            sigma_bytes = torch.tensor(
                [0xCF, 0x82] + [0] * (maximum - 2), dtype=torch.int64
            )
            values = torch.where(sigma[..., None], sigma_bytes, values)
        byte_valid = valid[..., None] & (
            torch.arange(maximum)[None, None, :] < output_width[..., None]
        )
        numerator, denominator = self.expansion[mode]
        capacity = (text.capacity * numerator + denominator - 1) // denominator
        data, lengths = compact(values.flatten(1), byte_valid.flatten(1), capacity)
        return Text(data, lengths, capacity, text.scalar)


def _nonoverlapping(match, size, limit):
    """Select the first ``limit`` greedy matches using binary successor jumps.

    Each candidate points to the first candidate after its consumed bytes.
    Doubling these jumps selects every requested match in logarithmically many
    tensor steps, with linear buffers and no capacity-sized unrolled scan.
    """
    capacity = match.shape[1]
    positions = torch.arange(capacity)[None, :]
    ordinal = torch.cumsum(match.to(torch.int64), dim=1)
    locations = (
        torch.zeros_like(ordinal[:, :1])
        .expand(-1, capacity + 1)
        .clone()
        .scatter_add(1, ordinal, torch.where(match, positions + 1, 0))
    )
    locations = torch.where(locations > 0, locations - 1, capacity)
    # Rank zero is unused. All ranks past the final match map to a sentinel.
    end = (locations + size - 1).clamp(max=capacity - 1)
    successor = (torch.gather(ordinal, 1, end) + 1).clamp(max=capacity)
    selected = torch.ones_like(ordinal[:, :1]).expand(-1, limit)
    offsets = torch.arange(limit)[None, :]
    for bit in range((limit - 1).bit_length()):
        selected = torch.where(
            (offsets & (1 << bit)) != 0,
            torch.gather(successor, 1, selected),
            selected,
        )
        if (1 << (bit + 1)) < limit:
            successor = torch.gather(successor, 1, successor)
    selected = torch.gather(locations, 1, selected)
    result = torch.zeros_like(locations).scatter_add(
        1, selected, (selected < capacity).to(torch.int64)
    )
    return result[:, :capacity] > 0


def _insert(output, destinations, match, new, limit):
    """Fill disjoint constant insertions without a capacity-by-new-length array."""
    capacity = output.shape[1]
    events = torch.zeros_like(output).scatter_add(
        1, destinations.clamp(0, capacity - 1), match.to(torch.int64)
    )
    starts = (
        torch.zeros_like(output[:, :1])
        .expand(-1, limit + 1)
        .clone()
        .scatter_add(
            1,
            torch.cumsum(match.to(torch.int64), dim=1),
            torch.where(match, destinations, 0),
        )
    )
    which = torch.cumsum(events, dim=1)
    offset = torch.arange(capacity)[None, :] - torch.gather(starts, 1, which)
    inserted = (which > 0) & (offset >= 0) & (offset < len(new))
    replacement = torch.tensor(list(new), dtype=torch.int64)
    return torch.where(inserted, replacement[offset.clamp(0, len(new) - 1)], output)


def _replace_empty(text, new, count):
    # Python inserts at code-point boundaries, including after the final code
    # point and once for an empty string. UTF-8 continuation bytes are excluded.
    _, valid = text.codepoints()
    positions = torch.arange(text.capacity + 1)[None, :]
    match = torch.nn.functional.pad(valid, (0, 1)) | (
        positions == text.lengths[:, None]
    )
    limit = text.capacity + 1 if count < 0 else min(count, text.capacity + 1)
    ordinal = torch.cumsum(match.to(torch.int64), dim=1)
    match = match & (ordinal <= limit)
    ordinal = torch.cumsum(match.to(torch.int64), dim=1)
    capacity = text.capacity + limit * len(new)
    destinations = positions[:, :-1] + ordinal[:, :-1] * len(new)
    output = torch.zeros_like(text.lengths[:, None]).expand(-1, capacity).clone()
    output = output.scatter_add(
        1,
        destinations.clamp(max=capacity - 1),
        torch.where(
            positions[:, :-1] < text.lengths[:, None], text.data.to(torch.int64), 0
        ),
    )
    starts = positions + (ordinal - match.to(torch.int64)) * len(new)
    output = _insert(output, starts, match, new, limit)
    lengths = text.lengths + match.sum(dim=1) * len(new)
    return Text(output.to(torch.uint8), lengths, capacity, text.scalar)


def replace(text, old, new, count=-1):
    old, new = old.encode("utf-8"), new.encode("utf-8")
    if count == 0 or old == new:
        return text
    if not old:
        return _replace_empty(text, new, count)
    size, added = len(old), max(0, len(new) - len(old))
    if size > text.capacity:
        return text
    limit = text.capacity // size if count < 0 else min(count, text.capacity // size)
    capacity = max(1, text.capacity + limit * added)
    positions = torch.arange(text.capacity)[None, :]
    match = positions + size <= text.lengths[:, None]
    for offset, byte in enumerate(old):
        match = match & (
            text.data[
                :,
                torch.clamp(
                    torch.arange(text.capacity) + offset, max=text.capacity - 1
                ),
            ]
            == byte
        )
    if count != 1 and any(old[:index] == old[-index:] for index in range(1, size)):
        match = _nonoverlapping(match, size, limit)
    elif count >= 0:
        ordinal = torch.cumsum(match.to(torch.int64), dim=1)
        match = match & (ordinal <= limit)
    covered = match.to(torch.int64)
    for offset in range(1, size):
        covered = covered + torch.nn.functional.pad(
            match[:, :-offset].to(torch.int64), (offset, 0)
        )
    unchanged = (covered == 0) & (positions < text.lengths[:, None])
    widths = torch.where(
        match,
        len(new),
        unchanged.to(torch.int64),
    )
    destinations = torch.cumsum(widths, dim=1) - widths
    output = torch.zeros_like(text.lengths[:, None]).expand(-1, capacity).clone()
    output = output.scatter_add(
        1,
        destinations.clamp(0, capacity - 1),
        torch.where(unchanged, text.data.to(torch.int64), 0),
    )
    if new:
        output = _insert(output, destinations, match, new, limit)
    return Text(output.to(torch.uint8), widths.sum(dim=1), capacity, text.scalar)


# Modules and helpers executed by captured FX graphs.


def text(value):
    if isinstance(value, Text):
        return value
    if type(value) is str:
        data, lengths = encode(value, max(1, len(value.encode("utf-8"))))
        return Text(
            torch.from_numpy(data), torch.from_numpy(lengths), data.shape[1], True
        )
    if isinstance(value, (list, tuple)) and value:
        items = [text(item) for item in value]
        if any(not item.scalar for item in items):
            raise StringCaptureError(
                "Nested text batches require paired-input semantics"
            )
        capacity = max(item.capacity for item in items)
        return Text(
            torch.cat(
                [
                    torch.nn.functional.pad(item.data, (0, capacity - item.capacity))
                    for item in items
                ]
            ),
            torch.cat([item.lengths for item in items]),
            capacity,
        )
    raise StringCaptureError("Expected a string or a nonempty batch of strings")


class TextOperations(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unicode = Unicode()

    def forward(self, operation, receiver, *args, **kwargs):
        if not kwargs:
            if operation == "add" and len(args) == 1:
                return concatenate(text(receiver), text(args[0]))
            if (
                operation == "join"
                and len(args) == 1
                and isinstance(args[0], (list, tuple))
            ):
                parts = args[0]
                result = text(parts[0]) if parts else text("")
                for part in parts[1:]:
                    result = concatenate(result, text(receiver))
                    result = concatenate(result, text(part))
                return result
            if operation in ("lower", "upper") and not args:
                return self.unicode.transform(text(receiver), operation)
            if operation in ("strip", "lstrip", "rstrip") and (
                not args
                or (len(args) == 1 and (args[0] is None or type(args[0]) is str))
            ):
                return self.unicode.transform(text(receiver), operation, *args)
            if (
                operation == "replace"
                and len(args) in (2, 3)
                and all(type(item) is str for item in args[:2])
                and (len(args) == 2 or type(args[2]) is int)
            ):
                return replace(text(receiver), *args)
        raise StringCaptureError(
            "unsupported call; supply a FunctionPatch with supported string "
            "operations or a string_lowering callback"
        )


class StringLowering(torch.nn.Module):
    """Own callback buffers and execute the callback only in the live graph."""

    def __init__(self, implementation):
        super().__init__()
        self.implementation = implementation

    def forward(self, *args, **kwargs):
        result = self.implementation(*args, **kwargs)
        if not isinstance(result, Text):
            raise StringCaptureError(
                "FunctionPatch.string_lowering must return Utf8Text"
            )
        return result


def unsupported(operation, *dependencies):
    raise StringCaptureError(f"unsupported call: {operation}")


# Application-supplied regex implementations.

_SUB = re.sub


class _RegexSub(torch.nn.Module):
    def __init__(self, implementations):
        super().__init__()
        self.implementations = dict(implementations)
        for index, value in enumerate(self.implementations.values()):
            if isinstance(value, torch.nn.Module):
                # Registered modules export their buffers with the graph.
                self.add_module(f"implementation_{index}", value)

    def forward(self, pattern, repl, string, count=0, flags=0):
        if type(pattern) is not str or type(repl) is not str:
            raise TypeError("regex_sub_patch requires string patterns and replacements")
        if type(count) is not int:
            raise TypeError("regex_sub_patch count must be a constant integer")
        if not isinstance(flags, int) or isinstance(flags, bool):
            raise TypeError(
                "regex_sub_patch flags must be a constant integer or RegexFlag"
            )
        if pattern not in self.implementations:
            raise ValueError(
                "Unregistered re.sub call: pattern must match a regex_sub_patch registration"
            )
        if not isinstance(string, Text):
            raise TypeError("regex_sub_patch requires a captured Utf8Text argument")
        return self.implementations[pattern](
            string, replacement=repl, count=count, flags=int(flags)
        )


def regex_sub_patch(implementations: Mapping[str, Callable[..., Text]]):
    """Implement ``re.sub`` for constant patterns with tensor callbacks.

    ``implementations`` maps each pattern to ``callback(text, *, replacement,
    count, flags) -> Utf8Text`` built from exportable tensor operations; an
    ``nn.Module`` callback can hold buffers. LEAPP has no regex engine, so each
    callback implements or rejects its call's options. Python's ``re.sub``
    still computes capture-time values. Pass the returned patch to
    ``leapp.start(patching=[...])``.
    """
    from leapp.leapp_graph.datatypes.patching import FunctionPatch

    if not isinstance(implementations, Mapping) or not implementations:
        raise TypeError(
            "regex_sub_patch requires a nonempty {pattern: callback} mapping"
        )
    for pattern, implementation in implementations.items():
        if type(pattern) is not str:
            raise TypeError("regex_sub_patch patterns must be strings")
        if not callable(implementation):
            raise TypeError("regex_sub_patch callbacks must be callable")
    return FunctionPatch(re, "sub", _SUB, string_lowering=_RegexSub(implementations))
