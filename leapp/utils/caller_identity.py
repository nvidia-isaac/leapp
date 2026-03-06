#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import site
import sysconfig
import traceback
from dataclasses import dataclass

_LEAPP_PKG_DIR = os.path.realpath(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STDLIB_DIRS = tuple(sorted({
    os.path.realpath(path) for path in (
        sysconfig.get_paths().get("stdlib"),
        sysconfig.get_paths().get("platstdlib"),
    ) if path
}))
_SITE_PACKAGE_DIRS = tuple(sorted({
    os.path.realpath(path) for path in (
        site.getusersitepackages(),
        *site.getsitepackages(),
        sysconfig.get_paths().get("purelib"),
        sysconfig.get_paths().get("platlib"),
    ) if path
}))


@dataclass(frozen=True)
class CallerFrame:
    filename: str
    lineno: int
    function: str


@dataclass(frozen=True)
class CallerIdentity:
    anchor: CallerFrame
    context: tuple[CallerFrame, ...] = ()


def _normalize_frame_path(path: str) -> str:
    if not path or path.startswith("<"):
        return path
    return os.path.realpath(path)


def _path_is_within(path: str, roots: tuple[str, ...]) -> bool:
    if not path or path.startswith("<"):
        return False
    return any(path == root or path.startswith(root + os.sep) for root in roots)


def _is_context_relevant_frame(frame: CallerFrame) -> bool:
    path = frame.filename
    if not path or path.startswith("<"):
        return False
    if _path_is_within(path, _SITE_PACKAGE_DIRS):
        return False
    if _path_is_within(path, _STDLIB_DIRS):
        return False
    return True


def caller_identity_has_same_anchor(lhs, rhs) -> bool:
    if isinstance(lhs, CallerIdentity) and isinstance(rhs, CallerIdentity):
        return lhs.anchor == rhs.anchor
    return lhs == rhs


def get_caller_stack_identity():
    """Return a normalized identity for the current annotation call site.

    The closest non-LEAPP frame is treated as the stable annotation origin.
    A short, filtered caller context is also recorded so LEAPP can warn when
    outer orchestration changes without rejecting the re-entry outright.
    """
    frames = []
    for frame in traceback.extract_stack():
        filename = _normalize_frame_path(frame.filename)
        if filename and filename.startswith(_LEAPP_PKG_DIR):
            continue
        frames.append(CallerFrame(filename, frame.lineno, frame.name))

    if not frames:
        return CallerIdentity(CallerFrame("<unknown>", -1, "<unknown>"))

    closest_first = list(reversed(frames))
    anchor = closest_first[0]
    context = tuple(
        frame for frame in closest_first[1:]
        if _is_context_relevant_frame(frame)
    )[:2]
    return CallerIdentity(anchor=anchor, context=context)


def format_caller_identity(identity):
    """Pretty-print a caller identity for error messages."""
    if isinstance(identity, CallerIdentity):
        lines = [
            f"  anchor: {identity.anchor.filename}:{identity.anchor.lineno} "
            f"in {identity.anchor.function}"
        ]
        if identity.context:
            lines.append("  context:")
            for frame in identity.context:
                lines.append(
                    f"    {frame.filename}:{frame.lineno} in {frame.function}")
        return "\n".join(lines)
    if isinstance(identity, CallerFrame):
        return f"{identity.filename}:{identity.lineno} in {identity.function}"
    if not isinstance(identity, tuple) or not identity:
        return str(identity)
    if len(identity) == 2 and isinstance(identity[0], str) and isinstance(identity[1], int):
        return f"{identity[0]}:{identity[1]}"
    lines = []
    for frame in identity:
        if isinstance(frame, tuple) and len(frame) == 3:
            lines.append(f"  {frame[0]}:{frame[1]} in {frame[2]}")
        else:
            lines.append(f"  {frame}")
    return "\n".join(lines)
