#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Shared installation and restoration for monkeypatched attributes."""

import inspect
from typing import Any


_MISSING = object()


class AttributePatchRegistry:
    """Track installed attribute wrappers and restore them safely."""

    def __init__(self) -> None:
        self._patches: list[tuple[Any, str, Any, Any]] = []

    def __len__(self) -> int:
        return len(self._patches)

    def contains(self, owner: Any, attr_name: str) -> bool:
        return any(
            patch_owner is owner and patch_name == attr_name
            for patch_owner, patch_name, _, _ in self._patches
        )

    def install(
        self,
        owner: Any,
        attr_name: str,
        original: Any,
        wrapper: Any,
    ) -> None:
        self._patches.append((owner, attr_name, original, wrapper))
        try:
            setattr(owner, attr_name, wrapper)
        except Exception:
            current = inspect.getattr_static(owner, attr_name, _MISSING)
            if current is wrapper:
                setattr(owner, attr_name, original)
            self._patches.pop()
            raise

    def restore(self, *, suppress_errors: bool = False) -> None:
        while self._patches:
            owner, attr_name, original, wrapper = self._patches[-1]
            try:
                try:
                    current = inspect.getattr_static(
                        owner,
                        attr_name,
                        _MISSING,
                    )
                except Exception:
                    current = getattr(owner, attr_name)
                if current is wrapper:
                    setattr(owner, attr_name, original)
            except Exception:
                if not suppress_errors:
                    raise
            self._patches.pop()
