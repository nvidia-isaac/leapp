#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Best-effort ONNXScript matcher optimization with a stock-matcher fallback.

Remove this module after qualifying an upstream release
with https://github.com/microsoft/onnxscript/pull/3215 or an equivalent fix.
"""

from functools import wraps
from threading import Lock

from leapp.utils.logging import _get_logger

_patch_lock = Lock()
_patch_disabled = False


def patch_onnxscript_matcher():
    """Install once; a guard failure disables the patch for this process."""
    # Keep ONNXScript's exporter initialization out of ordinary LEAPP imports
    # and runtime-only use. Only the Dynamo export backend calls this helper.
    from onnxscript import ir
    from onnxscript.rewriter import _matcher

    with _patch_lock:
        if _patch_disabled:
            return
        original = _matcher.SimplePatternMatcher.match
        if getattr(original, "_leapp_root_guard", False):
            return

        def reject_incompatible_root(
            self,
            model,
            graph_or_function,
            node,
            *,
            verbose=0,
            remove_nodes=True,
            tracer=None,
        ):
            if not self.pattern.has_single_output_node:
                # Match the upstream guard's state initialization and predicate.
                # Reject impossible roots before the original whole-graph scan.
                self._tracer = tracer
                self._graph = (
                    graph_or_function
                    if isinstance(graph_or_function, ir.Graph)
                    else graph_or_function.graph
                )
                self._init_match(verbose)
                outputs = self.pattern.output_nodes
                if outputs and not outputs[0].matches(node, self._match):
                    return self._match

        @wraps(original)
        def match(self, *args, **kwargs):
            global _patch_disabled
            if not _patch_disabled:
                try:
                    rejected = reject_incompatible_root(self, *args, **kwargs)
                    if rejected is not None:
                        return rejected
                except Exception as error:
                    with _patch_lock:
                        if not _patch_disabled:
                            _patch_disabled = True
                            if _matcher.SimplePatternMatcher.match is match:
                                _matcher.SimplePatternMatcher.match = original
                            _get_logger().info(
                                "Disabled ONNXScript matcher workaround after an "
                                f"error; using the stock matcher: {error}"
                            )
            # Keep native errors outside the guard's handler: never retry them.
            return original(self, *args, **kwargs)

        match._leapp_root_guard = True
        _matcher.SimplePatternMatcher.match = match
