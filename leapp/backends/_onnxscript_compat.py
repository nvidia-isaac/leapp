#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Temporary matcher workaround for ONNXScript 0.7.1 only.

Other versions use the stock exporter through the caller's fallback.
Remove this module after qualifying an upstream release
with https://github.com/microsoft/onnxscript/pull/3215 or an equivalent fix.
"""

from functools import wraps
from threading import Lock

_patch_lock = Lock()


def patch_onnxscript_matcher():
    """Install the root guard once, for subsequent exports in this process."""
    # Keep ONNXScript's exporter initialization out of ordinary LEAPP imports
    # and runtime-only use. Only the Dynamo export backend calls this helper.
    import onnxscript

    if onnxscript.__version__ != "0.7.1":
        raise RuntimeError(
            "LEAPP's ONNX matcher workaround applies only to onnxscript==0.7.1; "
            f"loaded {onnxscript.__version__} from {onnxscript.__file__}."
        )

    from onnxscript import ir
    from onnxscript.rewriter import _matcher

    with _patch_lock:
        original = _matcher.SimplePatternMatcher.match
        if getattr(original, "_leapp_root_guard", False):
            return

        @wraps(original)
        def match(
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
            return original(
                self,
                model,
                graph_or_function,
                node,
                verbose=verbose,
                remove_nodes=remove_nodes,
                tracer=tracer,
            )

        match._leapp_root_guard = True
        _matcher.SimplePatternMatcher.match = match
