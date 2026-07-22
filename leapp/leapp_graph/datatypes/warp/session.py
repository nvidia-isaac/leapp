from __future__ import annotations

import contextlib
from typing import Any

from leapp.utils.logging import _get_logger


class WarpTraceSession:
    """Session-scoped shared state for Warp tracing.

    The session is the persistent source of truth for the currently active Warp
    operation and segment. It intentionally contains no patching or APIC capture
    behavior; those remain in ``WarpPatchBackend`` and ``WarpOp`` respectively.
    """

    def __init__(self) -> None:
        self.active_segment: Any | None = None
        self.active_warp_op: Any | None = None
        self.owner_token: Any | None = None
        # TODO: revisit if we need 2 different pause states for cupti and patching
        self._pause_depth = 0 # used to detect nested pauses.

    @property
    def paused(self) -> bool:
        return self._pause_depth > 0

    @contextlib.contextmanager
    def pause(self):
        self._pause_depth += 1
        try:
            yield
        finally:
            self._pause_depth -= 1

    def register_warp_op(
        self,
        warp_op: Any,
        segment: Any,
        *,
        owner_token: Any | None = None,
        close_call_stack: Any | None = None,
    ) -> None:
        if self.active_warp_op is not None:
            closed = self.close_warp_segment(close_call_stack=close_call_stack)
            if not closed:
                _get_logger().fatal(
                    "Cannot register WarpOp because the active WarpOp is "
                    "protected by an owner token.",
                    error_type=RuntimeError,
                )

        self.active_warp_op = warp_op
        self.active_segment = segment
        self.owner_token = owner_token

    def close_warp_segment(
        self,
        *,
        requester: Any | None = None,
        close_call_stack: Any | None = None,
        exc_type: Any | None = None,
        exc_value: Any | None = None,
        traceback: Any | None = None,
    ) -> bool:
        if self.active_warp_op is None:
            return True
        if self.owner_token is not None and requester is not self.owner_token:
            return False

        warp_op = self.active_warp_op
        try:
            with self.pause():
                warp_op.terminate(
                    exc_type,
                    exc_value,
                    traceback,
                    close_call_stack=close_call_stack,
                )
        finally:
            self.reset()
        return True

    def reset(self) -> None:
        self.active_segment = None
        self.active_warp_op = None
        self.owner_token = None
        self._pause_depth = 0
