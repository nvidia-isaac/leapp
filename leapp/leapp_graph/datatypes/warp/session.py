from __future__ import annotations

import contextlib
import sys
from typing import Any

from leapp.utils.caller_identity import (
    caller_identity_has_same_anchor,
    format_caller_identity,
    get_caller_stack_identity,
)
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
        self._close_trace = None
        self._expected_close_call_stack: Any | None = None
        self._previous_trace = None
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
            closed = self.close_warp_segment(
                requester=owner_token,
                close_call_stack=close_call_stack,
            )
            if not closed:
                _get_logger().fatal(
                    "Cannot register WarpOp because the active WarpOp is "
                    "protected by an owner token.",
                    error_type=RuntimeError,
                )

        self.active_warp_op = warp_op
        self.active_segment = segment
        self.owner_token = owner_token

    def watch_for_close(
        self,
        close_call_stack: Any,
        *,
        requester: Any | None,
        require_context_exit: bool = False,
    ) -> None:
        """Close the active segment when its discovery close site is re-entered."""
        self.stop_watching_for_close()
        self._expected_close_call_stack = close_call_stack
        self._previous_trace = sys.gettrace()

        def close_trace(frame, event, _arg):
            if event != "call":
                return close_trace
            # A one-line with body shares the context's close anchor.
            if require_context_exit and (
                frame.f_code.co_name != "__exit__"
                or frame.f_locals.get("self") is not self.active_warp_op
            ):
                return close_trace

            call_stack = get_caller_stack_identity()
            if not caller_identity_has_same_anchor(
                close_call_stack,
                call_stack,
            ):
                return close_trace

            if require_context_exit:
                exc_type = frame.f_locals.get("exc_type")
                exc_value = frame.f_locals.get("exc_value")
                traceback = frame.f_locals.get("traceback")
            else:
                exc_type = exc_value = traceback = None
            self.stop_watching_for_close()
            self.close_warp_segment(
                requester=requester,
                close_call_stack=call_stack,
                exc_type=exc_type,
                exc_value=exc_value,
                traceback=traceback,
            )
            return None

        self._close_trace = close_trace
        sys.settrace(close_trace)

    def stop_watching_for_close(self) -> None:
        if self._close_trace is None:
            self._expected_close_call_stack = None
            return
        if sys.gettrace() is self._close_trace:
            sys.settrace(self._previous_trace)
        self._close_trace = None
        self._expected_close_call_stack = None
        self._previous_trace = None

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
        if self._close_trace is not None:
            attempted_close_call_stack = (
                close_call_stack
                if close_call_stack is not None
                else get_caller_stack_identity()
            )
            _get_logger().fatal(
                "Warp segment was expected to close at:\n"
                f"{format_caller_identity(self._expected_close_call_stack)}\n"
                "but a close was attempted at:\n"
                f"{format_caller_identity(attempted_close_call_stack)}",
                error_type=RuntimeError,
            )

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
        self.stop_watching_for_close()
        self.active_segment = None
        self.active_warp_op = None
        self.owner_token = None
        self._pause_depth = 0
