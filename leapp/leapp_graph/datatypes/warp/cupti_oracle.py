from __future__ import annotations

from typing import Any

from leapp.utils.logging import _get_logger

# Soft-import CUPTI so missing cupti-python (e.g. Windows, where the package
# has no wheels) does not break importing the Warp tracing modules. Torch and
# NumPy tracing must still work when warp-lang is present but CUPTI is not.
# WarpOp._verify_warp() enforces CUPTI/Linux when Warp tracing is actually used.
try:
    from cupti import cupti

    CUPTI_AVAILABLE = True
except ImportError:
    cupti = None
    CUPTI_AVAILABLE = False

if CUPTI_AVAILABLE:
    # cupti-python 12.x/13.0 exposes these enums in snake_case, while newer
    # 13.x versions also expose PascalCase aliases. Resolve by symbol so LEAPP
    # can run with either CUDA 12 or CUDA 13 CUPTI wheels.
    _DRIVER_API_TRACE_CBID = getattr(
        cupti,
        "Driver_api_trace_cbid",
        getattr(cupti, "driver_api_trace_cbid", None),
    )
    _RUNTIME_API_TRACE_CBID = getattr(
        cupti,
        "Runtime_api_trace_cbid",
        getattr(cupti, "runtime_api_trace_cbid", None),
    )
else:
    _DRIVER_API_TRACE_CBID = None
    _RUNTIME_API_TRACE_CBID = None


class WarpCudaOracle:
    """Warning-only CUPTI oracle for CUDA work during active Warp segments."""

    _SYNC_READBACK_FRAGMENTS = (
        "DeviceSynchronize",
        "CtxSynchronize",
        "StreamSynchronize",
        "EventSynchronize",
        "ThreadSynchronize",
        "STREAM_SYNCHRONIZED",
        "CONTEXT_SYNCHRONIZED",
        "StreamQuery",
        "EventQuery",
        "MemcpyDtoH",
        "MemcpyHtoD",
        "MemcpyAtoH",
        "MemcpyHtoA",
        "MemcpyFromSymbol",
        "MemcpyToSymbol",
        "DeviceReset",
    )

    _CUDA_WORK_FRAGMENTS = (
        "LaunchKernel",
        "cuLaunchKernel",
        "Malloc",
        "MemAlloc",
        "Free",
        "MemFree",
        "Memcpy",
        "Memset",
        "EventRecord",
        "StreamWaitEvent",
        "StreamCreate",
        "StreamDestroy",
        "StreamIsCapturing",
        "GetCaptureInfo",
    )

    # Known metadata/status callbacks seen during CUDA setup:
    # "GetDevice", "CtxGet", "GetCurrent", "GetLastError",
    # "PeekAtLastError", "GetError", "DevicePrimaryCtxGetState",
    # "KernelGetName", "KernelGetAttribute", "ModuleGetFunction",
    # "FuncSetAttribute", "LibraryGetKernel", "LibraryLoadData".
    # They are not explicitly filtered because non-listed callbacks are allowed
    # by default; only hard-ban and soft-ban fragments below create warnings.

    def __init__(self, boundary_handler=None) -> None:
        self._session: Any | None = None
        self._subscriber: int | None = None
        self._warned_by_segment: dict[int, set[tuple[str, str, str]]] = {}
        self._callback_error_logged = False
        self._callback = self._on_callback
        self._boundary_handler = boundary_handler

    def set_session(self, session: Any | None) -> None:
        self._session = session

    def start(self) -> None:
        if self._subscriber is not None:
            return
        if not CUPTI_AVAILABLE:
            return

        self._subscriber = int(cupti.subscribe(self._callback, None))
        cupti.enable_all_domains(1, self._subscriber)

    def stop(self) -> None:
        if self._subscriber is None:
            return

        try:
            cupti.enable_all_domains(0, self._subscriber)
        except Exception:
            pass
        try:
            cupti.unsubscribe(self._subscriber)
        except Exception:
            pass
        self._subscriber = None
        self._session = None
        self._warned_by_segment.clear()

    def _on_callback(
        self,
        _userdata: Any,
        domain_id: int,
        cbid: int,
        _cbdata: Any,
    ) -> None:
        try:
            self._handle_callback(domain_id, cbid)
        except Exception as exc:
            # Exceptions cannot escape CUPTI callbacks safely; CUDA reports
            # them as low-level SystemErrors at the original API callsite.
            if not self._callback_error_logged:
                _get_logger().warning(
                    "LEAPP disabled one CUPTI callback after an internal "
                    f"decode error: {exc}"
                )
                self._callback_error_logged = True

    def _handle_callback(self, domain_id: int, cbid: int) -> None:
        session = self._session
        segment = None if session is None else session.active_segment
        if segment is None:
            return

        domain = self._domain_name(domain_id)
        callback = self._callback_name(domain, cbid)
        reason = self._trip_reason(callback)
        if reason is None:
            return
        if self._inside_warp_cuda_window():
            return

        self._log_cupti_event(segment, domain, callback, reason)
        if self._boundary_handler is not None:
            self._boundary_handler(segment, domain, callback, reason)

    def _inside_warp_cuda_window(self) -> bool:
        return bool(self._session is not None and self._session.paused)

    def _log_cupti_event(
        self, segment: Any, domain: str, callback: str, reason: str
    ) -> None:
        segment_id = id(segment)
        key = (domain, callback, reason)
        warned = self._warned_by_segment.setdefault(segment_id, set())
        if key in warned:
            return
        warned.add(key)

        node_name = getattr(segment, "node_name", "<unknown>")
        segment_name = getattr(segment, "proxy_name", None) or node_name
        _get_logger().info(
            "CUDA boundary while Warp segment is open "
            f"(node={node_name}, segment={segment_name}, "
            f"reason={reason}, callback={domain}.{callback})"
        )

    def _domain_name(self, domain_id: int) -> str:
        return self._enum_name(cupti.CallbackDomain, int(domain_id)) or (
            f"domain#{domain_id}"
        )

    def _callback_name(self, domain: str, cbid: int) -> str:
        enum_type = {
            "DRIVER_API": _DRIVER_API_TRACE_CBID,
            "RUNTIME_API": _RUNTIME_API_TRACE_CBID,
            "SYNCHRONIZE": cupti.CallbackIdSync,
            "RESOURCE": cupti.CallbackIdResource,
            "STATE": cupti.CallbackIdState,
        }.get(domain)
        if enum_type is not None:
            name = self._enum_name(enum_type, int(cbid))
            if name is not None:
                return name
        return f"cbid#{cbid}"

    def _trip_reason(self, callback: str) -> str | None:
        if any(fragment in callback for fragment in self._SYNC_READBACK_FRAGMENTS):
            return "sync_readback"
        if any(fragment in callback for fragment in self._CUDA_WORK_FRAGMENTS):
            return "foreign_cuda"
        return None

    @staticmethod
    def _enum_name(enum_type: Any, value: int) -> str | None:
        try:
            return enum_type(value).name
        except Exception:
            return None
