from __future__ import annotations

import sys
import threading
import traceback
from typing import Any

from cupti import cupti



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

    def __init__(self) -> None:
        self._active_segment: Any | None = None
        self._subscriber: int | None = None
        self._thread_state = threading.local()
        self._warned_by_segment: dict[int, set[tuple[str, str, str]]] = {}
        self._callback = self._on_callback

    def start(self) -> None:
        if self._subscriber is not None:
            return

        self._subscriber = int(cupti.subscribe(self._callback, None))
        cupti.enable_all_domains(1, self._subscriber)

    def stop(self) -> None:
        if self._subscriber is None:
            self._active_segment = None
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
        self._active_segment = None
        self._warned_by_segment.clear()

    def set_segment(self, segment: Any | None) -> None:
        self._active_segment = segment
        if segment is None:
            return
        self._warned_by_segment.setdefault(id(segment), set())

    def set_warp_cuda_allowed(self, allowed: bool) -> bool:
        previous = bool(getattr(self._thread_state, "warp_cuda_allowed", False))
        self._thread_state.warp_cuda_allowed = allowed
        return previous

    def _on_callback(
        self,
        _userdata: Any,
        domain_id: int,
        cbid: int,
        _cbdata: Any,
    ) -> None:
        segment = self._active_segment
        if segment is None:
            return

        domain = self._domain_name(domain_id)
        callback = self._callback_name(domain, cbid)
        reason = self._trip_reason(callback)
        if reason is None:
            return
        if reason == "foreign_cuda" and self._inside_warp_cuda_window():
            return

        self._warn(segment, domain, callback, reason)

    def _inside_warp_cuda_window(self) -> bool:
        return bool(getattr(self._thread_state, "warp_cuda_allowed", False))

    def _warn(self, segment: Any, domain: str, callback: str, reason: str) -> None:
        segment_id = id(segment)
        key = (domain, callback, reason)
        warned = self._warned_by_segment.setdefault(segment_id, set())
        if key in warned:
            return
        warned.add(key)

        node_name = getattr(segment, "node_name", "<unknown>")
        segment_name = getattr(segment, "proxy_name", None) or node_name
        stack = "".join(traceback.format_stack())
        print(
            "[LEAPP][Warp][CUPTI] CUDA boundary warning while Warp segment is open\n"
            f"  node: {node_name}\n"
            f"  segment: {segment_name}\n"
            f"  reason: {reason}\n"
            f"  callback: {domain}.{callback}\n"
            f"  python_stack:\n{stack}",
            file=sys.stderr,
        )

    def _domain_name(self, domain_id: int) -> str:
        return self._enum_name(cupti.CallbackDomain, int(domain_id)) or (
            f"domain#{domain_id}"
        )

    def _callback_name(self, domain: str, cbid: int) -> str:
        enum_type = {
            "DRIVER_API": cupti.Driver_api_trace_cbid,
            "RUNTIME_API": cupti.Runtime_api_trace_cbid,
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
