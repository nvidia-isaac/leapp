#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Non-invasive capture of a Warp region into a LEAPP warp node.

The user wraps a region of their EXISTING Warp code in ``with leapp.warp_node(name) as wn:`` and
changes nothing else — their ``@wp.kernel`` definitions, ``wp.launch`` calls, and ``wp.array``
allocations are untouched.

The actual serialization rides Warp's own APIC capture (``wp.ScopedCapture(apic=True)`` +
``wp.capture_save``) — we do NOT reimplement the ``.wrp`` format or the capture engine. What we
cannot do is open that capture *in-line* around the user's region: CUDA-graph capture forbids GPU
memory allocation, and users routinely allocate intermediates/outputs inside the region (doing so
segfaults). So we run the region **eagerly** (allocations happen normally and the user gets correct
results), monkey-patch ``wp.launch`` to record each call, and then **replay just the recorded
launches into one APIC capture** to produce the ``.wrp`` (arrays already allocated -> nothing is
allocated during the capture). Region I/O is auto-detected from buffer read/write order:

    input  = a ``wp.array`` whose FIRST touch in the region is a read
    output = a ``wp.array`` whose LAST  touch in the region is a write

Scope/limits (v1): captures ``wp.launch``-based regions (the common case). Warp ops that do NOT go
through ``wp.launch`` (e.g. some Newton solver internals using ``wp.copy`` etc.) are not recorded
and so not captured — patch those entry points too, or use an explicit ``wp.ScopedCapture`` for
such regions. I/O names are auto-assigned ``in{i}``/``out{i}``.
"""
import inspect
import os

from leapp.backends.warp_export_backend import save_warp_node


class WarpNodeCapture:
    """Context manager: record plain ``wp.launch`` calls (run eagerly) and serialize the region's
    captured APIC graph as a LEAPP warp node."""

    def __init__(self, name: str, save_path: str = ".", device: str = None):
        self.name = name
        self.save_path = save_path
        self.device = str(device) if device else None
        self._wp = None
        self._orig_launch = None
        self._records = []
        # Populated on __exit__:
        self.node = None        # the LEAPP YAML ``models`` entry (backend: warp)
        self.inputs = {}        # auto-detected name -> wp.array
        self.outputs = {}
        self.wrp_path = None

    def __enter__(self):
        import warp as wp
        self._wp = wp
        self._orig_launch = wp.launch
        sig = inspect.signature(self._orig_launch)
        records, orig = self._records, self._orig_launch

        def patched_launch(*args, **kwargs):
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                ins = list(bound.arguments.get("inputs") or [])
                outs = list(bound.arguments.get("outputs") or [])
                device = bound.arguments.get("device")
            except TypeError:
                ins, outs, device = [], [], None
            records.append({"args": args, "kwargs": kwargs, "inputs": ins, "outputs": outs, "device": device})
            return orig(*args, **kwargs)  # EAGER — user gets correct results; allocations are normal

        wp.launch = patched_launch
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_launch is not None:
            self._wp.launch = self._orig_launch
        if exc_type is not None:
            return False
        if not self._records:
            raise RuntimeError(
                f"leapp.warp_node('{self.name}'): no wp.launch calls were recorded in the block")

        wp = self._wp
        self.inputs, self.outputs = self._detect_io(wp)
        if not self.outputs:
            raise RuntimeError(f"leapp.warp_node('{self.name}'): could not detect any output arrays")
        device = self._resolve_device(wp)

        # Kernels are already compiled (the eager pass above launched them); ScopedCapture with
        # force_module_load=True loads them before capture. Replay ONLY the recorded launches into
        # the APIC capture; arrays already exist (allocated eagerly), so nothing allocates in-capture.
        with wp.ScopedCapture(device=device, force_module_load=True, apic=True) as cap:
            for r in self._records:
                self._orig_launch(*r["args"], **r["kwargs"])

        os.makedirs(self.save_path, exist_ok=True)
        self.node = save_warp_node(cap.graph, self.save_path, self.name,
                                   inputs=self.inputs, outputs=self.outputs)
        self.wrp_path = os.path.join(self.save_path, f"{self.name}.wrp")
        return False

    def _detect_io(self, wp):
        first, last, arrs = {}, {}, {}
        for r in self._records:
            for a in r["inputs"]:
                if isinstance(a, wp.array):
                    first.setdefault(a.ptr, "read"); last[a.ptr] = "read"; arrs[a.ptr] = a
            for a in r["outputs"]:
                if isinstance(a, wp.array):
                    first.setdefault(a.ptr, "write"); last[a.ptr] = "write"; arrs[a.ptr] = a
        inputs = {f"in{i}": arrs[p] for i, p in enumerate(p for p in first if first[p] == "read")}
        outputs = {f"out{i}": arrs[p] for i, p in enumerate(p for p in last if last[p] == "write")}
        return inputs, outputs

    def _resolve_device(self, wp):
        if self.device:
            return self.device
        for r in self._records:
            if r["device"] is not None:
                return str(r["device"])
        for r in self._records:
            for a in r["inputs"] + r["outputs"]:
                if isinstance(a, wp.array):
                    return str(a.device)
        return "cuda:0"


def warp_node(name: str, save_path: str = ".", device: str = None) -> "WarpNodeCapture":
    """Non-invasively capture a region of plain Warp code as a LEAPP warp node.

    Example::

        import warp as wp, leapp
        with leapp.warp_node("solver", save_path="out") as wn:
            wp.launch(kernel_a, dim=n, inputs=[a], outputs=[b], device="cuda")
            wp.launch(kernel_b, dim=n, inputs=[b], outputs=[c], device="cuda")
        wn.node      # YAML ``models`` entry (backend: warp) for a LEAPP bundle / the
                     # leapp_runtimes.triton generator
        wn.wrp_path  # path to the emitted <name>.wrp

    The kernels / launches / arrays are unchanged; only the surrounding ``with`` is added. The
    region runs eagerly (correct results); the recorded launches are replayed into a Warp APIC
    capture to produce the ``.wrp``. I/O is auto-detected (``in{i}``/``out{i}``).
    """
    return WarpNodeCapture(name, save_path=save_path, device=device)
