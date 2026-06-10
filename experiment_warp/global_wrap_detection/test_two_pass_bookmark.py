"""Prove the two-pass bookmark / inject model for Warp APIC capture.

The user runs the whole pipeline TWICE:

  PASS 1 (record): everything runs eagerly (no capture). We tag each Warp
    launch with an ordinal at the dispatch boundary and record a "break"
    whenever a tracked-array host egress happens between launches. Segments =
    maximal runs of launches with no break. This yields bookmarks:
    [(start_ordinal, end_ordinal), ...].

  PASS 2 (inject): the SAME pipeline runs again. At the dispatch boundary we
    gate purely by ordinal: capture_begin before a segment's first launch,
    and capture_end -> capture_launch -> sync -> capture_save after its last.
    The replay-on-close means a .numpy() between segments observes real data.

Pipeline (one egress splits it into two segments):
    seg A: summed = a + b ; scaled = summed*2          --\
    [egress: read scaled.numpy()]  <- break              |  bookmark boundary
    seg B: averaged = scaled * 0.5                      --/

Proves:
  - manual capture_begin/capture_end gated mid-run produces valid .wrp,
  - a contiguous segment of >1 launch fuses into ONE .wrp,
  - the egress between segments sees correct eager values in pass 2,
  - per-segment external I/O (tracked arrays) is derived from the launches.

Run:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate exp_env
    python experiment_warp/global_wrap_detection/test_two_pass_bookmark.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import warp as wp


DEVICE = "cuda:0"
N = 8


@wp.kernel
def add_fields(a: wp.array(dtype=float), b: wp.array(dtype=float), summed: wp.array(dtype=float)):
    i = wp.tid()
    summed[i] = a[i] + b[i]


@wp.kernel
def scale(summed: wp.array(dtype=float), scaled: wp.array(dtype=float)):
    i = wp.tid()
    scaled[i] = summed[i] * 2.0


@wp.kernel
def finalize(scaled: wp.array(dtype=float), averaged: wp.array(dtype=float)):
    i = wp.tid()
    averaged[i] = scaled[i] * 0.5


# ---------------------------------------------------------------------------
# Minimal "warp_patching"-style driver. In the real design this wraps the
# low-level dispatch funnel; here we wrap wp.launch as a faithful stand-in.
# ---------------------------------------------------------------------------
class CaptureDriver:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.mode = "record"          # "record" | "replay"
        self.ordinal = 0
        # record-mode state
        self.events: list = []        # ("launch", ordinal, ins, outs) | ("egress",)
        # replay-mode state
        self.segments: list[tuple[int, int]] = []
        self.seg_io: list[dict] = []  # per-segment {"inputs": {...}, "outputs": {...}}
        self._active = None           # (seg_index, graph) when a capture is open
        self.saved_paths: list[str] = []

    # ---- pass 1: record -------------------------------------------------
    def record_launch(self, kernel, dim, inputs, outputs):
        self.ordinal += 1
        self.events.append(("launch", self.ordinal, list(inputs), list(outputs)))
        wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=DEVICE)

    def record_egress(self):
        self.events.append(("egress",))

    def finalize_bookmarks(self):
        """Turn the event log into segments + per-segment external tracked I/O."""
        segments = []
        cur: list[int] = []
        launch_args: dict[int, tuple] = {}
        for ev in self.events:
            if ev[0] == "launch":
                _, ordn, ins, outs = ev
                launch_args[ordn] = (ins, outs)
                cur.append(ordn)
            else:  # egress -> break
                if cur:
                    segments.append((cur[0], cur[-1]))
                    cur = []
        if cur:
            segments.append((cur[0], cur[-1]))

        # Derive external I/O per segment: inputs read but not produced inside
        # the segment; outputs written inside the segment.
        seg_io = []
        for (start, end) in segments:
            produced: list = []
            inputs: dict = {}
            outputs: dict = {}
            for ordn in range(start, end + 1):
                ins, outs = launch_args[ordn]
                for arr in ins:
                    if isinstance(arr, wp.array) and not any(arr is p for p in produced):
                        inputs[f"in_{id(arr)}"] = arr
                for arr in outs:
                    if isinstance(arr, wp.array):
                        produced.append(arr)
                        outputs[f"out_{id(arr)}"] = arr
            seg_io.append({"inputs": inputs, "outputs": outputs})

        self.segments = segments
        self.seg_io = seg_io

    def _seg_index_for_start(self, ordn):
        for i, (s, _e) in enumerate(self.segments):
            if s == ordn:
                return i
        return None

    def _seg_index_for_end(self, ordn):
        for i, (_s, e) in enumerate(self.segments):
            if e == ordn:
                return i
        return None

    # ---- pass 2: replay/inject -----------------------------------------
    def replay_launch(self, kernel, dim, inputs, outputs):
        self.ordinal += 1
        ordn = self.ordinal

        start_idx = self._seg_index_for_start(ordn)
        if start_idx is not None:
            wp.capture_begin(device=DEVICE, force_module_load=True, apic=True)
            self._active = start_idx

        wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=DEVICE)

        end_idx = self._seg_index_for_end(ordn)
        if end_idx is not None:
            graph = wp.capture_end(device=DEVICE)
            wp.capture_launch(graph)          # replay so buffers hold real data
            wp.synchronize_device(DEVICE)
            io = self.seg_io[end_idx]
            path = str(self.workdir / f"segment_{end_idx}")
            wp.capture_save(graph, path, inputs=io["inputs"], outputs=io["outputs"])
            self.saved_paths.append(path)
            self._active = None


def run_pipeline(driver: CaptureDriver, arrays: dict) -> np.ndarray:
    """The user's pipeline. Run verbatim in both passes."""
    a, b = arrays["a"], arrays["b"]
    summed, scaled, averaged = arrays["summed"], arrays["scaled"], arrays["averaged"]
    launch = driver.record_launch if driver.mode == "record" else driver.replay_launch

    launch(add_fields, N, [a, b], [summed])
    launch(scale, N, [summed], [scaled])

    # egress between segments: read a tracked array to host
    wp.synchronize_device(DEVICE)
    mid = scaled.numpy().copy()
    if driver.mode == "record":
        driver.record_egress()

    launch(finalize, N, [scaled], [averaged])
    wp.synchronize_device(DEVICE)
    return mid


def fresh_arrays(a_np, b_np):
    return {
        "a": wp.array(a_np, dtype=float, device=DEVICE),
        "b": wp.array(b_np, dtype=float, device=DEVICE),
        "summed": wp.zeros(N, dtype=float, device=DEVICE),
        "scaled": wp.zeros(N, dtype=float, device=DEVICE),
        "averaged": wp.zeros(N, dtype=float, device=DEVICE),
    }


def main() -> None:
    wp.init()
    workdir = Path(tempfile.mkdtemp(prefix="two_pass_"))
    try:
        a_np = np.arange(N, dtype=np.float32)
        b_np = np.full(N, 10.0, dtype=np.float32)
        ref_scaled = (a_np + b_np) * 2.0
        ref_averaged = ref_scaled * 0.5

        driver = CaptureDriver(workdir)

        # ---- PASS 1: record bookmarks (everything eager) ----------------
        driver.mode = "record"
        driver.ordinal = 0
        arrays1 = fresh_arrays(a_np, b_np)
        mid1 = run_pipeline(driver, arrays1)
        driver.finalize_bookmarks()
        pass1_egress_ok = bool(np.allclose(mid1, ref_scaled))

        # ---- PASS 2: inject capture by ordinal --------------------------
        driver.mode = "replay"
        driver.ordinal = 0
        arrays2 = fresh_arrays(a_np, b_np)
        mid2 = run_pipeline(driver, arrays2)
        pass2_egress_ok = bool(np.allclose(mid2, ref_scaled))

        # ---- VERIFY each saved segment .wrp reproduces correctly --------
        seg_results = []
        for i, path in enumerate(driver.saved_paths):
            io = driver.seg_io[i]
            loaded = wp.capture_load(path, device=DEVICE)
            # rebind external inputs to fresh buffers w/ known values
            for name, arr in io["inputs"].items():
                src = wp.array(arr.numpy(), dtype=float, device=DEVICE)
                loaded.set_param(name, src)
            wp.capture_launch(loaded)
            wp.synchronize_device(DEVICE)
            out_vals = {}
            for name, arr in io["outputs"].items():
                buf = wp.empty(N, dtype=float, device=DEVICE)
                loaded.get_param(name, buf)
                out_vals[name] = buf.numpy()
            seg_results.append(out_vals)

        # segment 0 produced summed & scaled; segment 1 produced averaged
        seg0_scaled_ok = any(np.allclose(v, ref_scaled) for v in seg_results[0].values())
        seg1_avg_ok = any(np.allclose(v, ref_averaged) for v in seg_results[1].values())

        num_segments = len(driver.segments)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print("=" * 72)
    print(f"pass 1 detected segments (bookmarks)         : {driver.segments}")
    print(f"  -> 2 segments split by one egress          : {'PASS' if num_segments == 2 else 'FAIL'}")
    print(f"pass 1 egress saw correct eager value        : {'PASS' if pass1_egress_ok else 'FAIL'}")
    print(f"pass 2 egress saw correct eager value        : {'PASS' if pass2_egress_ok else 'FAIL'}")
    print(f"  (proves capture_launch-on-close restores data mid-run)")
    print(f"segment 0 (.wrp, 2 fused launches) correct   : {'PASS' if seg0_scaled_ok else 'FAIL'}")
    print(f"segment 1 (.wrp) correct                     : {'PASS' if seg1_avg_ok else 'FAIL'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
