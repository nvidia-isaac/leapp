import gc
import os

import numpy as np
import warp as wp


wp.init()

DEVICE = "cuda:0"
HEIGHT = 4
WIDTH = 5
GRAPH_PATH = os.path.join(os.path.dirname(__file__), "kernel_chain_2d")


@wp.kernel
def add_fields(a: wp.array2d(dtype=float), b: wp.array2d(dtype=float), summed: wp.array2d(dtype=float)):
    row, col = wp.tid()
    summed[row, col] = a[row, col] + b[row, col]


@wp.kernel
def scale_and_bias(summed: wp.array2d(dtype=float), scale: float, bias: float, scaled: wp.array2d(dtype=float)):
    row, col = wp.tid()
    scaled[row, col] = summed[row, col] * scale + bias


@wp.kernel
def neighbor_average(scaled: wp.array2d(dtype=float), averaged: wp.array2d(dtype=float)):
    row, col = wp.tid()

    center = scaled[row, col]
    left = center
    right = center
    up = center
    down = center

    if col > 0:
        left = scaled[row, col - 1]
    if col + 1 < WIDTH:
        right = scaled[row, col + 1]
    if row > 0:
        up = scaled[row - 1, col]
    if row + 1 < HEIGHT:
        down = scaled[row + 1, col]

    averaged[row, col] = (center + left + right + up + down) * 0.2


def _as_warp_array(value: wp.array) -> wp.array:
    """Accept raw or traced warp arrays for APIC / wp.launch calls."""
    from leapp.leapp_graph.datatypes import TracedData

    unwrapped = TracedData.unwrap_traced_data(value)
    if not isinstance(unwrapped, wp.array):
        raise TypeError(f"Expected wp.array, got {type(unwrapped).__name__}")
    return unwrapped


def run_kernel_chain(a, b, summed, scaled, averaged) -> None:
    shape = (HEIGHT, WIDTH)
    wp.launch(add_fields, dim=shape, inputs=[a, b], outputs=[summed], device=DEVICE)
    wp.launch(scale_and_bias, dim=shape, inputs=[summed, 2.0, 1.0], outputs=[scaled], device=DEVICE)
    wp.launch(neighbor_average, dim=shape, inputs=[scaled], outputs=[averaged], device=DEVICE)


def main() -> None:
    import leapp
    from leapp import annotate
    from leapp.leapp_graph.datatypes.traced_warp_array import TracedWarpArray

    trace_dir = os.path.join(os.path.dirname(__file__), "leapp_trace")
    leapp.start(
        name="warp_kernel_chain",
        save_path=trace_dir,
        global_patching=False,
        verbose=True,
    )

    a_np = np.arange(HEIGHT * WIDTH, dtype=np.float32).reshape(HEIGHT, WIDTH)
    b_np = np.full((HEIGHT, WIDTH), 10.0, dtype=np.float32)

    a_raw = wp.array(a_np, dtype=float, device=DEVICE)
    b_raw = wp.array(b_np, dtype=float, device=DEVICE)
    summed = wp.zeros((HEIGHT, WIDTH), dtype=float, device=DEVICE)
    scaled = wp.zeros((HEIGHT, WIDTH), dtype=float, device=DEVICE)
    averaged = wp.zeros((HEIGHT, WIDTH), dtype=float, device=DEVICE)

    a, b = annotate.input_tensors("kernel_chain", {"a": a_raw, "b": b_raw})
    import pdb; pdb.set_trace()
    print(f"wrapped a: {type(a).__name__}, is TracedWarpArray={isinstance(a, TracedWarpArray)}")
    print(f"wrapped b: {type(b).__name__}, is TracedWarpArray={isinstance(b, TracedWarpArray)}")
    print(f"shared storage a: {a.ptr == a_raw.ptr}, b: {b.ptr == b_raw.ptr}")

    # APIC records a fixed 2D launch sequence: add -> scale/bias -> neighbor average.
    with wp.ScopedCapture(device=DEVICE, force_module_load=True, apic=True) as capture:
        run_kernel_chain(a, b, summed, scaled, averaged)

    annotate.output_tensors("kernel_chain", {"averaged": averaged}, export_with=None)

    graph = capture.graph
    wp.capture_launch(graph)
    wp.synchronize_device(DEVICE)
    print("in-process averaged:")
    print(averaged.numpy())

    wp.capture_save(
        graph,
        GRAPH_PATH,
        inputs={"a": a, "b": b},
        outputs={"summed": summed, "scaled": scaled, "averaged": averaged},
    )

    del graph, capture, a, b, summed, scaled, averaged
    gc.collect()

    loaded = wp.capture_load(GRAPH_PATH, device=DEVICE)
    wp.capture_launch(loaded)
    wp.synchronize_device(DEVICE)

    out = wp.empty((HEIGHT, WIDTH), dtype=float, device=DEVICE)
    loaded.get_param("averaged", out)
    print("loaded averaged:")
    print(out.numpy())

    leapp.stop()
    leapp.compile_graph(visualize=False, validate=False, dry_run=True)
    print(f"LEAPP trace artifacts written under: {trace_dir}/warp_kernel_chain")


if __name__ == "__main__":
    main()
