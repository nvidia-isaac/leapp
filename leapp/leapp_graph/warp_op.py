from typing import TYPE_CHECKING
from leapp.utils.logging import _get_logger

if TYPE_CHECKING:
    from leapp.leapp_graph.traced_node import TracedTensorNode

try:
    import warp as wp
    from leapp.leapp_graph.datatypes.global_warp_patching import WarpLeappCallDetector
except ModuleNotFoundError as exc:
    if exc.name != "warp":
        raise
    wp = None
    WarpOp = None
else:
    class WarpOp:
        def __init__(self, node_ref: "TracedTensorNode", device: str = "cuda:0"):
            self.node_ref = node_ref
            self.node_name = node_ref.name
            self.node_graph = node_ref.graph

            # scoped capture variables
            self._scope = None
            self._capture = None
            self._segment = None
            self._detector = None
            self.device = device

        def __enter__(self):
            self._segment = self.node_ref.add_warp_segment(
                device=self.device,
            )
            self._detector = WarpLeappCallDetector.instance()
            self._detector.push_segment(self._segment)
            self._scope = wp.ScopedCapture(
                device=self.device,
                force_module_load=True,
                apic=True,
            )
            self._capture = self._scope.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            scope_result = False
            scope_result = self._scope.__exit__(exc_type, exc_value, traceback)
            if self.node_ref.get_warp_segment() is not self._segment:
                raise ValueError(f"Warp segment {self._segment} is not the "
                                 f"current segment for node {self.node_ref.name}")
            try:
                if exc_type is None:
                    graph = self._capture.graph
                    # ScopedCapture only records the kernels; replay the graph here
                    # so the real buffers advance at trace time. Suppress detection
                    # during replay so the patched ``wp.capture_launch`` /
                    # ``wp.synchronize`` calls do not append spurious events to the
                    # still-active segment.
                    with self._detector.paused():
                        # still need to execute the graph to get outputs
                        wp.capture_launch(graph)
                        wp.synchronize()
                    if self._segment is not None:
                        self._segment.apic_graph = graph
                        self._segment.add_event({"kind": "scoped_capture"})
                        self.node_ref.close_warp_segment(self._segment)
            finally:
                if self._detector is not None:
                    self._detector.pop_segment(self._segment)

            return scope_result
