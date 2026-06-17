from leapp.utils.logging import _get_logger
from leapp.leapp_graph.traced_node import TracedTensorNode

try:
    import warp as wp
except ModuleNotFoundError as exc:
    if exc.name != "warp":
        raise
    wp = None
    WarpOp = None
else:
    # fx marker.
    def warp_operation_id(segment_id, *inputs):
        raise RuntimeError("leapp_warp_operation is an FX marker and should be lowered before execution")

    class WarpOp:
        def __init__(self, node_ref: TracedTensorNode, device: str = "cuda:0"):
            if not isinstance(node_ref, TracedTensorNode):
                _get_logger().error(f"LEAPP: warp_op received a non-TracedTensorNode reference: {type(node_ref)}")
                raise ValueError(f"LEAPP: warp_op received a non-TracedTensorNode reference: {type(node_ref)}")
            self.node_ref = node_ref
            self.node_name = node_ref.name
            self.node_graph = node_ref.graph

            # scoped capture variables
            self._scope = None
            self._capture = None
            self.device = device

        def __enter__(self):
            self._scope = wp.ScopedCapture(
                device=self.device,
                force_module_load=True,
                apic=True,
            )
            self._capture = self._scope.__enter__()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self._scope is not None:
                self._scope.__exit__(exc_type, exc_value, traceback)

            if exc_type is None:
                self.graph = self._capture.graph
                # later:
                # wp.capture_save(self.graph, path, inputs=..., outputs=...)


            proxy_out = self.node_ref.tracer.create_proxy(
                "call_function",
                warp_operation_id,
                self.inputs,
                {},
                name="warp_operation",
            )

            proxy_out.node.meta["apic_graph"] = self.graph
            
            return False