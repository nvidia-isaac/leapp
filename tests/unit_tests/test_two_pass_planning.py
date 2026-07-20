from leapp.leapp_graph.datatypes.warp import WarpSegment
from leapp.leapp_graph.datatypes.warp.patching import WarpPatchBackend
import unittest


class TestWarpTwoPassPlanning(unittest.TestCase):
    def test_discovery_open_close_records_segment_bookmark(self):
        backend = WarpPatchBackend()
        segment = backend.begin_discovery_segment(
            node_name="node",
            call_stack=("caller.py", 10, "func"),
        )

        segment.add_event({"kind": "warp_call", "qualname": "warp.launch"})
        segment.add_event({"kind": "warp_call", "qualname": "warp.copy"})

        discovered = backend.end_discovery_segment(
            segment,
            ("caller.py", 10, "func"),
        )

        self.assertIs(discovered, segment)
        self.assertEqual(discovered.node_name, "node")
        self.assertEqual(discovered.call_qualnames, ("warp.launch", "warp.copy"))
        self.assertIsNone(backend.active_segment)

    def test_discovery_close_noops_for_mismatched_call_stack(self):
        backend = WarpPatchBackend()
        segment = backend.begin_discovery_segment(
            node_name="node",
            call_stack=("caller.py", 10, "func"),
        )

        closed = backend.end_discovery_segment(
            segment,
            ("other.py", 20, "other"),
        )

        self.assertIsNone(closed)
        self.assertIs(backend.active_segment, segment)
        backend.deactivate_segment(segment)

    def test_capture_close_fails_on_divergent_call_sequence(self):
        backend = WarpPatchBackend()
        discovered = WarpSegment(
            node_name="node",
            call_stack=("caller.py", 10, "func"),
            call_qualnames=("warp.launch",),
            status="closed",
            events=[{"kind": "warp_call", "qualname": "warp.launch"}],
            input_refs={"input_0": object()},
            output_refs={"output_0": object()},
            marker_proxy=object(),
            proxy_name="warp_segment_0",
        )
        segment = backend.begin_capture_segment(
            segment=discovered,
        )
        self.assertIs(segment, discovered)
        self.assertEqual(segment.status, "open")
        self.assertEqual(segment.events, [])
        self.assertEqual(segment.input_refs, {})
        self.assertEqual(segment.output_refs, {})
        self.assertIsNone(segment.marker_proxy)
        self.assertIsNone(segment.proxy_name)
        self.assertEqual(segment.call_qualnames, ("warp.launch",))
        segment.add_event({"kind": "warp_call", "qualname": "warp.copy"})

        with self.assertRaisesRegex(RuntimeError, "diverged"):
            backend.end_capture_segment(
                segment,
                ("caller.py", 10, "func"),
            )

        self.assertIsNone(backend.active_segment)
