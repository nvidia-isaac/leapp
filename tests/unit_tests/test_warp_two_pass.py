#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest

from leapp.leapp_graph.datatypes.warp import WarpSegment
from leapp.leapp_graph.datatypes.warp.patching import WarpPatchBackend
from leapp.leapp_graph.traced_node import TracedTensorNode


class TestWarpTwoPassPlanning(unittest.TestCase):
    def test_discovery_open_close_records_segment_bookmark(self):
        backend = WarpPatchBackend()
        segment = backend.begin_discovery_segment(
            node_name="node",
            segment_ordinal=0,
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
        self.assertEqual(discovered.segment_ordinal, 0)
        self.assertEqual(discovered.call_qualnames, ("warp.launch", "warp.copy"))
        self.assertIsNone(backend.active_segment)

    def test_discovery_close_noops_for_mismatched_call_stack(self):
        backend = WarpPatchBackend()
        segment = backend.begin_discovery_segment(
            node_name="node",
            segment_ordinal=0,
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
            segment_ordinal=0,
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

    def test_node_warp_pending_state_comes_from_segment_graph(self):
        node = TracedTensorNode("node")
        discovered = WarpSegment(
            node_name="node",
            segment_ordinal=0,
            call_stack=("caller.py", 10, "func"),
        )

        self.assertTrue(node.is_tracing)
        self.assertFalse(node.has_pending_warp_segments)
        self.assertEqual(node.next_warp_segment_ordinal(), 0)
        node.add_warp_segment(discovered)
        self.assertTrue(node.has_pending_warp_segments)

        self.assertIs(node.get_warp_segment(0), discovered)
        discovered.apic_graph = object()
        node.complete_warp_segment(discovered)
        self.assertFalse(node.has_pending_warp_segments)
        self.assertIs(node.get_warp_segment(0), discovered)

        node.reset_trace_state()
        self.assertEqual(node.next_warp_segment_ordinal(), 0)


if __name__ == "__main__":
    unittest.main()
