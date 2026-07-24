from leapp.leapp_graph.datatypes import as_traced
from leapp.leapp_graph.datatypes.warp import WarpSegment
from leapp.leapp_graph.datatypes.warp import TracedWpArray, wp
from leapp.leapp_graph.datatypes.warp.patching import WarpPatchBackend
from leapp.leapp_graph.datatypes.warp.session import WarpTraceSession
from leapp.leapp_graph.warp_op import WarpOp
import unittest


class _FakeNode:
    name = "node"
    graph = None
    is_warp_capture_active = False


class _FakeWarpOp:
    def __init__(self, segment):
        self.segment = segment

    def terminate(self, *args, **kwargs):
        return self.segment


class TestWarpTwoPassPlanning(unittest.TestCase):
    def test_warp_array_tracing_preserves_identity_and_allocation_owner(self):
        array = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu")
        original_id = id(array)
        original_deleter = array.deleter

        traced = as_traced(array, "array", _FakeNode(), object())

        self.assertIs(traced, array)
        self.assertEqual(id(traced), original_id)
        self.assertIsInstance(traced, TracedWpArray)
        self.assertIs(traced.deleter, original_deleter)

    def test_warp_op_discovery_close_records_segment_bookmark(self):
        session = WarpTraceSession()
        warp_op = WarpOp(_FakeNode(), session=session, capture=False)
        warp_op.begin(("caller.py", 10, "func"))
        segment = warp_op.segment

        segment.add_event({"kind": "warp_call", "qualname": "warp.launch"})
        segment.add_event({"kind": "warp_call", "qualname": "warp.copy"})

        discovered = warp_op._finalize_discovery(
            close_call_stack=("caller.py", 10, "func"),
        )

        self.assertIs(discovered, segment)
        self.assertEqual(discovered.node_name, "node")
        self.assertEqual(discovered.open_call_stack, ("caller.py", 10, "func"))
        self.assertEqual(discovered.close_call_stack, ("caller.py", 10, "func"))
        self.assertEqual(discovered.call_qualnames, ("warp.launch", "warp.copy"))
        self.assertIs(session.active_segment, segment)

    def test_warp_op_exit_without_begin_is_fatal(self):
        warp_op = WarpOp(
            _FakeNode(),
            session=WarpTraceSession(),
            capture=False,
        )

        with self.assertRaisesRegex(RuntimeError, "before the Warp operation was initialized"):
            warp_op.__exit__(None, None, None)

    def test_backend_call_stack_match_uses_open_call_stack(self):
        backend = WarpPatchBackend()
        session = WarpTraceSession()
        backend._session = session
        segment = WarpSegment(
            node_name="node",
            open_call_stack=("caller.py", 10, "func"),
        )
        warp_op = _FakeWarpOp(segment)
        session.register_warp_op(warp_op, segment)

        matches = backend.call_stack_matches_segment(
            segment,
            ("other.py", 20, "other"),
        )

        self.assertFalse(matches)
        self.assertIs(session.active_segment, segment)
        session.close_warp_segment()

    def test_capture_close_fails_on_divergent_call_sequence(self):
        session = WarpTraceSession()
        discovered = WarpSegment(
            node_name="node",
            open_call_stack=("caller.py", 10, "func"),
            call_qualnames=("warp.launch",),
            status="closed",
            events=[{"kind": "warp_call", "qualname": "warp.launch"}],
            input_refs={"input_0": object()},
            output_refs={"output_0": object()},
            marker_proxy=object(),
            proxy_name="warp_segment_0",
        )
        warp_op = WarpOp(_FakeNode(), session=session, capture=True)
        segment = warp_op._prepare_capture_segment(discovered)
        warp_op._segment = segment
        session.register_warp_op(warp_op, segment)
        self.assertEqual(segment.status, "open")
        self.assertEqual(segment.events, [])
        self.assertEqual(segment.input_refs, {})
        self.assertEqual(segment.output_refs, {})
        self.assertIsNone(segment.marker_proxy)
        self.assertIsNone(segment.proxy_name)
        self.assertEqual(segment.call_qualnames, ("warp.launch",))
        segment.add_event({"kind": "warp_call", "qualname": "warp.copy"})

        with self.assertRaisesRegex(RuntimeError, "diverged"):
            warp_op._finalize_capture()

        self.assertIs(session.active_segment, segment)

    def test_capture_exit_fails_when_close_watcher_misses(self):
        session = WarpTraceSession()
        warp_op = WarpOp(_FakeNode(), session=session, capture=True)
        segment = WarpSegment(node_name="node", status="open")

        warp_op._segment = segment
        session.register_warp_op(warp_op, segment, owner_token=warp_op)

        with self.assertRaisesRegex(RuntimeError, "close call stack was detected"):
            warp_op.__exit__(None, None, None)
