from leapp.leapp_graph.datatypes import TracedData, as_traced
from leapp.leapp_graph.datatypes.warp import WarpSegment
from leapp.leapp_graph.datatypes.warp import TracedWpArray, wp
from leapp.leapp_graph.datatypes.warp.patching import WarpPatchBackend
from leapp.leapp_graph.datatypes.warp.session import WarpTraceSession
from leapp.leapp_graph.warp_op import WarpOp
from leapp.utils.caller_identity import CallerFrame, CallerIdentity
import types
import unittest
from unittest.mock import patch


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
        proxy = object()

        context = _FakeNode()
        traced = as_traced(array, "array", context, proxy)

        self.assertIs(traced, array)
        self.assertEqual(id(traced), original_id)
        self.assertIsInstance(traced, TracedWpArray)
        self.assertIsInstance(traced, TracedData)
        self.assertTrue(issubclass(TracedWpArray, TracedData))
        self.assertIs(traced.deleter, original_deleter)
        self.assertIs(traced.proxy, proxy)
        self.assertEqual(traced.name, "array")
        self.assertEqual(traced.context, "node")
        self.assertIs(traced.context_obj, context)
        self.assertIsNone(traced.output_port)

        traced.output_port = "output"
        self.assertEqual(traced.output_port, "output")

        raw = traced.data
        self.assertIs(type(raw), wp.array)
        self.assertEqual(raw.ptr, traced.ptr)
        self.assertIs(raw._ref, traced)

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

    def test_backend_install_failure_rolls_back_installed_attributes(self):
        backend = WarpPatchBackend()
        module = types.ModuleType("warp_patch_target")

        def original():
            return "original"

        def wrapper():
            return "wrapper"

        module.target = original

        def install_patch():
            backend._patches.install(module, "target", original, wrapper)

        oracle_path = (
            "leapp.leapp_graph.datatypes.warp.patching.WarpCudaOracle"
        )
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(oracle_path) as oracle_class,
            patch.object(backend, "_register_boundary_functions"),
            patch.object(
                backend,
                "_patch_warp_modules",
                side_effect=install_patch,
            ),
            patch.object(
                backend,
                "_patch_loaded_aliases",
                side_effect=RuntimeError("installation failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "installation failed"):
                backend.install()

        self.assertIs(module.target, original)
        self.assertFalse(backend.installed)
        self.assertIsNone(backend._session)
        oracle_class.return_value.stop.assert_called_once()

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

    def test_capture_close_fails_on_divergent_close_site(self):
        expected_close = CallerIdentity(
            CallerFrame("discovery.py", 10, "expected_close"),
        )
        attempted_close = CallerIdentity(
            CallerFrame("capture.py", 20, "attempted_close"),
        )
        warp_op = WarpOp(_FakeNode(), session=WarpTraceSession(), capture=True)
        warp_op._segment = WarpSegment(
            node_name="node",
            close_call_stack=expected_close,
            status="open",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "closed at a different annotation origin",
        ) as raised:
            warp_op._validate_capture_close(attempted_close)
        self.assertIn("discovery.py:10 in expected_close", str(raised.exception))
        self.assertIn("capture.py:20 in attempted_close", str(raised.exception))
        self.assertEqual(warp_op.segment.status, "invalid")

    def test_owned_capture_rejects_different_requester(self):
        session = WarpTraceSession()
        segment = WarpSegment(node_name="node", status="open")
        warp_op = _FakeWarpOp(segment)
        session.register_warp_op(
            warp_op,
            segment,
            owner_token=warp_op,
        )

        closed = session.close_warp_segment(
            requester=object(),
            close_call_stack=("capture.py", 20, "attempted_close"),
        )
        self.assertFalse(closed)
        self.assertIs(session.active_segment, segment)
        session.close_warp_segment(requester=warp_op)

    def test_explicit_capture_exit_closes_owned_segment_directly(self):
        session = WarpTraceSession()
        warp_op = WarpOp(_FakeNode(), session=session, capture=True)
        close_call_stack = ("caller.py", 10, "close")
        segment = WarpSegment(
            node_name="node",
            close_call_stack=close_call_stack,
            status="open",
        )

        warp_op._segment = segment
        session.register_warp_op(warp_op, segment, owner_token=warp_op)

        with patch(
            "leapp.leapp_graph.warp_op.get_caller_stack_identity",
            return_value=close_call_stack,
        ):
            warp_op.__exit__(RuntimeError, RuntimeError("failure"), None)

        self.assertIsNone(session.active_segment)
        self.assertIsNone(session.active_warp_op)
