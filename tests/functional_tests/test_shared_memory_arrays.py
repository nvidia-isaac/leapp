#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Tests for several traced carriers over one shared allocation."""

import unittest
from contextlib import nullcontext
import tempfile
from pathlib import Path
import numpy as np
import torch
import warp as wp
import leapp
from leapp.leapp import _MANAGER as annotate
from .base import LEAPPFunctionalTestBase
from tests.warp_support import WarpTestCase


class TestTorchSharedMemory(LEAPPFunctionalTestBase):
    """Provenance when two Torch carriers describe one buffer.

    An operation such as ``detach`` or a full ``[:]`` slice hands back a second
    handle onto the same bytes with the same layout, so both carriers name one
    value and a mutation through either has to be visible through both. Each
    test drives the question all the way to InferenceManager, because a carrier
    reading a stale proxy still exports a graph that runs and returns
    plausible-looking numbers rather than raising.
    """

    def test_view_alias_mutation_replays(self):
        """Replay both aliased output ports through public annotation and YAML reload."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        x = annotate.input_tensors("policy", {"x": torch.arange(6.0)})
        v = x[1:5]
        a = v
        b = v.detach()
        a.add_(5)
        expected = torch.tensor([0.0, 6.0, 7.0, 8.0, 9.0, 5.0])
        torch.testing.assert_close(x.tensor, expected)
        torch.testing.assert_close(b.tensor, expected[1:5])
        annotate.output_tensors("policy", {"base": x, "alias": b}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False, validate=True, strict=True)
        manager = leapp.InferenceManager(
            str(Path(self.TEST_GRAPH_NAME) / f"{self.TEST_GRAPH_NAME}.yaml")
        )
        for fresh in (
            torch.arange(10.0, 16.0),
            torch.tensor([-7.0, 3.0, -1.0, 100.0, 0.25, 8.0]),
        ):
            before = fresh.clone()
            expected = fresh.clone()
            expected[1:5] += 5
            actual = manager.run_policy({"policy/x": fresh})
            torch.testing.assert_close(actual["policy/base"], expected)
            torch.testing.assert_close(actual["policy/alias"], expected[1:5])
            torch.testing.assert_close(fresh, before)

    def test_inplace_dispatch_mutation_reaches_the_alias(self):
        """An in-place operation reaching dispatch rebinds the shared root.

        ``+=`` writes the view object directly, but an in-place operation called
        as a function goes through ``__torch_function__``, which has to replace
        the existing view's proxy rather than start a new root. A new root would
        leave the alias reading the value from before the mutation. The write and
        the read are on opposite carriers here, so only shared provenance
        connects them.
        """
        obs_value = torch.tensor([-1.0, 2.0, -3.0])
        expected = torch.relu(obs_value) * 2.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": obs_value.clone()})

        alias = obs.detach()
        torch.relu_(obs)
        action = alias * 2.0
        targets = [str(node.target) for node in annotate.nodes["policy"].graph.nodes]

        self.assertTrue(
            torch.equal(action.tensor, expected),
            f"eager value diverged: got {action.tensor}, expected {expected}",
        )

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # The mutation is a real operation, so unlike an alias it is recorded.
        self.assertTrue(
            any("relu" in target for target in targets),
            f"the in-place mutation was not recorded: {targets}",
        )
        self.verify_inference_manager(
            source_inputs={"policy/obs": obs_value},
            source_outputs={"policy/action": expected},
        )

    def test_two_declared_aliases_share_one_root_and_one_port(self):
        """One allocation declared twice is one graph value, not two.

        Both names are usable carriers and a mutation through either is visible
        through the other, but only the first declared name becomes a port: a
        second port over those bytes would make the graph accept a runtime
        tensor it never reads.
        """
        obs_value = torch.tensor([1.0, 2.0, 3.0])

        leapp.start(name=self.TEST_GRAPH_NAME)
        for pass_index in range(2):
            buffer = obs_value.clone()
            alias = buffer.detach()

            traced_buffer = annotate.input_tensors("policy", {"buffer": buffer})
            traced_alias = annotate.input_tensors("policy", {"alias": alias})

            if pass_index == 0:
                self.assertIs(
                    traced_buffer.proxy,
                    traced_alias.proxy,
                    "a declared alias did not adopt the declared buffer's root",
                )

            traced_buffer += 1.0
            if pass_index == 0:
                self.assertIs(
                    traced_alias.proxy,
                    traced_buffer.proxy,
                    "a mutation through one declared alias was invisible to the other",
                )

            annotate.output_tensors(
                "policy",
                {"action": traced_alias * 2.0},
                export_with="jit",
            )

        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=0,
        )
        self.verify_inference_manager(
            source_inputs={"policy/buffer": obs_value},
            source_outputs={"policy/action": (obs_value + 1.0) * 2.0},
        )

    def test_output_alias_with_a_surviving_different_root_faults(self):
        """A pre-declaration mutation cannot silently escape through an alias.

        ``a.copy_(source)`` promotes the plain destination in place while
        changing the bytes shared with plain ``b``. Declaring ``b`` afterwards
        gives it a placeholder root, and consuming it keeps that root in the
        graph, so the export would hold two representations of one buffer that
        can diverge.
        """
        b = torch.tensor([1.0, 2.0, 3.0])
        a = b.detach()

        leapp.start(name=self.TEST_GRAPH_NAME)
        source = annotate.input_tensors(
            "policy", {"source": torch.tensor([4.0, 5.0, 6.0])}
        )
        a.copy_(source)
        traced_b = annotate.input_tensors("policy", {"b": b})

        with self.assertRaisesRegex(Exception, "different tracing roots"):
            annotate.output_tensors(
                "policy", {"a": a, "scaled": traced_b * 2.0}, export_with="jit"
            )
        leapp.stop()

    def test_output_alias_over_a_trimmed_declaration_exports(self):
        """A declared alias nothing consumed is pruned rather than a fault.

        The same two roots over one buffer as above, except nothing reads ``b``.
        Its port is trimmed, which leaves ``a``'s root as the only
        representation of those bytes in the graph, so there is nothing left to
        diverge from.
        """
        b = torch.tensor([1.0, 2.0, 3.0])
        a = b.detach()
        source_value = torch.tensor([4.0, 5.0, 6.0])

        leapp.start(name=self.TEST_GRAPH_NAME)
        source = annotate.input_tensors("policy", {"source": source_value.clone()})
        a.copy_(source)
        annotate.input_tensors("policy", {"b": b})
        annotate.output_tensors("policy", {"a": a}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=0,
        )
        self.verify_inference_manager(
            source_inputs={"policy/source": source_value},
            source_outputs={"policy/a": source_value},
        )


class TestWarpSharedMemory(WarpTestCase, LEAPPFunctionalTestBase):
    """Provenance when a Torch carrier and a Warp array describe one buffer.

    ``wp.from_torch`` and ``wp.to_torch`` hand back a second view of the same
    allocation, so a Warp segment writing that buffer changes the value every
    carrier over it reports. This is the case the Torch tests cannot reach: the
    mutation happens inside an opaque ``leapp::warp_runner`` node, and a carrier
    left on the pre-segment proxy produces a graph whose runner has no consumers
    and gets pruned, so the export silently returns the unmodified input.
    """

    def test_strided_torch_views_bind_one_warp_allocation(self):
        """Replay Torch and Warp writes through sibling views after artifact reload."""
        for backend in ("onnx", "pt2"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as folder:
                leapp.start(name=self.TEST_GRAPH_NAME, save_path=folder)
                try:
                    for offset, amount in ((0, 1), (100, 7)):
                        source, delta = annotate.input_tensors(
                            "policy",
                            {
                                "source": torch.arange(32.0, device=self.DEVICE)
                                + offset,
                                "delta": torch.tensor(
                                    [float(amount)], device=self.DEVICE
                                ),
                            },
                        )
                        root = source * delta
                        first, second = root[1:12][::2], root[17:28:2]
                        first += delta
                        a, b = wp.from_torch(first), wp.from_torch(second)
                        wp.launch(
                            self.kernels.mutate_both_in_place,
                            dim=a.size,
                            inputs=[a, b],
                            device=self.DEVICE,
                        )
                        annotate.output_tensors(
                            "policy",
                            {"root": root * 2, "view": first},
                            export_with=backend,
                        )
                finally:
                    leapp.stop()
                leapp.compile_graph(visualize=False, validate=True, strict=True)
                manager = leapp.InferenceManager(
                    str(
                        Path(folder)
                        / self.TEST_GRAPH_NAME
                        / f"{self.TEST_GRAPH_NAME}.yaml"
                    )
                )
                for offset, amount in ((100, 1), (0, 7), (-20, -3)):
                    source = torch.arange(32.0, device=self.DEVICE) + offset
                    delta = torch.tensor([float(amount)], device=self.DEVICE)
                    expected = source * delta
                    expected[1:12:2] += amount + 1
                    expected[17:28:2] *= 2
                    actual = manager.run_policy(
                        {"policy/source": source, "policy/delta": delta}
                    )
                    torch.testing.assert_close(
                        actual["policy/root"].to(self.DEVICE), expected * 2
                    )
                    torch.testing.assert_close(
                        actual["policy/view"].to(self.DEVICE), expected[1:12:2]
                    )
                    torch.testing.assert_close(
                        source, torch.arange(32.0, device=self.DEVICE) + offset
                    )
                    torch.testing.assert_close(
                        delta, torch.tensor([float(amount)], device=self.DEVICE)
                    )

    def test_new_warp_output_alias_replays_after_segment_close(self):
        """A new output converted before publication retains the runner result."""
        for backend in ("onnx", "pt2"):
            for placement in ("inside", "after", "automatic"):
                with self.subTest(backend=backend, placement=placement):
                    with tempfile.TemporaryDirectory() as folder:
                        leapp.start(name="publication", save_path=folder)
                        try:
                            for offset in (0.0, 5.0):
                                x = annotate.input_tensors(
                                    "policy",
                                    {
                                        "x": torch.arange(3.0, device=self.DEVICE)
                                        + offset
                                    },
                                )
                                source = wp.from_torch(x)
                                region = (
                                    annotate.warp_op("policy")
                                    if placement != "automatic"
                                    else nullcontext()
                                )
                                with region:
                                    output = wp.empty_like(source)
                                    wp.launch(
                                        self.kernels.add_scalar,
                                        dim=3,
                                        inputs=[source, wp.float32(1.0)],
                                        outputs=[output],
                                        device=self.DEVICE,
                                    )
                                    if placement == "inside":
                                        alias = wp.to_torch(output)
                                if placement != "inside":
                                    alias = wp.to_torch(output)
                                annotate.output_tensors(
                                    "policy", {"y": alias * 2}, export_with=backend
                                )
                        finally:
                            leapp.stop()
                        node = annotate.get_nodes()["policy"]
                        # A new Warp result is already allocated by the runner.
                        # Publishing it must not allocate/copy another tensor.
                        self.assertFalse(
                            any(
                                item.target is torch.empty
                                or "aten.copy" in str(item.target)
                                for item in node.m.graph.nodes
                            )
                        )
                        leapp.compile_graph(visualize=False, validate=True, strict=True)
                        if backend == "pt2":
                            program = torch.export.load(
                                Path(folder) / "publication/policy.pt2"
                            )
                            self.assertFalse(
                                any(
                                    "aten.empty" in str(item.target)
                                    or "aten.copy" in str(item.target)
                                    for item in program.graph_module.graph.nodes
                                )
                            )
                        manager = leapp.InferenceManager(
                            str(Path(folder) / "publication/publication.yaml")
                        )
                        for values in ([10.0, 20.0, 30.0], [-2.0, 0.0, 4.0]):
                            fresh = torch.tensor(values, device=self.DEVICE)
                            given = fresh.clone()
                            actual = manager.run_policy({"policy/x": given})["policy/y"]
                            torch.testing.assert_close(
                                actual.cpu(), (fresh.cpu() + 1) * 2
                            )
                            torch.testing.assert_close(given, fresh)

    def test_launch_mutating_two_buffers_keeps_them_separate(self):
        """One launch writing two arrays gives each its own segment output.

        The complement of the tests above, and the way sharing fails most
        quietly. Both buffers reach the same runner, so merging their roots still
        executes and still returns plausible numbers for whichever buffer won the
        last write, while the other silently reports the wrong output index.
        """
        a_value = torch.tensor([1.0, 2.0, 3.0], device=self.DEVICE)
        b_value = torch.tensor([4.0, 5.0, 6.0], device=self.DEVICE)
        expected_a = (a_value + 1.0) + 10.0
        expected_b = (b_value * 2.0) * 3.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        for _ in range(2):
            tensor_a, tensor_b = annotate.input_tensors(
                "node_a", {"in_a": a_value.clone(), "in_b": b_value.clone()}
            )
            array_1 = wp.from_torch(tensor_a)
            array_2 = wp.from_torch(tensor_b)
            with annotate.warp_op("node_a", device=self.DEVICE):
                wp.launch(
                    self.kernels.mutate_both_in_place,
                    dim=array_1.size,
                    inputs=[array_1, array_2],
                    device=self.DEVICE,
                )

            self.assertIsNot(
                tensor_a.proxy.node,
                tensor_b.proxy.node,
                "both buffers were left on one segment output",
            )

            out_a = tensor_a + 10.0
            out_b = tensor_b * 3.0
            self.assertTrue(
                torch.equal(out_a.tensor, expected_a)
                and torch.equal(out_b.tensor, expected_b),
                f"eager values diverged: got {out_a.tensor} and {out_b.tensor}",
            )
            annotate.output_tensors(
                "node_a", {"out_a": out_a, "out_b": out_b}, export_with="onnx"
            )

        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_inference_manager(
            source_inputs={"node_a/in_a": a_value, "node_a/in_b": b_value},
            source_outputs={
                "node_a/out_a": expected_a,
                "node_a/out_b": expected_b,
            },
        )

    def test_warp_torch_warp_mutations_chain_on_one_buffer(self):
        """Replay ordered cross-framework writes and retained aliases via ONNX/YAML.

        Unlike conversion unit tests, this crosses two real Warp segments,
        creates a Torch alias before segment close, and reloads the pipeline.
        """
        leapp.start(name=self.TEST_GRAPH_NAME, max_cached_io=2)
        for offset in (0.0, 5.0):
            tensor = annotate.input_tensors(
                "node_a",
                {"in_a": torch.tensor([2.0, 4.0, 6.0], device=self.DEVICE) + offset},
            )
            array = wp.from_torch(tensor)
            with annotate.warp_op("node_a", device=self.DEVICE):
                second_handle = array[:]
                wp.launch(
                    self.kernels.add_scalar,
                    dim=array.size,
                    inputs=[array, wp.float32(1.0)],
                    outputs=[second_handle],
                    device=self.DEVICE,
                )
                alias = wp.to_torch(second_handle)
            alias.mul_(3.0)
            with annotate.warp_op("node_a", device=self.DEVICE):
                wp.launch(
                    self.kernels.divide_in_place,
                    dim=array.size,
                    inputs=[array, wp.float32(2.0)],
                    device=self.DEVICE,
                )
            annotate.output_tensors(
                "node_a", {"base": tensor, "alias": alias}, export_with="onnx"
            )
        leapp.stop()
        leapp.compile_graph(visualize=False, validate=True, strict=True)
        manager = leapp.InferenceManager(
            str(Path(self.TEST_GRAPH_NAME) / f"{self.TEST_GRAPH_NAME}.yaml")
        )
        for values in ([10.0, 20.0, 30.0], [-2.0, 0.0, 4.0]):
            fresh = torch.tensor(values, device=self.DEVICE)
            before = fresh.clone()
            expected = ((fresh + 1) * 3) / 2
            actual = manager.run_policy({"node_a/in_a": fresh})
            torch.testing.assert_close(actual["node_a/base"].cpu(), expected.cpu())
            torch.testing.assert_close(actual["node_a/alias"].cpu(), expected.cpu())
            torch.testing.assert_close(fresh, before)

    def test_host_readback_is_a_separate_value(self):
        """A device-to-host conversion copies, so it starts its own root.

        Sharing here would tie a host array to a device buffer that a later
        segment can overwrite, reporting a value the host copy never held.
        """
        leapp.start(name=self.TEST_GRAPH_NAME)
        array = annotate.input_tensors(
            "node_a",
            {"in_a": wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)},
        )

        host = array.numpy()
        self.assertIsNot(
            array.proxy,
            host.proxy,
            "a host readback shared a root with its device buffer",
        )

        returned = wp.from_numpy(host, device=self.DEVICE)
        self.assertIsNot(
            host.proxy,
            returned.proxy,
            "a host-to-device copy shared a root with its host source",
        )


class TestPersistentAliasAttachment(WarpTestCase, LEAPPFunctionalTestBase):
    """Provenance for an alias created before any tracing session existed.

    An adapter allocates a Torch buffer in ``__init__`` and keeps a Warp view of
    it for the life of the process. LEAPP never saw that conversion, so there is
    no call to intercept and nothing to share at the time it happened; both
    objects were untraced. Declaring one side through ``input_tensors`` is what
    lets the other be recognized by the bytes it covers.
    """

    def _make_persistent_pair(self, values):
        """Allocate a Torch buffer and a Warp view of it outside any session."""
        torch_buffer = values.clone()
        return torch_buffer, wp.from_torch(torch_buffer)

    def test_declared_persistent_alias_receives_the_segment_output(self):
        """The fabrics-sim shape: a kernel writes a buffer declared as an input.

        Nothing in the traced program converts between Torch and Warp, so the
        only thing connecting the launch to the declared value is that both
        aliases were declared and therefore share one root. Without that the
        runner has no consumers, pruning deletes it, and the export returns the
        input unchanged while eager Warp returns the incremented buffer.

        Declaring the Warp side is what promotes the caller's own array, so the
        loop is also the two-pass invariant: the same object is declared again on
        the second pass, and it has to land on that pass's root rather than stay
        on the graph the first pass discarded.
        """
        source_value = torch.tensor([1.0, 2.0, 3.0], device=self.DEVICE)
        expected = (source_value + 1.0) * 2.0

        torch_buffer, warp_buffer = self._make_persistent_pair(source_value)

        leapp.start(name=self.TEST_GRAPH_NAME)
        for _ in range(2):
            # The kernel writes the buffer in place, so each pass starts over.
            torch_buffer.copy_(source_value)
            obs, declared_array = annotate.input_tensors(
                "node_a", {"obs": torch_buffer, "warp_view": warp_buffer}
            )
            self.assertIs(
                declared_array,
                warp_buffer,
                "declaring a raw wp.array did not promote the caller's object",
            )
            self.assertIs(
                obs.proxy,
                warp_buffer.proxy,
                "the declared aliases did not share one root",
            )

            with annotate.warp_op("node_a", device=self.DEVICE):
                wp.launch(
                    self.kernels.increment_in_place,
                    dim=warp_buffer.size,
                    inputs=[warp_buffer],
                    device=self.DEVICE,
                )
            action = obs * 2.0

            self.assertTrue(
                torch.equal(action.tensor, expected),
                f"eager value diverged: got {action.tensor}, expected {expected}",
            )
            annotate.output_tensors("node_a", {"action": action}, export_with="onnx")

        node = annotate.get_nodes()["node_a"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertTrue(
            any(
                "warp_runner" in str(graph_node.target)
                for graph_node in node.m.graph.nodes
            ),
            "the Warp runner did not survive pruning",
        )
        self.assertEqual(
            node.trimmed_inputs,
            set(),
            "a declared alias produced a port nothing consumed",
        )
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=0,
        )
        self.verify_inference_manager(
            source_inputs={"node_a/obs": source_value},
            source_outputs={"node_a/action": expected},
        )

    def test_declaring_the_warp_side_first_reaches_the_same_root(self):
        """Declaration order does not decide which alias owns the root.

        The Warp array named first has no earlier declaration to adopt, so it
        founds the root itself and the Torch buffer joins it. That reverses the
        roles of the previous test, and the second pass is what the order used
        to change: nothing else covered those bytes yet, so re-declaring the
        promoted array is the only thing that can move it off the discarded
        graph. Leaving it behind sends the launch into an orphan the export
        never reads, which prunes the runner while eager still increments.
        """
        source_value = torch.tensor([1.0, 2.0, 3.0], device=self.DEVICE)
        expected = (source_value + 1.0) * 2.0

        torch_buffer, warp_buffer = self._make_persistent_pair(source_value)

        leapp.start(name=self.TEST_GRAPH_NAME)
        for _ in range(2):
            torch_buffer.copy_(source_value)
            declared_array, obs = annotate.input_tensors(
                "node_a", {"warp_view": warp_buffer, "obs": torch_buffer}
            )
            self.assertIs(
                declared_array,
                warp_buffer,
                "declaring a raw wp.array did not promote the caller's object",
            )
            self.assertIs(
                obs.proxy,
                warp_buffer.proxy,
                "the declared aliases did not share one root",
            )

            with annotate.warp_op("node_a", device=self.DEVICE):
                wp.launch(
                    self.kernels.increment_in_place,
                    dim=warp_buffer.size,
                    inputs=[warp_buffer],
                    device=self.DEVICE,
                )
            action = obs * 2.0

            self.assertTrue(
                torch.equal(action.tensor, expected),
                f"eager value diverged: got {action.tensor}, expected {expected}",
            )
            annotate.output_tensors("node_a", {"action": action}, export_with="onnx")

        node = annotate.get_nodes()["node_a"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertTrue(
            any(
                "warp_runner" in str(graph_node.target)
                for graph_node in node.m.graph.nodes
            ),
            "the Warp runner did not survive pruning",
        )
        self.assertEqual(
            node.trimmed_inputs,
            set(),
            "a declared alias produced a port nothing consumed",
        )
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=0,
        )
        self.verify_inference_manager(
            source_inputs={"node_a/warp_view": source_value},
            source_outputs={"node_a/action": expected},
        )

    def test_declaring_the_torch_alias_promotes_the_object_the_caller_holds(self):
        """Library code keeps reading its own attribute and still traces.

        An adapter that owns the pair reads the Torch side directly, with no
        per-step conversion for the tracer to intercept, so the return value of
        ``input_tensors`` never reaches the code doing the reading. Declaring
        promotes that object in place, which is what lets code that never saw
        LEAPP observe the runner output instead of a pre-segment value.
        """
        source_value = torch.tensor([1.0, 2.0, 3.0], device=self.DEVICE)
        expected = (source_value + 1.0) * 2.0

        torch_buffer, warp_buffer = self._make_persistent_pair(source_value)

        leapp.start(name=self.TEST_GRAPH_NAME)
        for _ in range(2):
            torch_buffer.copy_(source_value)
            annotate.input_tensors(
                "node_a", {"obs": torch_buffer, "warp_view": warp_buffer}
            )
            self.assertTrue(
                torch_buffer.is_tracing,
                "declaring a Torch input did not promote the caller's object",
            )

            with annotate.warp_op("node_a", device=self.DEVICE):
                wp.launch(
                    self.kernels.increment_in_place,
                    dim=warp_buffer.size,
                    inputs=[warp_buffer],
                    device=self.DEVICE,
                )
            # Deliberately ignores what input_tensors returned.
            action = torch_buffer * 2.0

            self.assertTrue(
                torch.equal(action.tensor, expected),
                f"eager value diverged: got {action.tensor}, expected {expected}",
            )
            annotate.output_tensors("node_a", {"action": action}, export_with="onnx")

        node = annotate.get_nodes()["node_a"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertTrue(
            any(
                "warp_runner" in str(graph_node.target)
                for graph_node in node.m.graph.nodes
            ),
            "the Warp runner did not survive pruning",
        )
        self.verify_inference_manager(
            source_inputs={"node_a/obs": source_value},
            source_outputs={"node_a/action": expected},
        )

    def test_undeclared_persistent_alias_attaches_to_nothing(self):
        """A buffer the user never declared is left alone, by design.

        Attachment is a claim the user makes by declaring the buffer. Sharing
        memory with a declared input is not itself a claim: a buffer a kernel
        reads is meant to go through ``input_tensors``, so nothing searches for
        an owner on its behalf and this program is a user error rather than a
        case LEAPP recovers.
        """
        obs_value = torch.tensor([1.0, 2.0, 3.0], device=self.DEVICE)
        _, warp_buffer = self._make_persistent_pair(obs_value)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("node_a", {"obs": obs_value.clone()})
        with annotate.warp_op("node_a", device=self.DEVICE):
            wp.launch(
                self.kernels.increment_in_place,
                dim=warp_buffer.size,
                inputs=[warp_buffer],
                device=self.DEVICE,
            )

        self.assertIs(
            type(warp_buffer), wp.array, "an undeclared persistent alias was promoted"
        )
        self.assertIsNone(
            getattr(warp_buffer, "_proxy", None),
            "an undeclared persistent alias was given provenance",
        )
        self.assertIsNotNone(obs.proxy, "the declared input lost its provenance")

    def test_closed_node_output_crosses_the_boundary_untouched(self):
        """An ordinary connection matches addresses too, and must not fault.

        Passing a tensor to the next node does not copy it, so every existing
        graph has one allocation reaching two nodes. What separates that from an
        overlap is that the producer has finished, and the consumer still gets a
        root of its own so a mutation it records cannot rewrite the producer's.
        """
        obs_value = torch.tensor([1.0, 2.0, 3.0])

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("node_a", {"obs": obs_value.clone()})
        hidden = obs * 2.0
        annotate.output_tensors("node_a", {"hidden": hidden}, export_with="jit")

        consumed = annotate.input_tensors("node_b", {"hidden": hidden})

        self.assertIsNot(
            consumed.proxy,
            hidden.proxy,
            "a boundary rewrap shared the producer's root",
        )


class TestNumpySharedMemory(LEAPPFunctionalTestBase):
    """Provenance when a Torch carrier and a NumPy array describe one buffer.

    A CPU tensor's ``.numpy()`` hands back the tensor's own bytes, so the two
    carriers name one value. NumPy views that cover different bytes -- a slice,
    a freshly allocated ufunc result -- must keep independent roots, because
    sharing there would report a mutation eager NumPy never applied.
    """

    def test_view_mutation_feedback(self):
        """A NumPy view write updates the state restored on the next policy call."""
        leapp.start(name=self.TEST_GRAPH_NAME, max_cached_io=2)
        try:
            for amount in (1.0, 3.0):
                observation = annotate.input_tensors(
                    "policy", {"obs": np.array([amount], dtype=np.float32)}
                )
                state = annotate.state_tensors(
                    "policy", {"history": np.zeros(12, dtype=np.float32)}
                )
                view = state.reshape(3, 4)[:, 1:3]
                view += observation
                annotate.update_state("policy", {"history": state})
                annotate.output_tensors(
                    "policy", {"action": state * 2}, export_with="pt2"
                )
        finally:
            leapp.stop()
        leapp.compile_graph(visualize=False, validate=True, strict=True)
        manager = leapp.InferenceManager(
            f"{self.TEST_GRAPH_NAME}/{self.TEST_GRAPH_NAME}.yaml"
        )
        expected = torch.zeros(12)
        for amount in (2.0, -1.0, 4.0):
            expected.reshape(3, 4)[:, 1:3] += amount
            device = manager.nodes["policy"].device
            result = manager.run_policy(
                {"policy/obs": torch.tensor([amount], device=device)}
            )
            torch.testing.assert_close(result["policy/action"].cpu(), expected * 2)

    def test_numpy_and_torch_mutations_replay_through_shared_storage(self):
        """Replay writes from both frameworks through YAML and both output ports.

        Unit tests check individual conversions; this checks their composition
        with public annotation, cached validation, and InferenceManager reload.
        """
        leapp.start(name=self.TEST_GRAPH_NAME, max_cached_io=2)
        for offset in (0.0, 5.0):
            obs = annotate.input_tensors(
                "policy", {"obs": torch.tensor([-2.0, 1.0, 3.0]) + offset}
            )
            np_view = obs.numpy()
            np_view += 1.0
            torch.relu_(obs)
            annotate.output_tensors(
                "policy",
                {"torch": obs * 2, "numpy": np_view * 3},
                export_with="jit",
            )
        leapp.stop()
        leapp.compile_graph(visualize=False, validate=True, strict=True)
        manager = leapp.InferenceManager(
            str(Path(self.TEST_GRAPH_NAME) / f"{self.TEST_GRAPH_NAME}.yaml")
        )
        for values in ([-4.0, 5.0, -6.0], [2.0, -0.5, 7.0]):
            fresh = torch.tensor(values)
            before = fresh.clone()
            expected = torch.relu(fresh + 1)
            actual = manager.run_policy({"policy/obs": fresh})
            torch.testing.assert_close(actual["policy/torch"], expected * 2)
            torch.testing.assert_close(actual["policy/numpy"], expected * 3)
            torch.testing.assert_close(fresh, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
