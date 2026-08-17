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

    def _verify_alias_mutation_propagates(self, make_alias):
        """Mutate a buffer through an alias and read it back through the source.

        ``make_alias`` takes the traced input and returns a second carrier over
        the same buffer. The mutation goes through that alias while the node's
        output is read through the original, so the exported graph matches eager
        Torch only if the two carriers share provenance.
        """
        obs_value = torch.tensor([1.0, 2.0, 3.0])
        expected = (obs_value + 1.0) * 2.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": obs_value.clone()})

        alias = make_alias(obs)
        alias += 1.0
        action = obs * 2.0

        # Eager Torch wrote through shared memory, so this is the value the
        # exported graph has to reproduce.
        self.assertTrue(
            torch.equal(action.tensor, expected),
            f"eager value diverged: got {action.tensor}, expected {expected}")

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1, internal_connections=0,
            feedback_connections=0)
        self.verify_inference_manager(
            source_inputs={"policy/obs": obs_value},
            source_outputs={"policy/action": expected},
        )

    def test_detach_alias_shares_provenance(self):
        """``detach`` returns this buffer at this layout."""
        self._verify_alias_mutation_propagates(lambda tensor: tensor.detach())

    def test_full_slice_alias_shares_provenance(self):
        """A ``[:]`` key selects everything, so it narrows nothing."""
        self._verify_alias_mutation_propagates(lambda tensor: tensor[:])

    def test_noop_conversion_alias_shares_provenance(self):
        """A ``to`` that neither casts nor moves allocates nothing."""
        self._verify_alias_mutation_propagates(
            lambda tensor: tensor.to(torch.float32))

    def test_chained_aliases_share_one_provenance(self):
        """An alias of an alias still names the one original value."""
        self._verify_alias_mutation_propagates(
            lambda tensor: tensor.detach()[:].to(torch.float32))

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
        targets = [str(node.target)
                   for node in annotate.nodes["policy"].graph.nodes]

        self.assertTrue(
            torch.equal(action.tensor, expected),
            f"eager value diverged: got {action.tensor}, expected {expected}")

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # The mutation is a real operation, so unlike an alias it is recorded.
        self.assertTrue(
            any("relu" in target for target in targets),
            f"the in-place mutation was not recorded: {targets}")
        self.verify_inference_manager(
            source_inputs={"policy/obs": obs_value},
            source_outputs={"policy/action": expected},
        )

    def test_clone_is_a_separate_value(self):
        """A copy shares no memory, so mutating it must not move the source.

        The complement of the tests above: sharing a root here would make the
        source inherit a mutation eager Torch never applied to it.
        """
        obs_value = torch.tensor([1.0, 2.0, 3.0])
        expected = obs_value * 2.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": obs_value.clone()})

        copy = obs.clone()
        copy += 100.0
        action = obs * 2.0

        self.assertTrue(
            torch.equal(action.tensor, expected),
            f"eager value diverged: got {action.tensor}, expected {expected}")

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_inference_manager(
            source_inputs={"policy/obs": obs_value},
            source_outputs={"policy/action": expected},
        )

    def test_aliasing_records_no_operation(self):
        """An alias is a second name for a value, not a computation.

        Recording a node for it would leave the graph carrying work that has no
        consumers, so the aliasing chain should contribute nothing between the
        placeholder and the multiply.
        """
        obs_value = torch.tensor([1.0, 2.0, 3.0])

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": obs_value.clone()})

        action = obs.detach()[:].to(torch.float32) * 2.0
        targets = [str(node.target)
                   for node in annotate.nodes["policy"].graph.nodes]

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertEqual(
            2, len(targets),
            f"expected only a placeholder and a multiply, got {targets}")
        self.assertFalse(
            any("detach" in target or "getitem" in target or target == "to"
                for target in targets),
            f"an aliasing operation was recorded: {targets}")
        self.verify_inference_manager(
            source_inputs={"policy/obs": obs_value},
            source_outputs={"policy/action": obs_value * 2.0},
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

    def test_from_torch_alias_follows_the_segment(self):
        """A segment writing a converted buffer moves its Torch carrier too.

        Nothing reads the Warp array after the launch. Only the Torch tensor is
        read, so the runner survives pruning solely because the conversion left
        the two sharing one root.
        """
        obs_value = torch.tensor([1.0, 2.0, 3.0], device=self.DEVICE)
        expected = (obs_value + 1.0) * 2.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        for _ in range(2):
            obs = annotate.input_tensors(
                "node_a", {"obs": obs_value.clone()})
            array = wp.from_torch(obs)
            with annotate.warp_op("node_a", device=self.DEVICE):
                wp.launch(
                    self.kernels.increment_in_place,
                    dim=array.size,
                    inputs=[array],
                    device=self.DEVICE,
                )
            action = obs * 2.0

            self.assertTrue(
                torch.equal(action.tensor, expected),
                f"eager value diverged: got {action.tensor}, expected {expected}")
            annotate.output_tensors(
                "node_a", {"action": action}, export_with="onnx")

        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_inference_manager(
            source_inputs={"node_a/obs": obs_value},
            source_outputs={"node_a/action": expected},
        )

    def test_to_torch_inside_an_open_segment_follows_the_close(self):
        """A conversion made before the runner exists still ends up on it.

        Inside an open segment the Warp array only carries placeholder
        provenance, because the runner output it will resolve to has not been
        created yet. Sharing a root is what lets the close move both carriers at
        once instead of rebinding the Warp array and orphaning the tensor.
        """
        source_value = torch.tensor([1.0, 2.0, 3.0], device=self.DEVICE)
        expected = (source_value + 1.0) * 2.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        for _ in range(2):
            array = annotate.input_tensors(
                "node_a",
                {"in_a": wp.from_torch(source_value.clone())},
            )
            with annotate.warp_op("node_a", device=self.DEVICE):
                wp.launch(
                    self.kernels.increment_in_place,
                    dim=array.size,
                    inputs=[array],
                    device=self.DEVICE,
                )
                tensor = wp.to_torch(array)
            action = tensor * 2.0

            self.assertTrue(
                torch.equal(action.tensor, expected),
                f"eager value diverged: got {action.tensor}, expected {expected}")
            annotate.output_tensors(
                "node_a", {"action": action}, export_with="onnx")

        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_inference_manager(
            source_inputs={"node_a/in_a": source_value},
            source_outputs={"node_a/action": expected},
        )

    def test_second_warp_handle_on_one_buffer_follows_the_segment(self):
        """A launch writing one buffer through a second Warp handle moves it too.

        ``array[:]`` selects everything, so Warp hands back another wp.array over
        the same bytes at the same layout. The launch writes the buffer through
        that second handle and the node's output is read back through it, so the
        exported graph reproduces eager Warp only if the handle ends up on a
        segment output. Folding it into the original's output leaves it on the
        pre-segment proxy, which prunes the runner and exports the input
        unchanged.
        """
        source_value = torch.tensor([1.0, 2.0, 3.0], device=self.DEVICE)
        expected = (source_value + 1.0) * 2.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        for _ in range(2):
            array = annotate.input_tensors(
                "node_a",
                {"in_a": wp.from_torch(source_value.clone())},
            )
            with annotate.warp_op("node_a", device=self.DEVICE):
                alias = array[:]
                wp.launch(
                    self.kernels.add_scalar,
                    dim=array.size,
                    inputs=[array, wp.float32(1.0)],
                    outputs=[alias],
                    device=self.DEVICE,
                )
                tensor = wp.to_torch(alias)
            action = tensor * 2.0

            self.assertTrue(
                torch.equal(action.tensor, expected),
                f"eager value diverged: got {action.tensor}, expected {expected}")
            annotate.output_tensors(
                "node_a", {"action": action}, export_with="onnx")

        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_inference_manager(
            source_inputs={"node_a/in_a": source_value},
            source_outputs={"node_a/action": expected},
        )

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
                "node_a", {"in_a": a_value.clone(), "in_b": b_value.clone()})
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
                tensor_a.proxy.node, tensor_b.proxy.node,
                "both buffers were left on one segment output")

            out_a = tensor_a + 10.0
            out_b = tensor_b * 3.0
            self.assertTrue(
                torch.equal(out_a.tensor, expected_a)
                and torch.equal(out_b.tensor, expected_b),
                f"eager values diverged: got {out_a.tensor} and {out_b.tensor}")
            annotate.output_tensors(
                "node_a", {"out_a": out_a, "out_b": out_b},
                export_with="onnx")

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
        """Three in-place writes to one allocation, alternating frameworks.

        ``test_cuda_warp_to_torch_to_warp_in_one_node`` already covers this
        ordering, but its Torch stage allocates, so the second segment works on
        a different buffer and each segment can own a root of its own. Here
        every stage writes the original bytes, so the Torch mutation has to land
        on the root the first segment left behind and the second segment has to
        pick up the root the Torch mutation left, all on one view that changes
        hands twice.

        Eager Torch reports the right answer either way, because in-place writes
        reach the buffer whatever the graph believes, so a broken chain shows up
        only after export. The three factors do not commute and none is the
        identity, so dropping or reordering any stage changes the result.
        """
        source_value = torch.tensor([2.0, 4.0, 6.0], device=self.DEVICE)
        # Warp adds one, Torch triples, Warp halves, then the output is read
        # back through the original Torch carrier.
        expected = (((source_value + 1.0) * 3.0) / 2.0) * 2.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        for _ in range(2):
            tensor = annotate.input_tensors(
                "node_a", {"in_a": source_value.clone()})
            array = wp.from_torch(tensor)

            with annotate.warp_op("node_a", device=self.DEVICE):
                wp.launch(
                    self.kernels.increment_in_place,
                    dim=array.size,
                    inputs=[array],
                    device=self.DEVICE,
                )

            # Torch writes the same bytes between the two segments, so this
            # mutation has to be visible to the Warp array as well.
            tensor.mul_(3.0)

            with annotate.warp_op("node_a", device=self.DEVICE):
                wp.launch(
                    self.kernels.divide_in_place,
                    dim=array.size,
                    inputs=[array, wp.float32(2.0)],
                    device=self.DEVICE,
                )

            action = tensor * 2.0
            self.assertTrue(
                torch.equal(action.tensor, expected),
                f"eager value diverged: got {action.tensor}, "
                f"expected {expected}")
            annotate.output_tensors(
                "node_a", {"action": action}, export_with="onnx")

        node = annotate.get_nodes()["node_a"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # The Torch mutation between the launches has to split them, otherwise
        # the two would have merged into one segment and lost the middle stage.
        self.assertEqual(
            ["warp_segment_0", "warp_segment_1"],
            [segment.runner_name for segment in node.warp_segments])
        self.verify_inference_manager(
            source_inputs={"node_a/in_a": source_value},
            source_outputs={"node_a/action": expected},
        )

    def test_host_readback_is_a_separate_value(self):
        """A device-to-host conversion copies, so it starts its own root.

        Sharing here would tie a host array to a device buffer that a later
        segment can overwrite, reporting a value the host copy never held.
        """
        leapp.start(name=self.TEST_GRAPH_NAME)
        array = annotate.input_tensors(
            "node_a",
            {"in_a": wp.array(
                [1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)},
        )

        host = array.numpy()
        self.assertIsNot(
            array.proxy_view, host.proxy_view,
            "a host readback shared a root with its device buffer")

        returned = wp.from_numpy(host, device=self.DEVICE)
        self.assertIsNot(
            host.proxy_view, returned.proxy_view,
            "a host-to-device copy shared a root with its host source")


class TestNumpySharedMemory(LEAPPFunctionalTestBase):
    """Provenance when a Torch carrier and a NumPy array describe one buffer.

    A CPU tensor's ``.numpy()`` hands back the tensor's own bytes, so the two
    carriers name one value. NumPy views that cover different bytes -- a slice,
    a freshly allocated ufunc result -- must keep independent roots, because
    sharing there would report a mutation eager NumPy never applied.
    """

    def test_cpu_numpy_conversion_shares_a_root(self):
        """``.numpy()`` on a CPU tensor is a second handle, not a copy."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.tensor([1.0, 2.0])})

        self.assertIs(
            obs.proxy_view, obs.numpy().proxy_view,
            "a zero-copy .numpy() did not share the tensor's root")

    def test_identical_layout_numpy_view_shares_a_root(self):
        """A view cast covers the same bytes at the same layout."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        arr = annotate.input_tensors(
            "policy", {"obs": np.array([1.0, 2.0, 3.0], dtype=np.float32)})

        self.assertIs(
            arr.proxy_view, arr.view(type(arr)).proxy_view,
            "an identical-layout view did not share its source's root")

    def test_narrowing_and_allocating_results_are_separate_values(self):
        """A slice covers fewer bytes and a ufunc result covers other bytes.

        Neither is the same value as its source, so each needs a root of its
        own; sharing would make the source report a value it never held.
        """
        leapp.start(name=self.TEST_GRAPH_NAME)
        arr = annotate.input_tensors(
            "policy", {"obs": np.array([1.0, 2.0, 3.0], dtype=np.float32)})

        self.assertIsNot(
            arr.proxy_view, arr[0:2].proxy_view,
            "a narrowing slice shared its source's root")
        self.assertIsNot(
            arr.proxy_view, np.add(arr, 1.0).proxy_view,
            "a freshly allocated ufunc result shared its source's root")

    def test_numpy_mutation_reaches_the_torch_carrier(self):
        """A write through the NumPy carrier moves the Torch carrier with it.

        The mutation and the read are on opposite carriers, so the exported
        graph reproduces eager NumPy only if the two share provenance. Driven to
        InferenceManager because a stale proxy still exports a graph that runs
        and returns plausible-looking numbers rather than raising.
        """
        obs_value = torch.tensor([1.0, 2.0, 3.0])
        expected = (obs_value + 1.0) * 2.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": obs_value.clone()})

        np_view = obs.numpy()
        np_view += 1.0
        action = obs * 2.0

        self.assertTrue(
            torch.equal(action.tensor, expected),
            f"eager value diverged: got {action.tensor}, expected {expected}")

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_inference_manager(
            source_inputs={"policy/obs": obs_value},
            source_outputs={"policy/action": expected},
        )

    def test_torch_mutation_reaches_the_numpy_carrier(self):
        """The reverse direction: a Torch write is visible through NumPy.

        Asserted on provenance rather than through an export, because reading
        the mutated value back out of the NumPy carrier is the same shared-cell
        property the test above already drives end to end.
        """
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors(
            "policy", {"obs": torch.tensor([-1.0, 2.0, -3.0])})

        np_view = obs.numpy()
        placeholder_proxy = obs.proxy
        torch.relu_(obs)

        self.assertIs(
            obs.proxy_view, np_view.proxy_view,
            "the Torch mutation replaced the root instead of its proxy")
        self.assertIsNot(
            placeholder_proxy, np_view.proxy,
            "the NumPy carrier still reports the pre-mutation proxy")


if __name__ == "__main__":
    unittest.main(verbosity=2)
