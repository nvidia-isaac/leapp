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
import torch
import leapp
from leapp.leapp import _MANAGER as annotate
from .base import LEAPPFunctionalTestBase


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
