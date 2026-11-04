#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for TracedInt, the traced integer read from data-dependent sizes."""

import copy
import operator
import pickle
import unittest

import numpy as np
import torch
import yaml

from leapp.leapp_graph.datatypes import TracedData, TracedInt, TracedTensor
from leapp.leapp_graph.traced_node import TracedTensorNode


def _traced_count(name="test", values=(1.0, -1.0, 2.0)):
    """A node and the traced number of positive entries in its input."""
    ctx = TracedTensorNode(name=name, node_index=0)
    x = ctx.create_input(torch.tensor(values), name="x")
    count = torch.nonzero(x > 0).shape[0]
    return ctx, count


def _node_count(ctx):
    return len(list(ctx.graph.nodes))


class TestTracedIntType(unittest.TestCase):
    def test_is_int_and_traced_data(self):
        ctx, n = _traced_count()
        self.assertIsInstance(n, TracedInt)
        self.assertIsInstance(n, int)
        self.assertIsInstance(n, TracedData)
        self.assertEqual(n, 2)
        self.assertIs(n.context_obj, ctx)
        self.assertTrue(n.is_tracing)

    def test_data_and_tensor_views(self):
        _, n = _traced_count()
        self.assertIs(type(n.data), int)
        self.assertEqual(n.data, 2)
        self.assertEqual(n.tensor.dtype, torch.int64)
        self.assertEqual(n.tensor.item(), 2)
        self.assertEqual(n.get_dtype_name(), "int64")
        np.testing.assert_array_equal(np.asarray(n), np.asarray(2))

    def test_string_conversions_use_the_value(self):
        _, n = _traced_count()
        self.assertEqual(str(n), "2")
        self.assertEqual(f"{n:03d}", "002")
        self.assertEqual(repr(n), "TracedInt(2)")

    def test_generic_traced_data_traversal(self):
        _, n = _traced_count()
        structure = {"sizes": [1, (n,)], "slice": slice(None, n)}
        self.assertIs(TracedData.find_traced_data(structure), n)
        self.assertEqual(TracedData.find_all_contexts(structure), {"test"})

        unwrapped = TracedData.unwrap_traced_data(structure)
        self.assertIs(type(unwrapped["sizes"][1][0]), int)
        self.assertIs(type(unwrapped["slice"].stop), int)

        proxies = TracedData.extract_proxy(structure)
        self.assertIs(proxies["sizes"][1][0], n.proxy)
        self.assertIs(proxies["slice"].stop, n.proxy)


class TestTracedIntArithmetic(unittest.TestCase):
    def test_integer_arithmetic_is_recorded(self):
        ctx, n = _traced_count()
        cases = [
            (n + 1, 3, operator.add),
            (1 + n, 3, operator.add),
            (n - 1, 1, operator.sub),
            (5 - n, 3, operator.sub),
            (n * 3, 6, operator.mul),
            (3 * n, 6, operator.mul),
            (n // 2, 1, operator.floordiv),
            (7 // n, 3, operator.floordiv),
            (n % 2, 0, operator.mod),
            (7 % n, 1, operator.mod),
            (-n, -2, operator.sub),
            (n + n, 4, operator.add),
        ]
        for result, expected, op in cases:
            with self.subTest(op=op.__name__, expected=expected):
                self.assertIsInstance(result, TracedInt)
                self.assertEqual(result, expected)
                self.assertIs(result.context_obj, ctx)
                self.assertEqual(result.proxy.node.target, op)

    def test_recorded_arithmetic_replays_with_new_count(self):
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, -1.0, 2.0]), name="x")
        n = torch.nonzero(x > 0).shape[0]
        out = x[: n * 2 - 3] + 0
        ctx.compile_trace({"out": out})
        replay = ctx.m(torch.tensor([3.0, 4.0, 5.0]))
        torch.testing.assert_close(replay, torch.tensor([3.0, 4.0, 5.0]))

    def test_float_arithmetic_is_a_plain_float(self):
        ctx, n = _traced_count()
        before = _node_count(ctx)
        for result in (n + 0.5, n / 2, 0.5 * n):
            self.assertIs(type(result), float)
        self.assertEqual(_node_count(ctx), before)

    def test_comparisons_are_plain_bools(self):
        ctx, n = _traced_count()
        before = _node_count(ctx)
        self.assertIs(n == 2, True)
        self.assertIs(n != 2, False)
        self.assertIs(n > 1, True)
        self.assertIs(n <= 1, False)
        self.assertEqual(hash(n), hash(2))
        self.assertEqual(_node_count(ctx), before)


class TestTracedIntSerialization(unittest.TestCase):
    def test_copy_and_deepcopy_keep_the_graph_value(self):
        _, n = _traced_count()
        self.assertIs(copy.copy(n), n)
        self.assertIs(copy.deepcopy(n), n)
        nested = copy.deepcopy({"shape": [n, 3]})
        self.assertIs(nested["shape"][0], n)

    def test_pickle_stores_the_plain_value(self):
        _, n = _traced_count()
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(n, protocol=protocol))
                self.assertIs(type(restored), int)
                self.assertEqual(restored, 2)

    def test_yaml_writes_a_plain_integer(self):
        _, n = _traced_count()
        self.assertEqual(yaml.safe_dump({"count": n}), "count: 2\n")
        self.assertEqual(yaml.dump({"count": n}), "count: 2\n")
        self.assertEqual(yaml.safe_load(yaml.safe_dump([n])), [2])


class TestTracedIntOutsideActiveTracing(unittest.TestCase):
    def test_arithmetic_after_capture_is_plain(self):
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, -1.0, 2.0]), name="x")
        n = torch.nonzero(x > 0).shape[0]
        ctx.compile_trace({"y": x * 2})
        self.assertFalse(ctx.is_tracing)
        before = _node_count(ctx)
        for result in (n + 1, 1 + n, n * 2, -n, n + n):
            self.assertIs(type(result), int)
        self.assertEqual(n + 1, 3)
        self.assertEqual(_node_count(ctx), before)

    def test_arithmetic_after_graph_replacement_is_plain(self):
        ctx, n = _traced_count()
        ctx.reset_trace_state()
        self.assertTrue(n.describes_replaced_graph())
        result = n + 1
        self.assertIs(type(result), int)
        self.assertEqual(result, 3)
        self.assertEqual(_node_count(ctx), 0)

    def test_stale_operand_is_used_as_a_constant(self):
        _, n = _traced_count()
        _, finished = _traced_count(name="finished", values=(1.0, 1.0, 1.0))
        finished.context_obj._model_captured = True
        result = n + finished
        self.assertIsInstance(result, TracedInt)
        self.assertEqual(result, 5)
        self.assertEqual(result.proxy.node.args, (n.proxy.node, 3))
        self.assertIs(type(result.proxy.node.args[1]), int)

    def test_operands_from_other_active_nodes_are_rejected(self):
        _, n = _traced_count(name="a")
        _, other = _traced_count(name="b")
        with self.assertRaisesRegex(Exception, "multiple"):
            n + other


class TestTracedIntInTensorOps(unittest.TestCase):
    def test_narrow_bound_uses_the_integer_proxy(self):
        ctx, n = _traced_count()
        destination = ctx.create_input(torch.zeros(5), name="d")
        self.assertIs(destination._narrow_bound(n, 0, 5), n.proxy)

    def test_slice_assignment_with_traced_bound_replays(self):
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, -1.0, 2.0]), name="x")
        destination = ctx.create_input(torch.zeros(5), name="d")
        n = torch.nonzero(x > 0).shape[0]
        out = destination.clone()
        out[:n] = 1.0
        torch.testing.assert_close(out.tensor, torch.tensor([1.0, 1.0, 0, 0, 0]))
        ctx.compile_trace({"out": out})
        replay = ctx.m(torch.tensor([1.0, 2.0, 3.0]), torch.zeros(5))
        torch.testing.assert_close(replay, torch.tensor([1.0, 1.0, 1.0, 0, 0]))

    def test_traced_int_as_assignment_value_replays(self):
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, -1.0, 2.0]), name="x")
        n = torch.nonzero(x > 0).shape[0]
        out = x * 0
        out[0] = n
        torch.testing.assert_close(out.tensor, torch.tensor([2.0, 0, 0]))
        ctx.compile_trace({"out": out})
        replay = ctx.m(torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(replay, torch.tensor([3.0, 0, 0]))

    def test_size_queries_on_finished_node_are_plain(self):
        ctx = TracedTensorNode(name="test", node_index=0)
        x = ctx.create_input(torch.tensor([1.0, -1.0, 2.0]), name="x")
        idx = torch.nonzero(x > 0)
        self.assertIsInstance(idx, TracedTensor)
        ctx._model_captured = True
        self.assertIs(type(idx.shape[0]), int)
        self.assertIs(type(idx.numel()), int)


if __name__ == "__main__":
    unittest.main()
