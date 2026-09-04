#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import dataclasses
import tempfile
import types
import unittest

import numpy as np

import leapp
from leapp.export_manager import ExportManager
from leapp.leapp_graph.datatypes._attribute_patching import (
    AttributePatchRegistry,
)
from leapp.leapp_graph.datatypes.patching import FunctionPatch, TracingPatcher
from leapp.leapp_graph.traced_node import TracedTensorNode


def _original(*args, **kwargs):
    return "original"


def _replacement(*args, **kwargs):
    return "replacement"


def _wrapper(*args, **kwargs):
    return "wrapper"


class TestUserFunctionPatching(unittest.TestCase):
    def _module(self):
        module = types.ModuleType("patch_target")
        module.target = _original
        return module

    def _active_input(self):
        node = TracedTensorNode(name="patch_test", node_index=0)
        return node, node.create_input(np.array([1.0, 2.0]), name="x")

    def test_patcher_def_is_public_and_frozen(self):
        self.assertIs(leapp.FunctionPatch, FunctionPatch)
        definition = leapp.FunctionPatch(self._module(), "target", _replacement)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            definition.function_name = "other"

    def test_dispatches_on_active_traced_arguments(self):
        module = self._module()
        original = module.target
        patcher = TracingPatcher()
        patcher.install(
            patching=[FunctionPatch(module, "target", _replacement)],
        )
        try:
            node, traced = self._active_input()
            self.assertEqual(module.target(traced), "replacement")
            self.assertEqual(module.target([{"value": traced}]), "replacement")
            self.assertEqual(module.target(value={"nested": traced}), "replacement")
            self.assertEqual(module.target(np.array([1.0])), "original")

            node.compile_trace({"output": traced})
            self.assertEqual(
                module.target(traced),
                "original",
            )
        finally:
            patcher.uninstall()

        self.assertIs(module.target, original)

    def test_global_patching_false_skips_user_patch(self):
        module = self._module()
        original = module.target
        with tempfile.TemporaryDirectory() as save_path:
            leapp.start(
                "user_patch",
                save_path=save_path,
                global_patching=False,
                patching=[FunctionPatch(module, "target", _replacement)],
            )
            _, traced = self._active_input()
            self.assertEqual(module.target(traced), "original")
            leapp.stop()

        self.assertIs(module.target, original)

    def test_stop_start_cycle_accepts_a_different_replacement(self):
        module = self._module()

        def replacement_one(*args, **kwargs):
            return "one"

        def replacement_two(*args, **kwargs):
            return "two"

        with tempfile.TemporaryDirectory() as save_path:
            for replacement, expected in (
                (replacement_one, "one"),
                (replacement_two, "two"),
            ):
                leapp.start(
                    "user_patch",
                    save_path=save_path,
                    patching=[FunctionPatch(module, "target", replacement)],
                )
                _, traced = self._active_input()
                self.assertEqual(module.target(traced), expected)
                leapp.stop()

    def test_invalid_definitions_fail_before_mutation(self):
        module = self._module()
        original = module.target
        valid = FunctionPatch(module, "target", _replacement)
        invalid_definitions = [
            [valid, object()],
            [valid, FunctionPatch(object(), "target", _replacement)],
            [valid, FunctionPatch(module, "", _replacement)],
            [valid, FunctionPatch(module, "missing", _replacement)],
            [valid, FunctionPatch(module, "target", None)],
            [valid, valid],
        ]

        module.not_callable = 1
        invalid_definitions.append(
            [valid, FunctionPatch(module, "not_callable", _replacement)]
        )

        for definitions in invalid_definitions:
            with self.subTest(definitions=definitions):
                patcher = TracingPatcher()
                with self.assertRaises((TypeError, ValueError)):
                    patcher.install(patching=definitions)
                self.assertIs(module.target, original)
                self.assertFalse(patcher.installed)

    def test_partial_installation_failure_rolls_back_all_backends(self):
        class FailingModule(types.ModuleType):
            def __setattr__(self, name, value):
                if name == "second" and getattr(self, "_fail", False):
                    types.ModuleType.__setattr__(self, "_fail", False)
                    types.ModuleType.__setattr__(self, name, value)
                    raise RuntimeError("installation failed")
                types.ModuleType.__setattr__(self, name, value)

        module = FailingModule("failing_patch_target")
        module.first = _original
        module.second = _original
        module._fail = True
        original_array = np.array
        patcher = TracingPatcher()

        with self.assertRaisesRegex(RuntimeError, "installation failed"):
            patcher.install(
                patching=[
                    FunctionPatch(module, "first", _replacement),
                    FunctionPatch(module, "second", _replacement),
                ]
            )

        self.assertIs(module.first, _original)
        self.assertIs(module.second, _original)
        self.assertIs(np.array, original_array)
        self.assertFalse(patcher.installed)

    def test_uninstall_preserves_external_attribute_change(self):
        module = self._module()
        patcher = TracingPatcher()
        patcher.install(
            patching=[FunctionPatch(module, "target", _replacement)],
        )

        def external(*args, **kwargs):
            return "external"

        module.target = external
        patcher.uninstall()
        self.assertIs(module.target, external)

    def test_external_change_survives_default_backend_uninstall(self):
        original = np.array
        patcher = TracingPatcher()
        patcher.install(
            patching=[FunctionPatch(np, "array", _replacement)],
        )

        def external(*args, **kwargs):
            return "external"

        try:
            np.array = external
            patcher.uninstall()
            self.assertIs(np.array, external)
        finally:
            np.array = original

    def tearDown(self):
        if ExportManager.is_interpret_graph_enabled():
            leapp.stop()


class TestAttributePatchRegistry(unittest.TestCase):
    def test_restores_installed_wrapper(self):
        module = types.ModuleType("patch_target")
        module.target = _original
        registry = AttributePatchRegistry()

        registry.install(module, "target", _original, _wrapper)
        self.assertIs(module.target, _wrapper)

        registry.restore()
        self.assertIs(module.target, _original)
        self.assertEqual(len(registry), 0)

    def test_does_not_overwrite_external_change(self):
        module = types.ModuleType("patch_target")
        module.target = _original
        registry = AttributePatchRegistry()
        registry.install(module, "target", _original, _wrapper)

        def external():
            return "external"

        module.target = external
        registry.restore()

        self.assertIs(module.target, external)

    def test_failed_install_restores_mutated_attribute(self):
        class FailingModule(types.ModuleType):
            def __setattr__(self, name, value):
                if name == "target" and getattr(self, "_fail", False):
                    types.ModuleType.__setattr__(self, "_fail", False)
                    types.ModuleType.__setattr__(self, name, value)
                    raise RuntimeError("installation failed")
                types.ModuleType.__setattr__(self, name, value)

        module = FailingModule("patch_target")
        module.target = _original
        module._fail = True
        registry = AttributePatchRegistry()

        with self.assertRaisesRegex(RuntimeError, "installation failed"):
            registry.install(module, "target", _original, _wrapper)

        self.assertIs(module.target, _original)
        self.assertEqual(len(registry), 0)

    def test_restore_can_suppress_owner_errors(self):
        class FailingModule(types.ModuleType):
            def __setattr__(self, name, value):
                if name == "target" and getattr(self, "_fail", False):
                    raise RuntimeError("restoration failed")
                types.ModuleType.__setattr__(self, name, value)

        module = FailingModule("patch_target")
        module.target = _original
        registry = AttributePatchRegistry()
        registry.install(module, "target", _original, _wrapper)
        module._fail = True

        registry.restore(suppress_errors=True)

        self.assertIs(module.target, _wrapper)
        self.assertEqual(len(registry), 0)


if __name__ == "__main__":
    unittest.main()
