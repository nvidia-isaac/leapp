# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for compiling native plugins with their own compiler and STL ABI flags."""

load("@rules_cc//cc:defs.bzl", "cc_binary")

def native_backend_plugin(name, srcs, deps = [], copts = [], **kwargs):
    """Build a shared backend, compiling the adapter with the plugin's flags.

    Args:
        name: Shared-library target name (for example, libleapp_example.so).
        srcs: Plugin implementation sources.
        deps: Additional plugin dependencies, including its model SDK.
        copts: Plugin compiler flags, including any STL ABI definition.
        **kwargs: Additional cc_binary attributes except linkshared.
    """
    cc_binary(
        name = name,
        srcs = srcs + [Label("//:native_backend_adapter")],
        deps = deps + [Label("//:native_backend")],
        copts = ["-std=c++20"] + copts,
        linkshared = True,
        **kwargs
    )
