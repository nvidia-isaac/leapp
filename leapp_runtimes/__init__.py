#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""LEAPP deployment runtimes.

Deploy-side code for *running* an exported LEAPP graph (as opposed to the ``leapp`` package, which
*traces and exports* it). Owned by LEAPP so downstream deployers (e.g. ``isaac_ros_deploy``) can
depend on it directly instead of reimplementing it.

Subpackages:
- ``leapp_runtimes.triton`` — convert a LEAPP graph (YAML + per-node artifacts) into a Triton
  Inference Server model repository + ensemble, including the Warp (APIC) python-backend node.
"""
