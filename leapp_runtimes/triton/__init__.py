#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Triton Inference Server runtime for LEAPP graphs.

``create_triton_model_repo(config_path, output_dir)`` turns a LEAPP export (the ``<graph>.yaml``
spec + per-node ``.onnx`` / ``.pt`` / ``.wrp`` artifacts) into a Triton model repository with an
ensemble that chains the nodes. ONNX/TorchScript nodes become standard backend models; a Warp
(APIC) node becomes a ``python``-backend model that replays its ``.wrp`` (templates in
``leapp_runtimes/triton/warp_node/``).

Downstream usage (e.g. isaac_ros_deploy):

    from leapp_runtimes.triton.create_triton_model_repo import create_triton_model_repo
    create_triton_model_repo(Path("graph.yaml"), Path("model_repo"))
"""
