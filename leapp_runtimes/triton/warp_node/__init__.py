#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Warp (APIC) Triton python-backend node templates + deploy-side runtime.

- ``warp_apic_runtime.py`` — importable deploy-side core (``WarpApicRunner``): loads a ``.wrp``,
  verifies its compiled modules, and replays it on torch tensors. Depends only on torch + warp.
- ``model.py`` — the Triton python-backend ``TritonPythonModel`` TEMPLATE. It is copied verbatim
  into each warp node's model-version dir by ``create_triton_model_repo`` and executed there by
  Triton's python backend; it is NOT imported as a module of this package (it imports
  ``triton_python_backend_utils``, which only exists inside Triton).
"""
