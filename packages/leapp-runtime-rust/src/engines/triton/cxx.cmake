# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set(TRITON_ROOT "$ENV{TRITON_ROOT}" CACHE PATH "Triton installation for integration tests")
set(LEAPP_TEST_BACKENDS "$ENV{LEAPP_TRITON_BACKENDS}" CACHE PATH "Test backend directory")
if(TRITON_ROOT)
  list(APPEND LEAPP_ENGINE_TEST_ENV "TRITON_ROOT=${TRITON_ROOT}")
endif()
if(LEAPP_TEST_BACKENDS)
  list(APPEND LEAPP_ENGINE_TEST_ENV "LEAPP_TRITON_BACKENDS=${LEAPP_TEST_BACKENDS}")
endif()
