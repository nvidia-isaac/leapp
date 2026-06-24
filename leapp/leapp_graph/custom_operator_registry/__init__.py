#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Registry of LEAPP custom operators.

Importing this package registers the custom ops it owns (currently
``leapp::warp_runner``) as an import side effect.
"""

from . import warp_custom_op as warp_custom_op
