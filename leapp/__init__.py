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

"""
LEAPP - Lightweight Export Annotations for Policy Pipelines

A Python package for tracing and exporting computational graphs from PyTorch code.
LEAPP is specifically designed for robotics and autonomous agent applications, allowing 
you to trace and export complex policy pipelines with interconnected components to 
various formats including PyTorch JIT, ONNX, and generate visualization and YAML specifications.
"""

from .export_manager import ExportManager
from .inference_manager import InferenceManager
from .leapp import annotate, start, stop, compile_graph
from .utils.enums import InputKindEnum, OutputKindEnum
from .utils.tensor_description import TensorSemantics

__version__ = "0.5.0"
__config_version__ = "1.1"
__author__ = "Frank Lai"
__email__ = "frlai@nvidia.com"

__all__ = [
    "ExportManager",
    "InferenceManager",
    "InputKindEnum",
    "OutputKindEnum",
    "annotate",
    "start",
    "stop",
    "compile_graph",
    "__version__",
    "TensorSemantics",
]
