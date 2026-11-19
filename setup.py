#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Copy the TensorRT catalogs into the built ``isaac_deploy_trt`` package.

The catalogs live at ``packages/isaac_deploy_trt/tensorrt_catalogs``. setuptools
only ships files that live inside a package directory, so the build copies
them into ``isaac_deploy_trt/tensorrt_catalogs`` for the wheel.
"""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

_CATALOGS = (
    Path(__file__).resolve().parent / "packages" / "isaac_deploy_trt" / "tensorrt_catalogs"
)


class build_py(_build_py):
    def run(self):
        super().run()
        dst = Path(self.build_lib) / "isaac_deploy_trt" / "tensorrt_catalogs"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(_CATALOGS, dst)


setup(cmdclass={"build_py": build_py})
