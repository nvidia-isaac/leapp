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

"""Compact YAML containers shared by semantics and graph descriptions."""

import yaml


class CompactYamlList(list):
    """Custom list class for tensor shapes that enables compact YAML formatting
    This class will behave exactly like a list. The only difference is that it will be
    formatted in a more compact way in the yaml file.

    """

    pass


class CompactYamlDict(dict):
    """Custom dict class for tensor shapes that enables compact YAML formatting
    This class will behave exactly like a dict. The only difference is that it will be
    formatted in a more compact way in the yaml file.
    """

    pass


# Register custom YAML representers so these types are always serialized in flow style
yaml.add_representer(
    CompactYamlList,
    lambda dumper, data: dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=True
    ),
)
yaml.add_representer(
    CompactYamlDict,
    lambda dumper, data: dumper.represent_mapping(
        "tag:yaml.org,2002:map", data, flow_style=True
    ),
)
