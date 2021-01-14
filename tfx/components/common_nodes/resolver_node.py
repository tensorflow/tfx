# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deprecated location for the TFX ResolverNode.

The new location is `tfx.dsl.components.common.resolver_node.ResolverNode`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfx.dsl.components.common import resolver_node
from tfx.utils import deprecation_utils

# Constant to access resolver class from resolver exec_properties.
RESOLVER_CLASS = resolver_node.RESOLVER_CLASS
# Constant to access resolver config from resolver exec_properties.
RESOLVER_CONFIGS = resolver_node.RESOLVER_CONFIGS

RESOLVER_CLASS_LIST = resolver_node.RESOLVER_CLASS_LIST
RESOLVER_CONFIG_LIST = resolver_node.RESOLVER_CONFIG_LIST

ResolverNode = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
    deprecated_name='tfx.components.common_nodes.resolver_node.ResolverNode',
    name='tfx.dsl.components.common.resolver_node.ResolverNode',
    func_or_class=resolver_node.ResolverNode)
