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
"""Deprecated location for the TFX Resolver.

The new location is `tfx.dsl.components.common.resolver.Resolver`.
"""

from typing import Dict, Type

from tfx import types
from tfx.dsl.components.common import resolver
from tfx.utils import deprecation_utils
from tfx.utils import json_utils


def _make_deprecated_resolver_node_alias():
  """Make ResolverNode alias class.

  Make the deprecation shim for ResolverNode.  Needed to conform to the
  convention expected by `tfx.utils.deprecation_utils` and to translate renamed
  constructor arguments.

  Returns:
      Deprecated ResolverNode alias class.
  """
  parent_deprecated_class = deprecation_utils.deprecated_alias(  # pylint: disable=invalid-name
      deprecated_name='tfx.components.common_nodes.resolver_node.ResolverNode',
      name='tfx.dsl.components.common.resolver.Resolver',
      func_or_class=resolver.Resolver)

  class _NewDeprecatedClass(parent_deprecated_class):
    """Deprecated ResolverNode alias constructor.

    This class location is DEPRECATED and is provided temporarily for
    compatibility. Please use `tfx.dsl.components.common.resolver.Resolver`
    instead.
    """

    def __init__(self,
                 resolver_class: Type[resolver.ResolverStrategy] = None,
                 resolver_configs: Dict[str, json_utils.JsonableType] = None,
                 **kwargs: types.Channel):
      """Forwarding shim for deprecated ResolverNode alias constructor.

      Args:
        resolver_class: a ResolverStrategy subclass which contains the artifact
          resolution logic.
        resolver_configs: a dict of key to Jsonable type representing
          configuration that will be used to construct the resolver strategy.
        **kwargs: a key -> Channel dict, describing what are the Channels to be
          resolved. This is set by user through keyword args.
      """
      super(ResolverNode, self).__init__(
          strategy_class=resolver_class,
          config=resolver_configs,
          **kwargs)

  return _NewDeprecatedClass

# Constant to access resolver class from resolver exec_properties.
RESOLVER_CLASS = resolver.RESOLVER_STRATEGY_CLASS
# Constant to access resolver config from resolver exec_properties.
RESOLVER_CONFIGS = resolver.RESOLVER_CONFIG

ResolverNode = _make_deprecated_resolver_node_alias()
