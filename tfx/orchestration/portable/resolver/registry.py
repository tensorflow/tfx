# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Registry for all public BaseResolvers."""
import importlib
from typing import Text

from tfx.dsl.resolvers import base_resolver


class ResolverRegistry:
  """Registry for all `BaseResolver`s."""

  def __init__(self):
    self._load_all_resolvers()

  def _load_all_resolvers(self):
    importlib.import_module('tfx.dsl.experimental.latest_artifacts_resolver')
    importlib.import_module(
        'tfx.dsl.experimental.latest_blessed_model_resolver')

  def get(self, resolver_cls_name: Text):
    for cls in base_resolver.BaseResolver.__subclasses__():
      if cls.__name__ == resolver_cls_name:
        return cls
    raise NameError(f'Unknown resolver class {resolver_cls_name}.')
