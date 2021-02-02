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
"""BaseResolver instance factory for portable orchestrator."""
import importlib

from tfx.dsl.components.common import resolver
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import json_utils


def make_resolver_strategy_instance(
    resolver_step: pipeline_pb2.ResolverConfig.ResolverStep
) -> resolver.ResolverStrategy:
  """Creates Resolver instance from ResolverStep."""
  module_name, class_name = resolver_step.class_path.rsplit('.', maxsplit=1)
  module = importlib.import_module(module_name)
  resolver_strategy_cls = getattr(module, class_name)
  if not issubclass(resolver_strategy_cls, resolver.ResolverStrategy):
    raise TypeError(f'Resolver strategy class should be a subclass of '
                    f'{resolver.__name__}.'
                    f'{resolver.ResolverStrategy.__name__}.')
  config = json_utils.loads(resolver_step.config_json)
  return resolver_strategy_cls(**config)
