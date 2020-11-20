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
from tfx.dsl.resolvers import base_resolver
from tfx.orchestration.portable.resolver import registry
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import json_utils


_REGISTRY = registry.ResolverRegistry()


def get_resolver_instance(
    resolver_pb: pipeline_pb2.ResolverConfig.Resolver
) -> base_resolver.BaseResolver:
  resolver_cls = _REGISTRY.get(resolver_pb.name)
  resolver_config = json_utils.loads(resolver_pb.config_json)
  return resolver_cls(**resolver_config)
