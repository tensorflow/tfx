# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Experimental Resolver for evaluating the condition."""

from typing import Dict, List, Optional

from tfx import types
from tfx.dsl.compiler import placeholder_utils
from tfx.dsl.components.common import resolver
from tfx.orchestration import metadata
from tfx.orchestration.portable import data_types as portable_data_types
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.proto.orchestration import placeholder_pb2


class ConditionalStrategy(resolver.ResolverStrategy):
  """Strategy that resolves artifacts if predicates are met.

  This resolver strategy is used by TFX internally to support conditional.
  Not intended to be directly used by users.
  """

  def __init__(self, predicates: List[placeholder_pb2.PlaceholderExpression]):
    self._predicates = predicates

  def resolve_artifacts(
      self, metadata_handler: metadata.Metadata,
      input_dict: Dict[str, List[types.Artifact]]
  ) -> Optional[Dict[str, List[types.Artifact]]]:
    for placeholder_pb in self._predicates:
      context = placeholder_utils.ResolutionContext(
          exec_info=portable_data_types.ExecutionInfo(input_dict=input_dict))
      predicate_result = placeholder_utils.resolve_placeholder_expression(
          placeholder_pb, context)
      if not isinstance(predicate_result, bool):
        raise ValueError("Predicate evaluates to a non-boolean result.")

      if not predicate_result:
        raise exceptions.SkipSignal("Predicate evaluates to False.")
    return input_dict
