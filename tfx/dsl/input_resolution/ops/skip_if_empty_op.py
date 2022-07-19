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
"""Module for SkipIfEmpty operator."""

from typing import Sequence

from tfx.dsl.input_resolution import resolver_op
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.utils import typing_utils


class SkipIfEmpty(
    resolver_op.ResolverOp,
    canonical_name='tfx.internal.SkipIfEmpty',
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP_LIST,
):
  """SkipIfEmpty operator."""

  def apply(
      self,
      input_dict_list: Sequence[typing_utils.ArtifactMultiMap],
  ) -> Sequence[typing_utils.ArtifactMultiMap]:
    if not input_dict_list:
      raise exceptions.SkipSignal()
    return input_dict_list
