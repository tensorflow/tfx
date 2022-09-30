# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Module for LatestVersion operator."""

from typing import Sequence

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops_utils


class LatestVersion(
    resolver_op.ResolverOp,
    canonical_name='tfx.LatestVersion',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST):
  """LatestVersion operator."""

  # The number of latest artifacts to return.
  n = resolver_op.Property(type=int, default=1)

  # If true, then n is set to the total number of unique spans.
  keep_all = resolver_op.Property(type=bool, default=False)

  def apply(self,
            input_list: Sequence[types.Artifact]) -> Sequence[types.Artifact]:
    """Returns n artifacts with the latest version, ties broken by id."""
    if not input_list:
      return []

    valid_artifacts = ops_utils.get_valid_artifacts(input_list,
                                                    ops_utils.VERSION_PROPERTY)
    if not valid_artifacts:
      return []

    valid_artifacts.sort(  # pytype: disable=attribute-error
        key=lambda a: (a.version, a.id),
        reverse=True)

    if self.keep_all:
      return valid_artifacts

    return valid_artifacts[0:self.n]
