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
"""Module for AllSpans operator."""

from typing import Sequence

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops_utils


class AllSpans(
    resolver_op.ResolverOp,
    canonical_name='tfx.AllSpans',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST):
  """AllSpans operator."""

  # If true, all versions of the n spans are returned. Else, only the latest
  # version is returned.
  keep_all_versions = resolver_op.Property(type=bool, default=False)

  # If true, then the artifacts will be sorted by span in descending order.
  # Else, they will be sorted in ascending order by span.
  span_descending = resolver_op.Property(type=bool, default=False)

  def apply(self,
            input_list: Sequence[types.Artifact]) -> Sequence[types.Artifact]:
    """Returns the sorted artifacts with unique spans."""

    # Get artifacts with "span" and "version" in PROPERTIES.
    valid_artifacts = ops_utils.get_valid_artifacts(
        input_list, ops_utils.SPAN_AND_VERSION_PROPERTIES)
    if not valid_artifacts:
      return []

    # Return the sorted artifacts.
    return ops_utils.filter_artifacts_by_span(
        artifacts=valid_artifacts,
        span_descending=self.span_descending,
        n=0,  # n = 0 so that all the spans are considered.
        keep_all_versions=self.keep_all_versions,
    )
