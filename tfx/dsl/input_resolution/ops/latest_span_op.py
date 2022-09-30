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
"""Module for LatestSpan operator."""

from typing import Sequence

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops_utils


class LatestSpan(
    resolver_op.ResolverOp,
    canonical_name='tfx.LatestSpan',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST):
  """LatestSpan operator."""

  # The number of latest spans to return. If n <= 0, then n is set to the total
  # number of unique spans.
  n = resolver_op.Property(type=int, default=1)

  # If true, all versions of the n spans are returned. Else, only the latest
  # version is returned.
  keep_all_versions = resolver_op.Property(type=bool, default=False)

  # Minimum span before which no span will be considered.
  min_span = resolver_op.Property(type=int, default=0)

  # Number of most recently available (largest) spans to skip.
  skip_last_n = resolver_op.Property(type=int, default=0)

  def apply(self,
            input_list: Sequence[types.Artifact]) -> Sequence[types.Artifact]:
    """Returns artifacts with the n latest spans.

    For example, if n=2, then only 2 artifacts with the latest 2 spans and
    latest versions are returned. If n=2 and all_versions=True, then all
    artifacts with the latest 2 spans but with all versions are included.

    Spans that are < min_span will be ignored. Additionally, the last
    skip_last_n number of spans will be ignored. For example, if min_span=2,
    skip_last_n=1, and the spans are [1, 2, 3, 4], then only spans [2, 3] will
    be kept.

    min_span and skip_last_n correspond to start_span_num and
    skip_num_recent_spans in the TFX RollingRange proto.

    Args:
      input_list: The list of Artifacts to parse.

    Returns:
      Artifacts with the n latest spans, all versions included.
    """
    # Verify that min_span and skip_last_n are >= to their minimum values.
    if self.skip_last_n < 0:
      raise ValueError(f'skip_last_n must be >= 0, but was set to '
                       f'{self.skip_last_n}.')

    if self.min_span < 0:
      raise ValueError(f'min_span must be >= 0, but was set to '
                       f'{self.min_span}.')

    valid_artifacts = ops_utils.get_valid_artifacts(
        input_list, ops_utils.SPAN_AND_VERSION_PROPERTIES)
    if not valid_artifacts:
      return []

    return ops_utils.filter_artifacts_by_span(
        artifacts=valid_artifacts,
        span_descending=True,
        n=self.n,
        min_span=self.min_span,
        skip_last_n=self.skip_last_n,
        keep_all_versions=self.keep_all_versions,
    )
