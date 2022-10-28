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
"""Module for ConsecutiveSpans operator."""

from typing import Sequence

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops_utils


class ConsecutiveSpans(
    resolver_op.ResolverOp,
    canonical_name='tfx.ConsecutiveSpans',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST):
  """ConsecutiveSpans operator."""

  # If true, all versions of the spans are returned. Else, only the latest
  # version is returned.
  keep_all_versions = resolver_op.Property(type=bool, default=False)

  # The first span number that consecutive spans should start with. If not
  # specified, the artifact with the smallest span is chosen.
  first_span = resolver_op.Property(type=int, default=-1)

  # Number of consecutive spans to skip at the tail end of the range.
  skip_last_n = resolver_op.Property(type=int, default=0)

  # The span numbers to exclude.
  denylist = resolver_op.Property(type=Sequence[int], default=[])

  def apply(self,
            input_list: Sequence[types.Artifact]) -> Sequence[types.Artifact]:
    """Returns the artifacts with the oldest consecutive spans.

    The consecutive spans in the range
    [first_span, max_span - skip_last_n] inclusive are considered. max_span is
    the largest span present in the artifacts. If there is an artifact with a
    span corresponding to each integer in that range, then the artifacts with
    spans in that range are returned.

    For example, if the artifacts have spans [1, 2, 3, 4, 5, 6], first_span=2,
    and skip_last_n=2, max_span is calculated as 6, and the range to consider is
    [2, 6 - 2] = [2, 4].

    If instead the artifacts have spans [1, 2, 3, 4, 6] first_span=2,
    and skip_last_n=2, the range to consider is still [2, 4]. However, only
    artifacts with spans [1, 2, 3, 4] will be returned because there is no
    artifact with span 5, so the the consecutive spans stop at 4.

    Args:
      input_list: The list of Artifacts to filter.

    Returns:
      Artifacts with spans in the range [first_span, max_span - skip_last_n].
      The artifacts are sorted in ascending order, first by span and then by
      version (if keep_all_versions = True).
    """
    if self.skip_last_n < 0:
      raise ValueError(f'skip_last_n must be >= 0, but was set to '
                       f'{self.skip_last_n}.')

    valid_artifacts = ops_utils.get_valid_artifacts(
        input_list, ops_utils.SPAN_AND_VERSION_PROPERTIES)
    if not valid_artifacts:
      return []

    # Override first_span to the smallest span in artifacts if it is < 0.
    if self.first_span < 0:
      self.first_span = min(a.span for a in valid_artifacts)

    # Increment first_span so that it is not in the denylist.
    while self.first_span in set(self.denylist):
      self.first_span += 1

    # Perform initial filtering to get artifacts with the relevant spans.
    artifacts = ops_utils.filter_artifacts_by_span(
        artifacts=valid_artifacts,
        span_descending=False,
        n=-1,
        min_span=self.first_span,
        # Pass 0 for skip_last_n, because in filter_artifacts_by_span it removes
        # the largest N span values (based on the index not by the value). We
        # instead manually filter the spans later.
        skip_last_n=0,
        keep_all_versions=self.keep_all_versions,
    )

    # Only consider spans in the range [first_span, max_span - skip_last_n].
    max_span = max(a.span for a in valid_artifacts)
    last_span = max_span - self.skip_last_n

    # Determine the last valid span accounting for the denylist.
    for span in sorted(self.denylist):
      if span <= self.first_span:
        continue

      last_span = min(span - 1, last_span)
      break

    valid_spans = set(range(self.first_span, last_span + 1))
    actual_spans = {a.span for a in artifacts}

    # Return the artifacts with consecutive spans.
    consecutive_spans = set()
    for i, span in enumerate(sorted(valid_spans & actual_spans)):
      # Break at the first instance the spans are no longer consecutive.
      if span != self.first_span + i:
        break
      consecutive_spans.add(span)

    return [a for a in artifacts if a.span in consecutive_spans]
