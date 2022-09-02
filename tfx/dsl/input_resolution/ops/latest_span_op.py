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


class LatestSpan(
    resolver_op.ResolverOp,
    canonical_name='tfx.LatestSpan',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST):
  """LatestSpan operator."""

  # The number of latest spans to return, must be > 0.
  n = resolver_op.Property(type=int, default=1)

  # If true, all versions of the n spans are returned. Else, only the latest
  # version is returned.
  keep_all_versions = resolver_op.Property(type=bool, default=False)

  # If true, then n is set to the total number of unique spans.
  keep_all_spans = resolver_op.Property(type=bool, default=False)

  # Minimum span before which no span will be considered.
  min_span = resolver_op.Property(type=int, default=0)

  # Number of most recently available spans to skip.
  offset = resolver_op.Property(type=int, default=0)

  def apply(self,
            input_list: Sequence[types.Artifact]) -> Sequence[types.Artifact]:
    """Returns artifacts with the n latest spans.

    For example, if n=2, then only 2 artifacts with the latest 2 spans and
    latest versions are returned. If n=2 and all_versions=True, then all
    artifacts with the latest 2 spans but with all versions are included.

    Spans that are < min_span will be ignored. Additionally, the first
    offset number of spans will be ignored. For example, if min_span=2,
    offset=1, and the spans are [4, 3, 2, 1], then only spans [3, 2] will be
    kept.

    min_span and offset correspond to start_span_num and skip_num_recent_spans
    in the TFX RollingRange proto.

    Args:
      input_list: The list of Artifacts to parse.

    Returns:
      Artifacts with the n latest spans, all versions included.
    """
    # Verify that n, min_span, and offset are >= to their minimum values.
    if self.n < 1:
      raise ValueError(f'n must be > 0, but was set to {self.n}.')

    if self.offset < 0:
      raise ValueError(f'offset must be >= 0, but was set to '
                       f'{self.offset}.')

    if self.min_span < 0:
      raise ValueError(f'min_span must be >= 0, but was set to '
                       f'{self.min_span}.')

    # Only consider artifacts that have both "span" and "version" in PROPERTIES
    # with PropertyType.INT.
    valid_artifacts = []
    for artifact in input_list:
      if artifact.PROPERTIES is None:
        continue

      if ('span' not in artifact.PROPERTIES or
          artifact.PROPERTIES['span'].type != types.artifact.PropertyType.INT):
        continue

      if ('version' not in artifact.PROPERTIES or
          artifact.PROPERTIES['version'].type !=
          types.artifact.PropertyType.INT):
        continue

      valid_artifacts.append(artifact)

    if not valid_artifacts:
      return []

    valid_artifacts.sort(key=lambda a: a.span, reverse=True)

    # Only keep artifacts with spans >= self.min_span.
    spans = list(set([a.span for a in valid_artifacts]))
    spans = [s for s in spans if s >= self.min_span]
    spans.sort(reverse=True)

    if not spans:
      return []

    # Only keep artifacts with the n latest spans, accounting for
    # offset.
    if self.keep_all_spans:
      spans = spans[self.offset:]
    else:
      spans = spans[self.offset:self.offset + self.n]

    valid_artifacts = [a for a in valid_artifacts if a.span in spans]

    if self.keep_all_versions:
      return valid_artifacts

    # Only keep artifacts with the latest version.
    span_artifact_map = {}
    for artifact in valid_artifacts:
      span = artifact.span

      if span not in span_artifact_map:
        span_artifact_map[span] = artifact
        continue

      # Latest version is defined as the largest version. Ties broken by id.
      span_artifact_map[span] = max(
          artifact, span_artifact_map[span], key=lambda a: (a.version, a.id))

    return list(span_artifact_map.values())
