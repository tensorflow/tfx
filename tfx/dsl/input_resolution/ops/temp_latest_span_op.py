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
"""Module for a temporary LatestSpan operator."""

from typing import Sequence

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.utils import typing_utils


class TempLatestSpan(
    resolver_op.ResolverOp,
    canonical_name='tfx.TempLatestSpan',
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP):
  """TempLatestSpan operator.

    It is a temporary resolver Op for development and testing. Do not use it in
    production.
    It is similar to LatestSpan Op, but it takes a dict as input.
  """

  # The number of latest spans to return, must be > 0.
  n = resolver_op.Property(type=int, default=1)

  # If true, all versions of the n spans are returned. Else, only the latest
  # version is returned.
  keep_all_versions = resolver_op.Property(type=bool, default=False)

  def _select_latest_span(
      self, input_list: Sequence[types.Artifact]) -> Sequence[types.Artifact]:
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

    # Only keep artifacts with the n latest spans.
    spans = list(set([a.span for a in valid_artifacts]))
    spans.sort(reverse=True)
    spans = spans[0:min(self.n, len(spans))]
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

  def apply(
      self, input_list: typing_utils.ArtifactMultiMap
  ) -> typing_utils.ArtifactMultiMap:
    """Returns artifacts with the n latest spans.

    For example, if n=2, then only 2 artifacts with the latest 2 spans and
    latest versions are returned. If n=2 and all_versions=True, then all
    artifacts with the latest 2 spans but with all versions are included.

    Args:
      input_list: The list of Artifacts to parse.

    Returns:
      Artifacts with the n latest spans, all versions included.
    """
    if self.n < 1:
      raise ValueError(f'n must be > 0, but was set to {self.n}.')

    return {
        key: self._select_latest_span(value)
        for key, value in input_list.items()
    }
