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
"""Module for StaticSpanRange operator."""

from typing import Optional, Sequence

from tfx import types
from tfx.dsl.input_resolution import resolver_op


class StaticSpanRange(
    resolver_op.ResolverOp,
    canonical_name='tfx.StaticSpanRange',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST):
  """StaticSpanRange operator."""

  # The smallest span number to keep, inclusive.
  start_span = resolver_op.Property(type=Optional[int], default=None)

  # The largest span number to keep, inclusive
  end_span = resolver_op.Property(type=Optional[int], default=None)

  # If true, all spans are kept. If false, then if multiple artifacts have the
  # same span, only the span with the latest version is kept.
  keep_all = resolver_op.Property(type=bool, default=False)

  def apply(self,
            input_list: Sequence[types.Artifact]) -> Sequence[types.Artifact]:
    """Returns artifacts with spans in [start_span, end_span] inclusive.

    If start_span is not specified, then the smallest span among the artifacts
    in input_list is chosen. Similarly, if end_span is not specified, then the
    largest span among the artifacts in input_list is chosen.

    If multiple artifacts have the same span, then the span with the latest
    version is chosen.

    This ResolverOp corresponds to StaticRange in TFX.

    Args:
      input_list: A list of artifacts.
    """
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

    if self.start_span is None:
      self.start_span = min(valid_artifacts, key=lambda a: a.span).span

    if self.end_span is None:
      self.end_span = max(valid_artifacts, key=lambda a: a.span).span

    # Sort the artifacts by span and then by version, in ascending order.
    valid_artifacts.sort(key=lambda a: (a.span, a.version))  # pytype: disable=attribute-error

    # Only consider artifacts with spans are in [start_span, end_span]
    # inclusive.
    span_artifact_map = {}
    for artifact in valid_artifacts:
      if self.start_span <= artifact.span <= self.end_span:
        span_artifact_map.setdefault(artifact.span, []).append(artifact)

    result = []
    for artifacts in span_artifact_map.values():
      if self.keep_all:
        result.extend(sorted(artifacts, key=lambda a: a.version))
      else:
        # If keep_all is False we take the artifact with the latest (largest)
        # version.
        result.append(max(artifacts, key=lambda a: a.version))
    return result
