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
"""Shared utility functions for ResolverOps."""

from typing import List

from tfx import types

# TODO(b/241109157): Add a artifacts utility method.
# TODO(b/241109157): Add a verify_n, verify_offset, verify_min_span, etc.
# methods.


def filter_artifacts_by_span(
    artifacts: List[types.Artifact],
    span_descending: bool,
    n: int = 1,
    min_span: int = 0,
    skip_last_n: int = 0,
    keep_all_versions: bool = False) -> List[types.Artifact]:
  """Filters artifacts by their "span" PROPERTY.

  This should only be used a shared utility for LatestSpan and ConsecutiveSpans.

  Args:
    artifacts: The list of Artifacts to filter.
    span_descending: If true, then the artifacts will be sorted by span in
      descending order.  Else, they will be sorted in ascending order by span.
      Set to true for LatestSpan, and set to false for ConsecutiveSpans.
    n: The number of spans to return. If n <= 0, then n is set to the total
      number of unique spans.
    min_span: Minimum span before which no span will be considered.
    skip_last_n: Number of largest spans to skip. For example, if the spans are
      [1, 2, 3] and skip_last_n=1, then only spans [1, 2] will be considered.
    keep_all_versions: If true, all versions of the n spans are returned. Else,
      only the latest version is returned.

  Returns:
    The filtered artifacts.
  """
  if not artifacts:
    return []

  # Only keep artifacts with spans >= min_span and account for skip_last_n
  spans = sorted({a.span for a in artifacts if a.span >= min_span})
  if skip_last_n:
    spans = spans[:-skip_last_n]

  # Sort spans in descending order, if specified.
  if span_descending:
    spans = spans[::-1]

  # Keep n spans, if n is positive.
  if n > 0:
    spans = spans[:n]

  if not spans:
    return []

  artifacts_by_span = {}
  for artifact in artifacts:
    artifacts_by_span.setdefault(artifact.span, []).append(artifact)

  result = []
  version_and_id = lambda a: (a.version, a.id)
  for span in spans:
    if keep_all_versions:
      # span_descending only applies to sorting by span, but version should
      # always be sorted in ascending order.
      result.extend(sorted(artifacts_by_span[span], key=version_and_id))
    else:
      # Latest version is defined as the largest version. Ties broken by id.
      result.append(max(artifacts_by_span[span], key=version_and_id))

  return result
