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
"""Module for public facing, canned resolver functions."""

from typing import Sequence

from tfx.dsl.input_resolution import resolver_function
from tfx.dsl.input_resolution.ops import ops


@resolver_function.resolver_function
def latest_created(artifacts, n: int = 1):
  """Returns the n latest createst artifacts, ties broken by artifact id.

  Args:
    artifacts: The artifacts to filter.
    n: The number of latest artifacts to return, must be > 0.

  Returns:
    The n latest artifacts.
  """
  return ops.LatestCreateTime(artifacts, n=n)


@resolver_function.resolver_function
def static_range(artifacts,
                 *,
                 start_span_number: int = -1,
                 end_span_number: int = -1,
                 keep_all_versions: bool = False,
                 exclude_span_numbers: Sequence[int] = ()):
  """Returns artifacts with spans in [start_span, end_span] inclusive.

  Artifacts are expected to have both a span and a version. If there are
  multiple artifacts with the same span, then only the one with the latest
  version is considered, if keep_all_versions=False. If keep_all_versions=True,
  then all artifacts, even those with duplicate spans, are considered.

  Please note that the spans in exclude_span_numbers are excluded AFTER getting
  the artifacts with spans in the range.

  Corresponds to StaticRange in TFX.

  Example usage:

    Consider 8 artifacts with:
      spans    = [0, 1, 2, 3, 3, 5, 7, 10]
      versions = [0, 0, 0, 0, 3, 0, 0, 0]

    static_range(
      end_span_number=5,
      keep_all_versions=False,
      exclude_span_numbers=[2])

    Because start_span_number = -1, it is set to the smallest span, 0.

    Spans in the range [0, 5] will be considered.

    Because keep_all_versions=False, only the artifact with span=3 and version=3
    will be considered, even though there are two artifacts with span=3.

    Because exclude_span_numbers=[2], the artifacts with span=2 will not be
    kept, even though it is in the range.

    The artifacts that will be returned are:
      spans    = [0, 1, 3, 5]
      versions = [0, 0, 3, 0]

  Args:
    artifacts: The artifacts to filter.
    start_span_number: The smallest span number to keep, inclusive. If < 0, set
      to the the smallest span in the artifacts.
    end_span_number: The largest span number to keep, inclusive. If < 0, set to
      the largest span in the artifacts.
    keep_all_versions: If true, all artifacts with spans in the range are kept.
      If false then if multiple artifacts have the same span, only the span with
      the latest version is kept. Defaults to False.
    exclude_span_numbers: The span numbers to exclude.

  Returns:
    Artifacts with spans in [start_span, end_span] inclusive.
  """
  resolved_artifacts = ops.StaticSpanRange(
      artifacts,
      start_span=start_span_number,
      end_span=end_span_number,
      keep_all_versions=keep_all_versions)
  if exclude_span_numbers:
    resolved_artifacts = ops.ExcludeSpans(
        resolved_artifacts, denylist=exclude_span_numbers)
  return resolved_artifacts


@resolver_function.resolver_function
def rolling_range(artifacts,
                  *,
                  start_span_number: int = 0,
                  num_spans: int = 1,
                  skip_num_recent_spans: int = 0,
                  keep_all_versions: bool = False,
                  exclude_span_numbers: Sequence[int] = ()):
  """Returns artifacts with spans in a rolling range.

  First, spans < start_span_number are excluded, and then the spans are sorted.
  Then, artifacts with spans in
  sorted_spans[:-skip_num_recent_spans][-num_spans:] are returned.

  Note that the missing spans are not counted in num_spans. The artifacts do NOT
  have to have consecutive spans.

  Artifacts are expected to have both a span and a version. If there are
  multiple artifacts with the same span, then only the one with the latest
  version is considered, if keep_all_versions=False. If keep_all_versions=True,
  then all artifacts, even those with duplicate spans, are considered.

  Please note that the spans in exclude_span_numbers are excluded AFTER getting
  the latest spans.

  Corresponds to RollingRange in TFX.

  Example usage:

    Consider 6 artifacts with:
      spans    = [1, 2, 3, 3, 7, 8]
      versions = [0, 0, 1, 0, 1, 2]

    rolling_range(
        start_span_number=3,
        num_spans=5,
        skip_num_recent_spans=1,
        keep_all_versions=True,
        exclude_span_numbers=[7])

    spans 1 and 2 are removed because they are < start_span_number=3. The
    sorted unique spans are [3, 7, 8].

    span 8 is removed because skip_num_recent_spans=1, leaving spans [3, 7].

    Although num_spans=5, only two unique span numbers are availble, 3 and 7,
    so both spans [3, 7] are kept.

    Because keep_all_versions=True, both artifacts with span=3 are kept.

    Because exclude_span_numbers=[7], the artifact with span=7 will not be
    kept, even though it is in the range.

    The artifacts that will be returned are:
      spans = [3, 3]
      versions = [1, 0]

  Args:
    artifacts: The artifacts to filter.
    start_span_number: The smallest span number to keep, inclusive. Defaults to
      0.
    num_spans: The length of the range. If num_spans <= 0, then num_spans is set
      to the total number of unique spans.
    skip_num_recent_spans: Number of most recently available (largest) spans to
      skip. Defaults to 0.
    keep_all_versions: If true, all artifacts with spans in the range are kept.
      If false then if multiple artifacts have the same span, only the span with
      the latest version is kept. Defaults to False.
    exclude_span_numbers: The span numbers to exclude.

  Returns:
    Artifacts with spans in the rolling range.
  """
  resolved_artifacts = ops.LatestSpan(
      artifacts,
      min_span=start_span_number,
      n=num_spans,
      skip_last_n=skip_num_recent_spans,
      keep_all_versions=keep_all_versions)
  if exclude_span_numbers:
    resolved_artifacts = ops.ExcludeSpans(
        resolved_artifacts, denylist=exclude_span_numbers)
  return resolved_artifacts


@resolver_function.resolver_function
def all_spans(artifacts,
              *,
              span_descending: bool = False,
              keep_all_versions: bool = False):
  """Returns the sorted artifacts with unique spans.

  If keep_all_versions = False, then all artifacts with unique spans (ties
  broken by version) are returned. Else, all artifacts are returned.

  The artifacts

  Example usage:

    Consider 6 artifacts with:
      spans    = [1, 3, 3, 2, 8, 7]
      versions = [0, 0, 1, 0, 1, 2]

    all_spans(
        span_descending=False,
        keep_all_versions=False)

    will return artifacts:
      spans    = [1, 2, 3, 7, 8]
      versions = [0, 0, 1, 1, 2]

    Note that there are 2 artifacts with span 3, but only the one with the
    latest version is returned. Spans are sorted in ascending order.

    all_spans(
        span_descending=True,
        keep_all_versions=True)

    will return all the artifacts:
      spans    = [8, 7, 3, 3, 2, 1]
      versions = [2, 1, 0, 1, 0, 0]

    Note that both artifacts with span 3 are returned. Spans are sorted in
    descending order, but versions are always sorted ascending order.

  Args:
    artifacts: The artifacts to filter.
    span_descending: If true, then the artifacts will be sorted by span in
      descending order. Else, they will be sorted in ascending order by span.
      Note that sorting happens first by span and then by version, and that
      version is always sorted in ascending order.
    keep_all_versions: If true, all artifacts with spans in the range are kept.
      If false then if multiple artifacts have the same span, only the span with
      the latest version is kept. Defaults to False.

  Returns:
    Sorted Artifacts with unique spans.
  """
  return ops.AllSpans(
      artifacts,
      span_descending=span_descending,
      keep_all_versions=keep_all_versions)
