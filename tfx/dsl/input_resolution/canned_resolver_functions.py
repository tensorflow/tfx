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

  Composes the ExcludeSpans(StaticSpanRange(artifacts)) ResolverOps. Corresponds
  to StaticRange in TFX.

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
