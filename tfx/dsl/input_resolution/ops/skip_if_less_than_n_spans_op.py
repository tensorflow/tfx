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
"""Module for SkipIfLessThanNSpans operator."""

from typing import Sequence

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.orchestration.portable.input_resolution import exceptions


class SkipIfLessThanNSpans(
    resolver_op.ResolverOp,
    canonical_name='tfx.internal.SkipIfLessThanNSpans',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST,
):
  """SkipIfLessThanNSpans operator."""

  # The minimum number of unique spans that must be present in the artifacts.
  # If < 0, then all the artifacts are returned.
  n = resolver_op.Property(type=int, default=0)

  def apply(
      self,
      input_list: Sequence[types.Artifact],
  ) -> Sequence[types.Artifact]:
    """Raises a SkipSignal if the artifacts have less than n unique spans.

    For example, if the artifacts have spans [1, 3, 6, 7, 8] but n = 7, then a
    SkipSignal will be raised.

    Corresponds to min_spans in the TFX RangeConfig proto.

    Args:
      input_list: The artifacts to check.

    Returns:
      The same artifacts passed in, unmodified.

    Raises:
      SkipSignal if the artifacts have less than n unique spans.
    """
    spans = set()
    for artifact in input_list:
      spans.add(artifact.span)

    if self.n >= 0 and len(spans) < self.n:
      raise exceptions.SkipSignal(
          f'[SkipIfLessThanNSpans] len(spans): {len(spans)} < N: {self.n}, '
          'skipping.')

    return input_list
