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
"""Module for TempLatestCreateTime operator."""

from typing import Sequence

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.utils import typing_utils


class TempLatestCreateTime(
    resolver_op.ResolverOp,
    canonical_name='tfx.dev.TempLatestCreateTime',
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP):
  """TempLatestCreateTime operator.

    It is a temporary resolver Op for development and testing. Do not use it in
    production.
    It is similar to LatestCreateTime Op, but it takes a dict as input.
  """

  # The number of latest artifacts to return, must be > 0.
  n = resolver_op.Property(type=int, default=1)

  def _select_latest_artifacts(
      self, input_list: Sequence[types.Artifact]) -> Sequence[types.Artifact]:
    input_list.sort(  # pytype: disable=attribute-error
        key=lambda a: (a.mlmd_artifact.create_time_since_epoch, a.id),
        reverse=True)
    return input_list[:self.n]

  def apply(
      self, input_dict: typing_utils.ArtifactMultiMap
  ) -> typing_utils.ArtifactMultiMap:
    """Returns the n latest created artifacts, ties broken by artifact id."""
    if not input_dict:
      return {}

    if self.n < 1:
      raise ValueError(f'n must be > 0, but was set to {self.n}.')

    return {
        key: self._select_latest_artifacts(value)
        for key, value in input_dict.items()
    }
