# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Slicing resolver operators for ARTIFACT_LIST."""
from typing import Sequence

from tfx import types as tfx_types
from tfx.dsl.input_resolution import resolver_op
from tfx.orchestration.portable.input_resolution import exceptions


class FirstNArtifacts(
    resolver_op.ResolverOp,
    canonical_name='tfx.FirstNArtifacts',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST,
):
  """FirstNArtifacts selects `artifacts[:n]`."""

  n = resolver_op.Property(type=int)
  skip_if_insufficient = resolver_op.Property(type=bool, default=False)

  def apply(
      self, artifacts: Sequence[tfx_types.Artifact]
  ) -> Sequence[tfx_types.Artifact]:
    if self.n < 1:
      raise exceptions.InvalidArgument('FirstNArtifacts.n should be >= 1.')
    if self.skip_if_insufficient and len(artifacts) < self.n:
      raise exceptions.SkipSignal(
          f'Skipped from FirstNArtifacts. Expected {self.n} elements but got '
          f'{len(artifacts)}.'
      )
    return artifacts[: self.n]


class LastNArtifacts(
    resolver_op.ResolverOp,
    canonical_name='tfx.LastNArtifacts',
    arg_data_types=(resolver_op.DataType.ARTIFACT_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST,
):
  """LastNArtifacts selects `artifacts[:n]`."""

  n = resolver_op.Property(type=int)
  skip_if_insufficient = resolver_op.Property(type=bool, default=False)

  def apply(
      self, artifacts: Sequence[tfx_types.Artifact]
  ) -> Sequence[tfx_types.Artifact]:
    if self.n < 1:
      raise exceptions.InvalidArgument('LastNArtifacts.n should be >= 1.')
    if self.skip_if_insufficient and len(artifacts) < self.n:
      raise exceptions.SkipSignal(
          f'Skipped from LastNArtifacts. Expected {self.n} elements but got '
          f'{len(artifacts)}.'
      )
    return artifacts[-self.n :]
