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
"""Slicing resolver operators for ARTIFACT_MULTIMAP_LIST."""
from typing import Sequence

from tfx.dsl.input_resolution import resolver_op
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.utils import typing_utils


class FirstArtifactMapOrSkip(
    resolver_op.ResolverOp,
    canonical_name='tfx.FirstArtifactMapOrSkip',
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP,
):
  """FirstArtifactMapOrSkip selects `artifact_map_list[0]`."""

  def apply(
      self, artifact_map_list: Sequence[typing_utils.ArtifactMultiMap]
      ) -> typing_utils.ArtifactMultiMap:
    if not artifact_map_list:
      raise exceptions.SkipSignal('Skipped from FirstArtifactMapOrSkip.')
    return artifact_map_list[0]


class LastArtifactMapOrSkip(
    resolver_op.ResolverOp,
    canonical_name='tfx.LastArtifactMapOrSkip',
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP,
):
  """LastArtifactMapOrSkip selects `artifact_map_list[-1]`."""

  def apply(
      self, artifact_map_list: Sequence[typing_utils.ArtifactMultiMap]
      ) -> typing_utils.ArtifactMultiMap:
    if not artifact_map_list:
      raise exceptions.SkipSignal('Skipped from LastArtifactMapOrSkip.')
    return artifact_map_list[-1]


class FirstNArtifactMaps(
    resolver_op.ResolverOp,
    canonical_name='tfx.FirstNArtifactMaps',
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP_LIST,
):
  """FirstNArtifactMaps selects `artifact_map_list[:n]`."""
  n = resolver_op.Property(type=int)
  skip_if_insufficient = resolver_op.Property(type=bool, default=False)

  def apply(
      self, artifact_map_list: Sequence[typing_utils.ArtifactMultiMap]
      ) -> Sequence[typing_utils.ArtifactMultiMap]:
    if self.n < 1:
      raise exceptions.InvalidArgument('FirstNArtifactMaps.n should be >= 1.')
    if self.skip_if_insufficient and len(artifact_map_list) < self.n:
      raise exceptions.SkipSignal(
          f'Skipped from FirstNArtifactMaps. Expected {self.n} elements but '
          f'got {len(artifact_map_list)}.')
    return artifact_map_list[:self.n]


class LastNArtifactMaps(
    resolver_op.ResolverOp,
    canonical_name='tfx.LastNArtifactMaps',
    arg_data_types=(resolver_op.DataType.ARTIFACT_MULTIMAP_LIST,),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP_LIST,
):
  """LastNArtifactMaps selects `artifact_map_list[-n:]`."""
  n = resolver_op.Property(type=int)
  skip_if_insufficient = resolver_op.Property(type=bool, default=False)

  def apply(
      self, artifact_map_list: Sequence[typing_utils.ArtifactMultiMap]
      ) -> Sequence[typing_utils.ArtifactMultiMap]:
    if self.n < 1:
      raise exceptions.InvalidArgument('LastNArtifactMaps.n should be >= 1.')
    if self.skip_if_insufficient and len(artifact_map_list) < self.n:
      raise exceptions.SkipSignal(
          f'Skipped from LastNArtifactMaps. Expected {self.n} elements but got '
          f'{len(artifact_map_list)}.')
    return artifact_map_list[-self.n:]
