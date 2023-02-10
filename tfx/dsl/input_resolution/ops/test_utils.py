# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Testing utility for builtin resolver ops."""
from typing import Type, Any, Optional, Tuple, Mapping
from unittest import mock

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops_utils
from tfx.types import artifact
from tfx.utils import typing_utils

import ml_metadata as mlmd


class DummyArtifact(types.Artifact):
  """A dummy Artifact used for testing."""

  TYPE_NAME = 'DummyArtifact'

  PROPERTIES = {
      'span': artifact.Property(type=artifact.PropertyType.INT),
      'version': artifact.Property(type=artifact.PropertyType.INT),
  }

  # pylint: disable=redefined-builtin
  def __init__(
      self,
      id: Optional[str] = None,
      uri: Optional[str] = None,
      create_time_since_epoch: Optional[int] = None,
  ):
    super().__init__()
    if id is not None:
      self.id = id
    if uri is not None:
      self.uri = uri
    if create_time_since_epoch is not None:
      self.mlmd_artifact.create_time_since_epoch = create_time_since_epoch


class Examples(DummyArtifact):
  TYPE_NAME = ops_utils.EXAMPLES_TYPE_NAME


class TransformGraph(DummyArtifact):
  TYPE_NAME = ops_utils.TRANSFORM_GRAPH_TYPE_NAME


class Model(DummyArtifact):
  TYPE_NAME = ops_utils.MODEL_TYPE_NAME


class ModelBlessing(DummyArtifact):
  TYPE_NAME = ops_utils.MODEL_BLESSING_TYPE_NAME


class ModelInfraBlessing(DummyArtifact):
  TYPE_NAME = ops_utils.MODEL_INFRA_BLESSSING_TYPE_NAME


class ModelPush(DummyArtifact):
  TYPE_NAME = ops_utils.MODEL_PUSH_TYPE_NAME


def run_resolver_op(
    op_type: Type[resolver_op.ResolverOp],
    *arg: Any,
    context: Optional[resolver_op.Context] = None,
    **kwargs: Any,
):
  op = op_type.create(**kwargs)
  if context:
    op.set_context(context)
  return op.apply(*arg)


def strict_run_resolver_op(
    op_type: Type[resolver_op.ResolverOp],
    *,
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    store: Optional[mlmd.MetadataStore] = None,
):
  """Runs ResolverOp with strict type checking."""
  if len(args) != len(op_type.arg_data_types):
    raise TypeError(
        f'len({op_type}.arg_data_types) = {len(op_type.arg_data_types)} but'
        f' got len(args) = {len(args)}'
    )
  if op_type.arg_data_types:
    for i, arg, expected_data_type in zip(
        range(len(args)), args, op_type.arg_data_types
    ):
      if expected_data_type == resolver_op.DataType.ARTIFACT_LIST:
        if not typing_utils.is_artifact_list(arg):
          raise TypeError(f'Expected ARTIFACT_LIST but arg[{i}] = {arg}')
      elif expected_data_type == resolver_op.DataType.ARTIFACT_MULTIMAP:
        if not typing_utils.is_artifact_multimap(arg):
          raise TypeError(f'Expected ARTIFACT_MULTIMAP but arg[{i}] = {arg}')
      elif expected_data_type == resolver_op.DataType.ARTIFACT_MULTIMAP_LIST:
        if not typing_utils.is_list_of_artifact_multimap(arg):
          raise TypeError(
              f'Expected ARTIFACT_MULTIMAP_LIST but arg[{i}] = {arg}'
          )
  op = op_type.create(**kwargs)
  context = resolver_op.Context(
      store=store
      if store is not None
      else mock.MagicMock(spec=mlmd.MetadataStore)
  )
  op.set_context(context)
  result = op.apply(*args)
  if op_type.return_data_type == resolver_op.DataType.ARTIFACT_LIST:
    if not typing_utils.is_homogeneous_artifact_list(result):
      raise TypeError(f'Expected ARTIFACT_LIST result but got {result}')
  elif op_type.return_data_type == resolver_op.DataType.ARTIFACT_MULTIMAP:
    if not typing_utils.is_artifact_multimap(result):
      raise TypeError(f'Expected ARTIFACT_MULTIMAP result but got {result}')
  elif op_type.return_data_type == resolver_op.DataType.ARTIFACT_MULTIMAP_LIST:
    if not typing_utils.is_list_of_artifact_multimap(result):
      raise TypeError(
          f'Expected ARTIFACT_MULTIMAP_LIST result but got {result}'
      )
  return result
