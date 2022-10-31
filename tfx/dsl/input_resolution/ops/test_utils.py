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
from typing import Type, Any, Optional

from tfx import types
from tfx.dsl.input_resolution import resolver_op
from tfx.types import artifact


class DummyArtifact(types.Artifact):
  """A dummy Artifact used for testing."""

  TYPE_NAME = 'DummyArtifact'

  PROPERTIES = {
      'span': artifact.Property(type=artifact.PropertyType.INT),
      'version': artifact.Property(type=artifact.PropertyType.INT),
  }

  # pylint: disable=redefined-builtin
  def __init__(self,
               id: Optional[str] = None,
               uri: Optional[str] = None,
               create_time_since_epoch: Optional[int] = None):
    super().__init__()
    if id is not None:
      self.id = id
    if uri is not None:
      self.uri = uri
    if create_time_since_epoch is not None:
      self.mlmd_artifact.create_time_since_epoch = create_time_since_epoch


def run_resolver_op(op_type: Type[resolver_op.ResolverOp],
                    *arg: Any,
                    context: Optional[resolver_op.Context] = None,
                    **kwargs: Any):
  op = op_type.create(**kwargs)
  if context:
    op.set_context(context)
  return op.apply(*arg)
