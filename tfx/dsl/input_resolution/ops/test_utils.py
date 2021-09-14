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
from typing import Type, Mapping, Any, Optional

from tfx import types
from tfx.dsl.input_resolution import resolver_function
from tfx.dsl.input_resolution import resolver_op
from tfx.utils import typing_utils


class DummyArtifact(types.Artifact):
  TYPE_NAME = 'DummyArtifact'


def _instantiate_op(
    op_type: Type[resolver_op.ResolverOp],
    kwargs: Mapping[str, Any],
) -> resolver_op.ResolverOp:
  if (isinstance(op_type, type) and
      issubclass(op_type, resolver_op.ResolverOp)):
    return op_type.create(**kwargs)
  else:
    raise TypeError(f'Invalid op_type: {op_type}.')


def create_dummy_artifact(
    id: Optional[str] = None,  # pylint: disable=redefined-builtin
    uri: Optional[str] = None,
) -> types.Artifact:
  result = DummyArtifact()
  if id:
    result.id = id
  if uri:
    result.uri = uri
  return result


def run_resolver_function(
    f: resolver_function.ResolverFunction,
    input_dict: typing_utils.ArtifactMultiMap,
):
  """Run a resolver function with given input dict."""
  input_node = resolver_op.OpNode.INPUT_NODE
  result_node = f.trace(input_node)
  outputs_by_node = {}
  outputs_by_node[input_node] = input_dict

  def memoized_run(node: resolver_op.OpNode):
    if node in outputs_by_node:
      return outputs_by_node[node]
    arg = memoized_run(node.arg)
    op = _instantiate_op(node.op_type, node.kwargs)
    result = op.apply(arg)
    outputs_by_node[node] = result
    return result

  return memoized_run(result_node)
