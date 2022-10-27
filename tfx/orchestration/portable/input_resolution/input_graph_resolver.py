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
"""Module for pipeline_pb2.InputGraph resolution.

`InputGraph` is not resolved directly, but is converted to a function that can
be evaluated later with lazy inputs. This is because `InputNode` of the
`InputGraph` should be injected from the outside.

Consider the following example:

    with ForEach(xs) as x:
      y = resolver_function(x)
      my_component = MyComponent(y=y)

then the `resolver_function` (corresponds to the `InputGraph`) should be invoked
multiple times with different inputs `x`.
"""
import collections
import dataclasses
import functools
from typing import Union, Sequence, Mapping, Tuple, List, Iterable, Callable

from tfx import types
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution import resolver_op
from tfx.dsl.input_resolution.ops import ops
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import topsort
from tfx.utils import typing_utils

_Data = Union[
    Sequence[types.Artifact],
    typing_utils.ArtifactMultiMap,
    Sequence[typing_utils.ArtifactMultiMap],
]
_GraphFn = Callable[[Mapping[str, _Data]], _Data]


@dataclasses.dataclass
class _Context:
  mlmd_handle: metadata.Metadata
  input_graph: pipeline_pb2.InputGraph


def _topologically_sorted_node_ids(
    input_graph: pipeline_pb2.InputGraph) -> Iterable[str]:
  """Get topologically sorted InputGraph.nodes ids."""
  parents = collections.defaultdict(list)
  children = collections.defaultdict(list)

  for node_id, node_def in input_graph.nodes.items():
    kind = node_def.WhichOneof('kind')
    if kind == 'op_node':
      for arg in [*node_def.op_node.args, *node_def.op_node.kwargs.values()]:
        if arg.WhichOneof('kind') == 'node_id':
          parents[node_id].append(arg.node_id)
    elif kind == 'dict_node':
      parents[node_id].extend(node_def.dict_node.node_ids.values())
    elif kind == 'input_node':
      continue  # InputNode does not have further dependency within the graph.
    else:
      raise exceptions.UnimplementedError(
          f'InputGraph node {node_id} has unknown kind {kind}.')

  for me, parents_of_me in parents.items():
    for parent in parents_of_me:
      children[parent].append(me)

  try:
    topsorted_layers = topsort.topsorted_layers(
        list(input_graph.nodes.keys()),
        get_node_id_fn=lambda x: x,
        get_parent_nodes=parents.__getitem__,
        get_child_nodes=children.__getitem__)
  except topsort.InvalidDAGError as e:
    raise exceptions.FailedPreconditionError(
        f'InputGraph has a cycle. parents = {parents}.') from e

  for layer in topsorted_layers:
    for node_id in layer:
      yield node_id


def _unwrap_arg(
    arg: pipeline_pb2.InputGraph.OpNode.Arg, data: Mapping[str, _Data]):
  """Unwraps InputGraph.OpNode.Arg."""
  kind = arg.WhichOneof('kind')
  if kind == 'node_id':
    return data[arg.node_id]
  elif kind == 'value':
    if not arg.value.HasField('field_value'):
      raise exceptions.UnimplementedError(
          f'OpNode.Arg should be a static value but got {arg.value}')
    return data_types_utils.get_parsed_value(
        arg.value.field_value,
        arg.value.schema if arg.value.HasField('schema') else None)
  else:
    raise exceptions.InternalError('OpNode.Arg not set.')


def _evaluate_op_node(
    ctx: _Context,
    node_id: str,
    data: Mapping[str, _Data]) -> _Data:
  """A functional interface for `InputNode`."""
  op_node = ctx.input_graph.nodes[node_id].op_node
  args = [_unwrap_arg(v, data) for v in op_node.args]
  kwargs = {k: _unwrap_arg(v, data) for k, v in op_node.kwargs.items()}
  try:
    op_type = ops.get_by_name(op_node.op_type)
  except KeyError as e:
    try:
      # Currently ResolverStrategy is stored as a class path.
      op_type = ops.get_by_class_path(op_node.op_type)
    except ValueError:
      raise exceptions.InternalError(
          f'nodes[{node_id}] has unknown op_type {op_node.op_type}.') from e
  if issubclass(op_type, resolver_op.ResolverOp):
    op: resolver_op.ResolverOp = op_type.create(**kwargs)
    op.set_context(resolver_op.Context(store=ctx.mlmd_handle.store))
    return op.apply(*args)
  elif issubclass(op_type, resolver.ResolverStrategy):
    if len(args) != 1 or not typing_utils.is_artifact_multimap(args[0]):
      raise exceptions.FailedPreconditionError(
          f'Invalid {op_type} argument: {args!r}')
    strategy: resolver.ResolverStrategy = op_type(**kwargs)
    result = strategy.resolve_artifacts(ctx.mlmd_handle.store, args[0])
    if result is None:
      raise exceptions.InputResolutionError(f'{strategy} returned None.')
    return result
  else:
    raise exceptions.InternalError(f'Unknown op_type {op_type}.')


def _evaluate_dict_node(
    ctx: _Context, node_id: str, data: Mapping[str, _Data]) -> _Data:
  """A functional interface for `DictNode`."""
  dict_node = ctx.input_graph.nodes[node_id].dict_node
  result = {}
  for dict_key, node_id in dict_node.node_ids.items():
    result[dict_key] = data[node_id]
  return result  # pytype: disable=bad-return-type  # mapping-is-not-sequence


def _reduce_graph_fn(ctx: _Context, node_id: str, graph_fn: _GraphFn):
  """Construct a new `GraphFn` with `GraphFn` and an `InputGraph.Node`.

  If `graph_fn` has a signature of ({x, y}) -> z and a node (of `node_id`) has
  a signature of ({w, v}) -> x, then the reduced graph function (return value)
  has a signature of ({w, v, y}) -> z.

  `_reduce_graph_fn` assumes the dependent nodes of the `node_id` is not yet
  reduced into the current `graph_fn`, so nodes must be reduced in a reversed
  topologically sorted order. This ensures each node is invoked exactly once in
  a correct order.

  Args:
    ctx: Current `_Context`.
    node_id: A `InputGraph.Node` ID.
    graph_fn: An accumulated `GraphFn`.

  Returns:
    A reduced `GraphFn`.
  """
  node_def = ctx.input_graph.nodes[node_id]
  if node_def.WhichOneof('kind') == 'op_node':
    node_fn = functools.partial(_evaluate_op_node, ctx, node_id)
  elif node_def.WhichOneof('kind') == 'dict_node':
    node_fn = functools.partial(_evaluate_dict_node, ctx, node_id)
  else:
    # Skip for kind == 'input_node' as the final facade graph function take
    # cares of them.
    return graph_fn

  def new_graph_fn(data: Mapping[str, _Data]):
    try:
      output = node_fn(data)
    except (exceptions.InputResolutionError, exceptions.InputResolutionSignal):  # pylint: disable=try-except-raise
      raise
    except Exception as e:  # pylint: disable=broad-except
      raise exceptions.InternalError(
          'Internal error occurred during evaluating input_graph.nodes['
          f'{node_id}].') from e
    return graph_fn({**data, node_id: output})

  return new_graph_fn


def build_graph_fn(
    mlmd_handle: metadata.Metadata,
    input_graph: pipeline_pb2.InputGraph,
) -> Tuple[_GraphFn, List[str]]:
  """Build a functional interface for the `input_graph`.

  Returned function can be called with a dict of input artifacts that are
  specified from the `InputNode` of the `InputGraph`.

  Example:
    inputs = previously_resolved_inputs()
    graph_fn, input_keys = build_graph_fn(mlmd_handle, input_graph)
    # input_keys == ['x', 'y']
    z = graph_fn({'x': inputs['x'], 'y': inputs['y']})

  Args:
    mlmd_handle: A `Metadata` instance.
    input_graph: An `pipeline_pb2.InputGraph` proto.

  Returns:
    Tuple of (graph_fn, graph_fn_inputs).
    graph_fn_inputs is a list of input keys, where all input keys must exist in
    the input argument dict to the graph_fn.
  """
  if input_graph.result_node not in input_graph.nodes:
    raise exceptions.FailedPreconditionError(
        f'result_node {input_graph.result_node} does not exist in input_graph. '
        f'Valid node ids: {list(input_graph.nodes.keys())}')

  context = _Context(mlmd_handle=mlmd_handle, input_graph=input_graph)

  input_key_to_node_id = {}
  for node_id in input_graph.nodes:
    node_def = input_graph.nodes[node_id]
    if node_def.WhichOneof('kind') == 'input_node':
      input_key_to_node_id[node_def.input_node.input_key] = node_id

  def initial_graph_fn(data: Mapping[str, _Data]) -> _Data:
    return data[input_graph.result_node]

  graph_fn = initial_graph_fn
  for node_id in reversed(list(_topologically_sorted_node_ids(input_graph))):
    graph_fn = _reduce_graph_fn(context, node_id, graph_fn)

  def facade_fn(inputs: Mapping[str, _Data]) -> _Data:
    return graph_fn({
        node_id: inputs[input_key]
        for input_key, node_id in input_key_to_node_id.items()
    })

  return facade_fn, list(input_key_to_node_id.keys())
