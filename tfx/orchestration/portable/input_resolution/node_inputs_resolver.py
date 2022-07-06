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
"""Module for NodeInputs.inputs.input_graph_ref resolution."""

import collections
from typing import List, TypeVar, Mapping, Tuple, Sequence, Dict, Callable, Iterable

import attr
from tfx.orchestration.portable.input_resolution import channel_resolver
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.input_resolution import input_graph_resolver
from tfx.proto.orchestration import pipeline_pb2
import tfx.types
from tfx.utils import topsort
from tfx.utils import typing_utils

import ml_metadata as mlmd

_T = TypeVar('_T')
_IRDataType = pipeline_pb2.InputGraph.DataType


def _check_cycle(nodes: Iterable[str], parents: Mapping[str, Iterable[str]]):
  """Check whether the graph has the cycle."""
  visiting = set()
  visited = set()

  def dfs(here):
    if here in visiting:
      raise exceptions.FailedPreconditionError(
          f'NodeInputs has a cycle. dependencies = {parents}')
    if here not in visited:
      visiting.add(here)
      for parent in parents[here]:
        dfs(parent)
      visiting.remove(here)
      visited.add(here)

  for node in nodes:
    dfs(node)


def _topologically_sorted_input_keys(
    input_specs: Mapping[str, pipeline_pb2.InputSpec],
    input_graphs: Mapping[str, pipeline_pb2.InputGraph]) -> List[str]:
  """Get topologically sorted input keys."""
  parents = collections.defaultdict(list)
  children = collections.defaultdict(list)

  for input_key, input_spec in input_specs.items():
    if input_spec.input_graph_ref.graph_id:
      graph_id = input_spec.input_graph_ref.graph_id
      # Should check graph_id existence, otherwise input_graphs[graph_id]
      # silently creates and returns an empty InputGraph (proto stub behavior).
      if graph_id not in input_graphs:
        raise exceptions.FailedPreconditionError(
            f'NodeInputs.inputs[{input_key}] has invalid input_graph_ref; '
            f'graph_id {graph_id} does not exist. '
            f'Available: {list(input_graphs)}')
      input_graph = input_graphs[input_spec.input_graph_ref.graph_id]
      for node_def in input_graph.nodes.values():
        if node_def.WhichOneof('kind') == 'input_node':
          parents[input_key].append(node_def.input_node.input_key)
    elif input_spec.mixed_inputs.input_keys:
      parents[input_key].extend(input_spec.mixed_inputs.input_keys)

  _check_cycle(input_specs.keys(), parents)

  for me, parents_of_me in parents.items():
    for parent in parents_of_me:
      children[parent].append(me)

  topsorted_layers = topsort.topsorted_layers(
      list(input_specs.keys()),
      get_node_id_fn=lambda x: x,
      get_parent_nodes=parents.__getitem__,
      get_child_nodes=children.__getitem__)

  result = []
  for layer in topsorted_layers:
    for input_key in layer:
      result.append(input_key)
  return result


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _CompositeKey:
  """An immutable key of multiple dimension values (str -> int)."""
  data: Mapping[str, int]

  @property
  def dims(self) -> Sequence[str]:
    return tuple(sorted(self.data.keys()))

  def partial(self, dims: Sequence[str]) -> Sequence[int]:
    return tuple(self.data[d] for d in dims)

  def __bool__(self) -> bool:
    return bool(self.data)

  def __or__(self, other) -> '_CompositeKey':
    if not isinstance(other, _CompositeKey):
      return NotImplemented  # pytype: disable=bad-return-type
    common_dims = list(set(self.data) & set(other.data))
    if self.partial(common_dims) != other.partial(common_dims):
      raise ValueError(
          f'Cannot merge {self.data} and {other.data} on {common_dims}')
    return _CompositeKey({**self.data, **other.data})


_EMPTY = _CompositeKey({})


def _group_by(
    values: Sequence[Tuple[_CompositeKey, _T]],
    dims: Sequence[str],
) -> Mapping[Sequence[int], Sequence[Tuple[_CompositeKey, _T]]]:
  result = collections.defaultdict(list)
  for composite_key, value in values:
    result[composite_key.partial(dims)].append((composite_key, value))
  return result


def _inner_join(
    lhs: Sequence[Tuple[_CompositeKey, _T]],
    rhs: Sequence[Tuple[_CompositeKey, _T]],
    join_dims: Sequence[str],
    merge_fn: Callable[[_T, _T], _T],
) -> Sequence[Tuple[_CompositeKey, _T]]:
  """Inner-join values by the composite key.

  Example:
    inner_join(
        [
            ({x=1, y=1}, 'xy-11'),
            ({x=1, y=2}, 'xy-12'),
            ({x=2, y=1}, 'xy-21'),
            ({x=2, y=2}, 'xy-22'),
        ],
        [
            ({x=1, z=1}, 'xz-11'),
            ({x=1, z=2}, 'xz-12'),
            ({x=2, z=1}, 'xz-21'),
            ({x=2, z=2}, 'xz-22'),
        ],
        join_dims=['x'],
        merge_fn=lambda left, right: f'{left}_{right}'
    ) == [
        ({x=1, y=1, z=1}, 'xy-11_xz-11'),
        ({x=1, y=1, z=2}, 'xy-11_xz-12'),
        ({x=1, y=2, z=1}, 'xy-12_xz-11'),
        ({x=1, y=2, z=2}, 'xy-12_xz-12'),
        ({x=2, y=1, z=1}, 'xy-21_xz-21'),
        ({x=2, y=1, z=2}, 'xy-21_xz-22'),
        ({x=2, y=2, z=1}, 'xy-22_xz-21'),
        ({x=2, y=2, z=2}, 'xy-22_xz-22'),
    ]

  Args:
    lhs: LHS values.
    rhs: RHS values.
    join_dims: A list of dimensions to join on.
    merge_fn: A merge function that is called for each joined pair of values.

  Returns:
    A inner-joined value with merged _CompositeKey and values.
  """
  result = []
  lhs_by_dim = _group_by(lhs, join_dims)
  rhs_by_dim = _group_by(rhs, join_dims)
  for dim, sub_lhs in lhs_by_dim.items():
    if dim not in rhs_by_dim:
      continue
    sub_rhs = rhs_by_dim[dim]
    for left_key, left_value in sub_lhs:
      for right_key, right_value in sub_rhs:
        merged_key = left_key | right_key
        merged_value = merge_fn(left_value, right_value)
        result.append((merged_key, merged_value))
  return result


_Entry = Tuple[_CompositeKey, Sequence[tfx.types.Artifact]]


def _join_artifacts(
    entries_map: Mapping[str, Sequence[_Entry]],
    input_keys: Sequence[str],
) -> Sequence[Tuple[_CompositeKey, typing_utils.ArtifactMultiMap]]:
  """Materialize entries map into actual List[ArtifactMultiMap] to be used."""
  accumulated = [(_EMPTY, {})]
  accumulated_dims = set()

  for input_key in input_keys:
    # Example:
    # entries == [
    #   (CompositeKey({'a': 1}), [Artifact(1)]),
    #   (CompositeKey({'a': 2}), [Artifact(2)]),
    # ]
    entries = entries_map[input_key]
    if not entries:
      return []

    # Example:
    # new_dims == ['a']
    new_dims = entries[0][0].dims
    common_dims = accumulated_dims & set(new_dims)
    accumulated_dims.update(new_dims)

    # Example:
    # wrapped_entries == [
    #   (CompositeKey({'a': 1}), {'x': [Artifact(1)]}),
    #   (CompositeKey({'a': 2}), {'x': [Artifact(2)]}),
    # ]
    wrapped_entries = [
        (composite_key, {input_key: artifacts})
        for composite_key, artifacts in entries]
    accumulated = _inner_join(
        accumulated, wrapped_entries,
        join_dims=list(common_dims),
        merge_fn=lambda left, right: {**left, **right})

  return accumulated


def _resolve_input_graph_ref(
    store: mlmd.MetadataStore,
    node_inputs: pipeline_pb2.NodeInputs,
    input_key: str,
    resolved: Dict[str, List[_Entry]],
) -> None:
  """Resolves an `InputGraphRef` and put resolved result into `result`.

  This also resolves other `node_inputs.inputs` that share the same graph source
  (i.e. `InputGraphRef` with the same `graph_id`).

  Args:
    store: A `MetadataStore` instance.
    node_inputs: A `NodeInputs` proto.
    input_key: A target input key whose corresponding `InputSpec` has an
        `InputGraphRef`.
    resolved: A dict that contains the already resolved inputs, and the resolved
        result would be written to from this function.
  """
  graph_id = node_inputs.inputs[input_key].input_graph_ref.graph_id
  input_graph = node_inputs.input_graphs[graph_id]
  graph_output_type = (
      input_graph.nodes[input_graph.result_node].output_data_type)
  # We will resolve all node_inputs.inputs at once that has the `InputGraphRef`
  # for the same graph (== `graph_id`).
  same_graph_inputs = {
      key: input_spec.input_graph_ref
      for key, input_spec in node_inputs.inputs.items()
      if input_spec.input_graph_ref.graph_id == graph_id
  }

  graph_fn, graph_input_keys = input_graph_resolver.build_graph_fn(
      store, node_inputs.input_graphs[graph_id])
  for composite_key, input_dict in _join_artifacts(resolved, graph_input_keys):
    result = graph_fn(input_dict)
    if graph_output_type == _IRDataType.ARTIFACT_LIST:
      # result == [Artifact()]
      resolved[input_key].append((composite_key, result))
    elif graph_output_type == _IRDataType.ARTIFACT_MULTIMAP:
      # result == {'x': [Artifact()], 'y': [Artifact()]}
      for each_input_key, input_graph_ref in same_graph_inputs.items():
        resolved[each_input_key].append(
            (composite_key, result[input_graph_ref.key]))
    elif graph_output_type == _IRDataType.ARTIFACT_MULTIMAP_LIST:
      # result == [{'x': [Artifact()]}, {'x': [Artifact()]}]
      for index, each_result in enumerate(result):
        new_composite_key = composite_key | _CompositeKey({graph_id: index})
        for each_input_key, input_graph_ref in same_graph_inputs.items():
          resolved[each_input_key].append(
              (new_composite_key, each_result[input_graph_ref.key]))


def _resolve_mixed_inputs(
    node_inputs: pipeline_pb2.NodeInputs,
    input_key: str,
    resolved: Dict[str, List[_Entry]],
) -> None:
  """Resolves an InputSpec.Mixed."""
  mixed_inputs = node_inputs.inputs[input_key].mixed_inputs
  result = []
  for composite_key, input_dict in _join_artifacts(
      resolved, mixed_inputs.input_keys):
    if mixed_inputs.method == pipeline_pb2.InputSpec.Mixed.Method.UNION:
      artifacts_by_id = {}
      for sub_key in mixed_inputs.input_keys:
        artifacts_by_id.update({
            artifact.id: artifact for artifact in input_dict[sub_key]
        })
      artifacts = list(artifacts_by_id.values())
    elif mixed_inputs.method == pipeline_pb2.InputSpec.Mixed.Method.CONCAT:
      artifacts = []
      for sub_key in mixed_inputs.input_keys:
        artifacts.extend(input_dict[sub_key])
    elif mixed_inputs.method == pipeline_pb2.InputSpec.Mixed.Method.COALESCE:
      for sub_key in mixed_inputs.input_keys:
        if input_dict[sub_key]:
          artifacts = input_dict[sub_key]
          break
      else:
        artifacts = []
    result.append((composite_key, artifacts))

  resolved[input_key] = result


def resolve(
    store: mlmd.MetadataStore,
    node_inputs: pipeline_pb2.NodeInputs,
) -> List[typing_utils.ArtifactMultiMap]:
  """Resolve a NodeInputs."""
  resolved: Dict[str, List[_Entry]] = collections.defaultdict(list)

  for input_key in _topologically_sorted_input_keys(
      node_inputs.inputs, node_inputs.input_graphs):
    # This input_key may have been already resolved while resolving for another
    # input key.
    if input_key in resolved:
      continue

    input_spec = node_inputs.inputs[input_key]
    if input_spec.channels:
      artifacts = channel_resolver.resolve_union_channels(
          store, input_spec.channels)
      resolved[input_key] = [(_EMPTY, artifacts)]
      continue

    if input_spec.input_graph_ref.graph_id:
      _resolve_input_graph_ref(store, node_inputs, input_key, resolved)
      continue

    if input_spec.mixed_inputs.input_keys:
      _resolve_mixed_inputs(node_inputs, input_key, resolved)
      continue

    raise exceptions.FailedPreconditionError(
        'Exactly one of InputSpec.channels, InputSpec.input_graph_ref, or '
        'InputSpec.mixed_inputs should be set.')

  result_with_key = _join_artifacts(resolved, list(node_inputs.inputs.keys()))
  return [t[1] for t in result_with_key]
