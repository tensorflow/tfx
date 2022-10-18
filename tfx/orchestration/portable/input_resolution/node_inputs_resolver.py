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
"""Module for NodeInputs resolution.

Notably, NodeInputs resolution uses Partition object to populate a list of
artifact maps. This helps us resolve each InputSpec independently, but still
get a intended zipped or cartesian product result, depending on the partition.

For example, consider the pipeline:

    inputs = my_resolver_function()
    with ForEach(inputs) as each_input:
      my_component = MyComponent(
          x=each_result['x'],
          y=each_result['y'])

Then the corresponding NodeInputs IR for `my_component` looks like:

    inputs {
      key: "x"
      value { // InputSpec
        input_graph_ref {
          graph_id: "graph_1"
          key: "x"
        }
      }
    }
    inputs {
      key: "y"
      value {
        input_graph_ref {
          graph_id: "graph_1"
          key: "y"
        }
      }
    }

We use the graph_id as a partition dimension, so that the resolved artifacts
with the same partition can be grouped into the same input dict:

    # `resolved` contains resolved InputSpec per channel.
    resolved == {
        'x': [
            ({'graph_1': 1}, [a1]),
              ^^^^^^^^^^^^^ --- partition
            ({'graph_1': 2}, [a2]),
            ({'graph_1': 3}, [a3]),
        ],
        'y': [
            ({'graph_1': 1}, [b1]),
            ({'graph_1': 2}, [b2]),
            ({'graph_1': 3}, [b3]),
        ]
    }
    # Joined result
    result == [
        {'x': [a1], 'y': [b1]},
        {'x': [a2], 'y': [b2]},
        {'x': [a3], 'y': [b3]},
    ]

Note that `resolved` is the intermediate data structure with partition
information, and `result` is the final rendered result from the `resolved`.
This intermediate partition data makes the implementation much simpler.

Now let's think about a more complicated pipeline:

  inputs = my_resolver_function()
  with ForEach(inputs) as input_1:
    with ForEach(inputs) as input_2:
      with ForEach(inputs) as input_3:
        x1 = union(input_1['x'], input_2['x'])
        x2 = union(input_1['x'], input_3['x'])
        my_component = MyComponent(x1=x1, x2=x2)

In order to do cartesian product, each ForEach yields different partition
dimension (i.e. different graph_id). Let's assume each for loop produces a list
of length two. We would expect total 8 dict for the final results.

For `input_{1,2,3}`, their partitioned resolution would looks like this:

    resolved == {
        'input_1': [
            ({'graph_1': 1}, [a1]),
            ({'graph_1': 2}, [a2]),
        ],
        'input_2': [
            ({'graph_2': 1}, [b1]),
            ({'graph_2': 2}, [b2]),
        ],
        'input_3': [
            ({'graph_3': 1}, [c1]),
            ({'graph_3': 2}, [c2]),
        ],
        ...,
    }

The actual input keys x1 and x2, are assembled from the resolved inputs:

    resolved == {
        ...,
        'x1': [
            ({'graph_1': 1, 'graph_2': 1}, [a1, b1]),
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ --- partition
            ({'graph_1': 1, 'graph_2': 2}, [a1, b2]),
            ({'graph_1': 2, 'graph_2': 1}, [a2, b1]),
            ({'graph_1': 2, 'graph_2': 2}, [a2, b2]),
        ],
        'x2': [
            ({'graph_1': 1, 'graph_3': 1}, [a1, c1]),
            ({'graph_1': 1, 'graph_3': 2}, [a1, c2]),
            ({'graph_1': 2, 'graph_3': 1}, [a2, c1]),
            ({'graph_1': 2, 'graph_3': 2}, [a2, c2]),
        ]
    }

With such `resolved`, the final list of artifact map would be rendered as:

    result == [
        {'x1': [a1, b1], 'x2': [a1, c1]},
        {'x1': [a1, b1], 'x2': [a1, c2]},
        {'x1': [a1, b2], 'x2': [a1, c1]},
        {'x1': [a1, b2], 'x2': [a1, c2]},
        {'x1': [a2, b1], 'x2': [a2, c1]},
        {'x1': [a2, b1], 'x2': [a2, c2]},
        {'x1': [a2, b2], 'x2': [a2, c1]},
        {'x1': [a2, b2], 'x2': [a2, c2]},
    ]

Note that each dict has exactly one kind of a1 or a2 (not mixed), which is the
correct intention from the pipeline DSL.

Some InputSpec kind (e.g. Mixed or InputGraphRef) can refer to another InputSpec
as an input. This is because some BaseChannel often wraps another BaseChannel(s)
(e.g. UnionChannel or ResolvedChannel). What intermediate data structure with
partition information (`resolved` in the above example) really help us is that
it generalizes InputSpec as taking a partitioned input (another input spec) and
produces a partitioned output (intermediate data -> intermediate data). Final
result can easily be deduced from this intermediate data.
"""

import collections
from typing import List, TypeVar, Mapping, Tuple, Sequence, Dict, Iterable

from tfx import types
from tfx.dsl.compiler import placeholder_utils
from tfx.orchestration import metadata
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable.input_resolution import channel_resolver
from tfx.orchestration.portable.input_resolution import exceptions
from tfx.orchestration.portable.input_resolution import input_graph_resolver
from tfx.orchestration.portable.input_resolution import partition_utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import topsort
from tfx.utils import typing_utils

_T = TypeVar('_T')
_DataType = pipeline_pb2.InputGraph.DataType


def _check_cycle(
    nodes: Iterable[str], dependencies: Mapping[str, Iterable[str]]):
  """Check whether the graph has the cycle."""
  visiting = set()
  visited = set()

  def dfs(here):
    if here in visiting:
      raise exceptions.FailedPreconditionError(
          f'NodeInputs has a cycle. dependencies = {dependencies}')
    if here not in visited:
      visiting.add(here)
      for there in dependencies[here]:
        dfs(there)
      visiting.remove(here)
      visited.add(here)

  for node in nodes:
    dfs(node)


def _get_dependencies(
    input_specs: Mapping[str, pipeline_pb2.InputSpec],
    input_graphs: Mapping[str, pipeline_pb2.InputGraph],
) -> Dict[str, List[str]]:
  """Get dependencies of input_specs."""
  result = collections.defaultdict(list)
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
          result[input_key].append(node_def.input_node.input_key)
    elif input_spec.mixed_inputs.input_keys:
      result[input_key].extend(input_spec.mixed_inputs.input_keys)
  return result


def _get_reverse_dependencies(
    dependencies: Mapping[str, Iterable[str]]) -> Dict[str, List[str]]:
  result = collections.defaultdict(list)
  for me, deps in dependencies.items():
    for dep in deps:
      result[dep].append(me)
  return result


def _topologically_sorted_input_keys(
    input_specs: Mapping[str, pipeline_pb2.InputSpec],
    input_graphs: Mapping[str, pipeline_pb2.InputGraph]) -> List[str]:
  """Get topologically sorted input keys."""
  parents = _get_dependencies(input_specs, input_graphs)
  children = _get_reverse_dependencies(parents)
  _check_cycle(input_specs.keys(), parents)

  topsorted_layers = topsort.topsorted_layers(
      list(input_specs.keys()),
      get_node_id_fn=lambda x: x,
      get_parent_nodes=parents.__getitem__,
      get_child_nodes=children.__getitem__)

  result = []
  for layer in topsorted_layers:
    result.extend(layer)
  return result


_Entry = Tuple[partition_utils.Partition, List[types.Artifact]]


def _join_artifacts(
    entries_map: Mapping[str, Sequence[_Entry]],
    input_keys: Sequence[str],
) -> Sequence[Tuple[partition_utils.Partition, typing_utils.ArtifactMultiDict]]:
  """Materialize entries map into actual List[ArtifactMultiDict] to be used."""
  accumulated = [(partition_utils.NO_PARTITION, {})]

  for input_key in input_keys:
    # Example:
    # entries == [
    #   ({'a': 1}, [Artifact(1)]),
    #   ({'a': 2}, [Artifact(2)]),
    # ]
    entries = entries_map[input_key]
    if not entries:
      return []

    # Example:
    # wrapped_entries == [
    #   ({'a': 1}, {'x': [Artifact(1)]}),
    #   ({'a': 2}, {'x': [Artifact(2)]}),
    # ]
    wrapped_entries = [
        (partition, {input_key: artifacts})
        for partition, artifacts in entries]
    accumulated = partition_utils.join(
        accumulated, wrapped_entries,
        merge_fn=lambda left, right: {**left, **right})

  return accumulated


def _resolve_input_graph_ref(
    mlmd_handle: metadata.Metadata,
    node_inputs: pipeline_pb2.NodeInputs,
    input_key: str,
    resolved: Dict[str, List[_Entry]],
) -> None:
  """Resolves an `InputGraphRef` and put resolved result into `resolved`.

  This also resolves other `node_inputs.inputs` that share the same graph source
  (i.e. `InputGraphRef` with the same `graph_id`).

  Args:
    mlmd_handle: A `Metadata` instance.
    node_inputs: A `NodeInputs` proto.
    input_key: A target input key whose corresponding `InputSpec` has an
        `InputGraphRef`.
    resolved: A dict that contains the already resolved inputs, and to which the
        resolved result would be written from this function.
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
      mlmd_handle, node_inputs.input_graphs[graph_id])
  for partition, input_dict in _join_artifacts(resolved, graph_input_keys):
    result = graph_fn(input_dict)
    if graph_output_type == _DataType.ARTIFACT_LIST:
      # result == [Artifact()]
      resolved[input_key].append((partition, result))
    elif graph_output_type == _DataType.ARTIFACT_MULTIMAP:
      # result == {'x': [Artifact()], 'y': [Artifact()]}
      for each_input_key, input_graph_ref in same_graph_inputs.items():
        resolved[each_input_key].append(
            (partition, result[input_graph_ref.key]))
    elif graph_output_type == _DataType.ARTIFACT_MULTIMAP_LIST:
      # result == [{'x': [Artifact()]}, {'x': [Artifact()]}]
      for index, each_result in enumerate(result):
        new_partition = partition | {graph_id: index}
        for each_input_key, input_graph_ref in same_graph_inputs.items():
          resolved[each_input_key].append(
              (new_partition, each_result[input_graph_ref.key]))


def _resolve_mixed_inputs(
    node_inputs: pipeline_pb2.NodeInputs,
    input_key: str,
    resolved: Dict[str, List[_Entry]],
) -> None:
  """Resolves an InputSpec.Mixed."""
  mixed_inputs = node_inputs.inputs[input_key].mixed_inputs
  result = []
  for partition, input_dict in _join_artifacts(
      resolved, mixed_inputs.input_keys):
    if mixed_inputs.method == pipeline_pb2.InputSpec.Mixed.Method.UNION:
      artifacts_by_id = {}
      for sub_key in mixed_inputs.input_keys:
        artifacts_by_id.update({
            artifact.id: artifact for artifact in input_dict[sub_key]
        })
      artifacts = list(artifacts_by_id.values())
    result.append((partition, artifacts))

  resolved[input_key] = result


def _filter_conditionals(
    artifact_maps: List[typing_utils.ArtifactMultiDict],
    conditionals: Mapping[str, pipeline_pb2.NodeInputs.Conditional],
) -> List[typing_utils.ArtifactMultiDict]:
  """Filter artifact maps by conditionals."""
  result = []
  for artifact_map in artifact_maps:
    context = placeholder_utils.ResolutionContext(
        exec_info=data_types.ExecutionInfo(input_dict=artifact_map))
    for cond_id, cond in conditionals.items():
      ok = placeholder_utils.resolve_placeholder_expression(
          cond.placeholder_expression, context)
      if not isinstance(ok, bool):
        raise exceptions.FailedPreconditionError(
            f'Invalid conditional expression for {cond_id}; '
            f'Expected boolean type but got {type(ok)} type.')
      if not ok:
        break
    else:
      result.append(artifact_map)
  return result


def resolve(
    mlmd_handle: metadata.Metadata,
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
          mlmd_handle, input_spec.channels)
      resolved[input_key] = [(partition_utils.NO_PARTITION, artifacts)]
    elif input_spec.input_graph_ref.graph_id:
      _resolve_input_graph_ref(mlmd_handle, node_inputs, input_key, resolved)
    elif input_spec.mixed_inputs.input_keys:
      _resolve_mixed_inputs(node_inputs, input_key, resolved)
    else:
      raise exceptions.FailedPreconditionError(
          'Exactly one of InputSpec.channels, InputSpec.input_graph_ref, or '
          'InputSpec.mixed_inputs should be set.')

    if input_spec.min_count:
      for _, artifacts in resolved[input_key]:
        if len(artifacts) < input_spec.min_count:
          raise exceptions.FailedPreconditionError(
              'InputSpec min_count violation; '
              f'inputs[{input_key}] has min_count = {input_spec.min_count} '
              f'but only got {len(artifacts)} artifacts. '
              f'(Artifact IDs: {[a.id for a in artifacts]})')

  visible_keys = [
      k for k, input_spec in node_inputs.inputs.items()
      if not input_spec.hidden
  ]
  result = [
      artifact_map for composite_key, artifact_map
      in _join_artifacts(resolved, visible_keys)
  ]

  if node_inputs.conditionals:
    result = _filter_conditionals(result, node_inputs.conditionals)

  return result
