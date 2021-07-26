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
"""Portable library for partial runs."""

import collections
import enum

from typing import Any, Callable, Collection, List, Mapping, MutableMapping, Optional, Set, Tuple, Union

from absl import logging
from tfx.dsl.compiler import compiler_utils
from tfx.dsl.compiler import constants
from tfx.orchestration import metadata
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from ml_metadata.proto import metadata_store_pb2
from google.protobuf import any_pb2


def filter_pipeline(
    input_pipeline: pipeline_pb2.Pipeline,
    from_nodes: Callable[[str], bool] = lambda _: True,
    to_nodes: Callable[[str], bool] = lambda _: True,
) -> Tuple[pipeline_pb2.Pipeline, Mapping[
    str, List[pipeline_pb2.InputSpec.Channel]]]:
  """Filters the Pipeline IR proto, thus enabling partial runs.

  The set of nodes included in the filtered pipeline is the set of nodes between
  from_nodes and to_nodes -- i.e., the set of nodes that are reachable by
  traversing downstream from `from_nodes` AND also reachable by traversing
  upstream from `to_nodes`. Also, if the input_pipeline contains per-node
  DeploymentConfigs, they will be filtered as well.

  Args:
    input_pipeline: A valid compiled Pipeline IR proto to be filtered.
    from_nodes: A predicate function that selects nodes by their ids. The set of
      nodes whose node_ids return True determine where the "sweep" starts from
      (see detailed description).
      Defaults to lambda _: True (i.e., select all nodes).
    to_nodes: A predicate function that selects nodes by their ids. The set of
      nodes whose node_ids return True determine where the "sweep" ends (see
      detailed description).
      Defaults to lambda _: True (i.e., select all nodes).

  Returns:
    A Tuple consisting of two elements:
    - The filtered Pipeline IR proto, with the order of nodes preserved.
    - A Mapping from node_ids of nodes that were filtered out, to the input
      channels that depend on them as producer nodes.

  Raises:
    ValueError: If input_pipeline's execution_mode is not SYNC.
    ValueError: If input_pipeline contains a sub-pipeline.
    ValueError: If input_pipeline is not topologically sorted.
  """
  _ensure_sync_pipeline(input_pipeline)
  _ensure_no_subpipeline_nodes(input_pipeline)
  _ensure_topologically_sorted(input_pipeline)

  node_map = _make_ordered_node_map(input_pipeline)
  from_node_ids = [node_id for node_id in node_map if from_nodes(node_id)]
  to_node_ids = [node_id for node_id in node_map if to_nodes(node_id)]
  node_map = _filter_node_map(node_map, from_node_ids, to_node_ids)
  node_map, excluded_direct_dependencies = _fix_nodes(node_map)
  fixed_deployment_config = _fix_deployment_config(input_pipeline, node_map)
  filtered_pipeline = _make_filtered_pipeline(input_pipeline, node_map,
                                              fixed_deployment_config)
  return filtered_pipeline, excluded_direct_dependencies


class _Direction(enum.Enum):
  UPSTREAM = 1
  DOWNSTREAM = 2


def _ensure_sync_pipeline(pipeline: pipeline_pb2.Pipeline):
  """Raises ValueError if the pipeline's execution_mode is not SYNC."""
  if pipeline.execution_mode != pipeline_pb2.Pipeline.SYNC:
    raise ValueError('Pipeline filtering is only supported for '
                     'SYNC pipelines.')


def _ensure_no_subpipeline_nodes(pipeline: pipeline_pb2.Pipeline):
  """Raises ValueError if the pipeline contains a sub-pipeline.

  If the pipeline comes from the compiler, it should already be
  flattened. This is just in case the IR proto was created in another way.

  Args:
    pipeline: The input pipeline.

  Raises:
    ValueError: If the pipeline contains a sub-pipeline.
  """
  for pipeline_or_node in pipeline.nodes:
    if pipeline_or_node.HasField('sub_pipeline'):
      raise ValueError(
          'Pipeline filtering not supported for pipelines with sub-pipelines. '
          f'sub-pipeline found: {pipeline_or_node}')


def _ensure_topologically_sorted(pipeline: pipeline_pb2.Pipeline):
  """Raises ValueError if nodes are not topologically sorted.

  If the pipeline comes from the compiler, it should already be
  topologically sorted. This is just in case the IR proto was modified or
  created in another way.

  Args:
    pipeline: The input pipeline.

  Raises:
    ValueError: If the pipeline is not topologically sorted.
  """
  # Upstream check
  visited = set()
  for pipeline_or_node in pipeline.nodes:
    node = pipeline_or_node.pipeline_node
    for upstream_node in node.upstream_nodes:
      if upstream_node not in visited:
        raise ValueError(
            'Input pipeline is not topologically sorted. '
            f'node {node.node_info.id} has upstream_node {upstream_node}, but '
            f'{upstream_node} does not appear before {node.node_info.id}')
    visited.add(node.node_info.id)
  # Downstream check
  visited.clear()
  for pipeline_or_node in reversed(pipeline.nodes):
    node = pipeline_or_node.pipeline_node
    for downstream_node in node.downstream_nodes:
      if downstream_node not in visited:
        raise ValueError(
            'Input pipeline is not topologically sorted. '
            f'node {node.node_info.id} has downstream_node {downstream_node}, '
            f'but {downstream_node} does not appear after {node.node_info.id}')
    visited.add(node.node_info.id)


def _make_ordered_node_map(
    pipeline: pipeline_pb2.Pipeline
) -> 'collections.OrderedDict[str, pipeline_pb2.PipelineNode]':
  """Prepares the Pipeline proto for DAG traversal.

  Args:
    pipeline: The input Pipeline proto, which must already be topologically
      sorted.

  Returns:
    An OrderedDict that maps node_ids to PipelineNodes.
  """
  result = collections.OrderedDict()
  for pipeline_or_node in pipeline.nodes:
    node_id = pipeline_or_node.pipeline_node.node_info.id
    result[node_id] = pipeline_or_node.pipeline_node
  return result


def _traverse(node_map: Mapping[str, pipeline_pb2.PipelineNode],
              direction: _Direction, start_nodes: Collection[str]) -> Set[str]:
  """Traverses a DAG from start_nodes, either upstream or downstream.

  Args:
    node_map: Mapping of node_id to nodes.
    direction: _Direction.UPSTREAM or _Direction.DOWNSTREAM.
    start_nodes: node_ids to start from.

  Returns:
    Set of node_ids visited by this traversal.
  """
  result = set()
  stack = []
  for start_node in start_nodes:
    # Depth-first traversal
    stack.append(start_node)
    while stack:
      current_node_id = stack.pop()
      if current_node_id in result:
        continue
      result.add(current_node_id)
      if direction == _Direction.UPSTREAM:
        stack.extend(node_map[current_node_id].upstream_nodes)
      elif direction == _Direction.DOWNSTREAM:
        stack.extend(node_map[current_node_id].downstream_nodes)
  return result


def _filter_node_map(
    node_map: 'collections.OrderedDict[str, pipeline_pb2.PipelineNode]',
    from_node_ids: Collection[str],
    to_node_ids: Collection[str],
) -> 'collections.OrderedDict[str, pipeline_pb2.PipelineNode]':
  """Returns an OrderedDict with only the nodes we want to include."""
  ancestors_of_to_nodes = _traverse(node_map, _Direction.UPSTREAM, to_node_ids)
  descendents_of_from_nodes = _traverse(node_map, _Direction.DOWNSTREAM,
                                        from_node_ids)
  nodes_to_keep = ancestors_of_to_nodes.intersection(descendents_of_from_nodes)
  result = collections.OrderedDict()
  for node_id, node in node_map.items():
    if node_id in nodes_to_keep:
      result[node_id] = node
  return result


def _remove_dangling_downstream_nodes(
    node: pipeline_pb2.PipelineNode,
    node_ids_to_keep: Collection[str]) -> pipeline_pb2.PipelineNode:
  """Removes node.downstream_nodes that have been filtered out."""
  # Using a loop instead of set intersection to ensure the same order.
  downstream_nodes_to_keep = [
      downstream_node for downstream_node in node.downstream_nodes
      if downstream_node in node_ids_to_keep
  ]
  if len(downstream_nodes_to_keep) == len(node.downstream_nodes):
    return node
  result = pipeline_pb2.PipelineNode()
  result.CopyFrom(node)
  result.downstream_nodes[:] = downstream_nodes_to_keep
  return result


def _handle_missing_inputs(
    node: pipeline_pb2.PipelineNode,
    node_ids_to_keep: Collection[str],
) -> Tuple[pipeline_pb2.PipelineNode, Mapping[
    str, List[pipeline_pb2.InputSpec.Channel]]]:
  """Handles missing inputs.

  Args:
    node: The Pipeline node to check for missing inputs.
    node_ids_to_keep: The node_ids that are not filtered out.

  Returns:
    A Tuple containing two elements:
    - A copy of the Pipeline node with some nodes removed,
    - A Mapping from removed node_ids to a list of input channels that use it as
      the producer node.
  """
  upstream_nodes_removed = set()
  upstream_nodes_to_keep = []
  for upstream_node in node.upstream_nodes:
    if upstream_node in node_ids_to_keep:
      upstream_nodes_to_keep.append(upstream_node)
    else:
      upstream_nodes_removed.add(upstream_node)

  if not upstream_nodes_removed:
    return node, {}  # No parent missing, no need to change anything.

  excluded_direct_deps = collections.defaultdict(list)
  new_node = pipeline_pb2.PipelineNode()
  new_node.CopyFrom(node)
  for input_spec in new_node.inputs.inputs.values():
    for channel in input_spec.channels:
      if channel.producer_node_query.id in upstream_nodes_removed:
        excluded_direct_deps[channel.producer_node_query.id].append(channel)
  new_node.upstream_nodes[:] = upstream_nodes_to_keep
  return new_node, excluded_direct_deps


def _fix_nodes(
    node_map: 'collections.OrderedDict[str, pipeline_pb2.PipelineNode]',
) -> Tuple['collections.OrderedDict[str, pipeline_pb2.PipelineNode]', Mapping[
    str, List[pipeline_pb2.InputSpec.Channel]]]:
  """Removes dangling references and handle missing inputs."""
  fixed_nodes = collections.OrderedDict()
  merged_excluded_direct_deps = collections.defaultdict(list)
  for node_id in node_map:
    new_node = _remove_dangling_downstream_nodes(
        node=node_map[node_id], node_ids_to_keep=node_map.keys())
    new_node, excluded_direct_deps = _handle_missing_inputs(
        node=new_node, node_ids_to_keep=node_map.keys())
    fixed_nodes[node_id] = new_node
    for inner_node_id, channel_list in excluded_direct_deps.items():
      merged_excluded_direct_deps[inner_node_id] += channel_list
  return fixed_nodes, merged_excluded_direct_deps


def _fix_deployment_config(
    input_pipeline: pipeline_pb2.Pipeline,
    node_ids_to_keep: Collection[str]) -> Union[any_pb2.Any, None]:
  """Filters per-node deployment configs.

  Cast deployment configs from Any proto to IntermediateDeploymentConfig.
  Take all three per-node fields and filter out the nodes using
  node_ids_to_keep. This works because those fields don't contain references to
  other nodes.

  Args:
    input_pipeline: The input Pipeline IR proto.
    node_ids_to_keep: Set of node_ids to keep.

  Returns:
    If the deployment_config field is set in the input_pipeline, this would
    output the deployment config with filtered per-node configs, then cast into
    an Any proto. If the deployment_config field is unset in the input_pipeline,
    then this function would return None.
  """
  if not input_pipeline.HasField('deployment_config'):
    return None

  deployment_config = pipeline_pb2.IntermediateDeploymentConfig()
  input_pipeline.deployment_config.Unpack(deployment_config)

  def _fix_per_node_config(config_map: MutableMapping[str, Any]):
    for node_id in list(config_map.keys()):  # make a temporary copy of the keys
      if node_id not in node_ids_to_keep:
        del config_map[node_id]

  _fix_per_node_config(deployment_config.executor_specs)
  _fix_per_node_config(deployment_config.custom_driver_specs)
  _fix_per_node_config(deployment_config.node_level_platform_configs)

  result = any_pb2.Any()
  result.Pack(deployment_config)
  return result


def _make_filtered_pipeline(
    input_pipeline: pipeline_pb2.Pipeline,
    node_map: 'collections.OrderedDict[str, pipeline_pb2.PipelineNode]',
    fixed_deployment_config: Optional[any_pb2.Any] = None
) -> pipeline_pb2.Pipeline:
  """Pieces different parts of the Pipeline proto together."""
  result = pipeline_pb2.Pipeline()
  result.CopyFrom(input_pipeline)
  del result.nodes[:]
  result.nodes.extend(
      pipeline_pb2.Pipeline.PipelineOrNode(pipeline_node=node_map[node_id])
      for node_id in node_map)
  if fixed_deployment_config:
    result.deployment_config.CopyFrom(fixed_deployment_config)
  return result


def reuse_node_outputs(metadata_handler: metadata.Metadata, pipeline_name: str,
                       node_id: str, old_run_id: str, new_run_id: str):
  """Reuse the output Artifacts of a pipeline node from a previous pipeline run.

  This copies the latest successful execution associated with the pipeline,
  the old pipeline run id, and node_id, and publishes it as a new cache
  execution, but associated with the new pipeline run id. This makes the output
  artifacts from that execution available for the new pipeline run, which is
  necessary to make partial run work.

  Args:
    metadata_handler: A handler to access MLMD store.
    pipeline_name: The name of the pipeline.
    node_id: The node id.
    old_run_id: The pipeline_run_id where the output artifacts were produced.
    new_run_id: The pipeline_run_id to make the output artifacts available in.
  """
  artifact_recycler = _ArtifactRecycler(metadata_handler, pipeline_name,
                                        old_run_id)
  artifact_recycler.reuse_node_outputs(node_id, new_run_id)


class _ArtifactRecycler:
  """Allows previously-generated Artifacts to be used in a new pipeline run.

  By implementing this in a class (instead of a function), we reduce the
  number of MLMD reads when reusing the outputs of multiple nodes in the same
  pipeline run.
  """

  def __init__(self, metadata_handler: metadata.Metadata, pipeline_name: str,
               old_run_id: str):
    self._mlmd = metadata_handler
    self._pipeline_name = pipeline_name
    self._pipeline_context = self._get_pipeline_context()
    self._old_pipeline_run_context = self._get_pipeline_run_context(old_run_id)
    self._pipeline_run_type_id = self._old_pipeline_run_context.type_id

  def _get_pipeline_context(self) -> metadata_store_pb2.Context:
    result = self._mlmd.store.get_context_by_type_and_name(
        type_name=constants.PIPELINE_CONTEXT_TYPE_NAME,
        context_name=self._pipeline_name)
    if result is None:
      raise LookupError(f'pipeline {self._pipeline_name} not found in MLMD.')
    return result

  def _get_pipeline_run_context(
      self, pipeline_run_id: str) -> metadata_store_pb2.Context:
    result = self._mlmd.store.get_context_by_type_and_name(
        type_name=constants.PIPELINE_RUN_CONTEXT_TYPE_NAME,
        context_name=pipeline_run_id)
    if result is None:
      raise LookupError(f'pipeline_run_id {pipeline_run_id} not found in MLMD.')
    return result

  def _get_node_context(self, node_id: str) -> metadata_store_pb2.Context:
    node_context_name = compiler_utils.node_context_name(
        self._pipeline_name, node_id)
    result = self._mlmd.store.get_context_by_type_and_name(
        type_name=constants.NODE_CONTEXT_TYPE_NAME,
        context_name=node_context_name)
    if result is None:
      raise LookupError(f'node context {node_context_name} not found in MLMD.')
    return result

  def _get_previous_execution(
      self, node_id: str) -> List[metadata_store_pb2.Execution]:
    """Gets all successful Executions of a given node.

    Args:
      node_id: The node whose Executions to query.

    Returns:
      The latest successful execution for that node.

    Raises:
      LookupError: If no successful Execution was found.
    """
    node_context = self._get_node_context(node_id)
    all_associated_executions = (
        execution_lib.get_executions_associated_with_all_contexts(
            self._mlmd,
            contexts=[
                node_context, self._old_pipeline_run_context,
                self._pipeline_context
            ]))
    prev_successful_executions = [
        e for e in all_associated_executions
        if execution_lib.is_execution_successful(e)
    ]
    if not prev_successful_executions:
      raise LookupError(
          f'No previous successful executions found for node_id {node_id} in '
          f'pipeline_run {self._old_pipeline_run_context.name}')

    return execution_lib.sort_executions_newest_to_oldest(
        prev_successful_executions)

  def _get_cached_execution_contexts(
      self,
      existing_execution: metadata_store_pb2.Execution,
      new_pipeline_run_id: str,
  ) -> List[metadata_store_pb2.Context]:
    """Gets the list of Contexts to be associated with the new cached Execution.

    Copies all the Contexts associated with the existing execution, except for
    the pipeline run context, which is updated with new pipeline run id.

    Args:
      existing_execution: The existing execution to copy from.
      new_pipeline_run_id: The pipeline run id to associate the new cached
        execution with.

    Returns:
      The list of Contexts to be associated with the new cached Execution.
    """
    result = []
    for old_context in self._mlmd.store.get_contexts_by_execution(
        existing_execution.id):
      if old_context.type_id == self._pipeline_run_type_id:
        new_pipeline_run_context = context_lib.register_context_if_not_exists(
            self._mlmd,
            context_type_name=constants.PIPELINE_RUN_CONTEXT_TYPE_NAME,
            context_name=new_pipeline_run_id)
        result.append(new_pipeline_run_context)
      else:
        result.append(old_context)
    return result

  def _cache_and_publish(self, existing_execution: metadata_store_pb2.Execution,
                         new_pipeline_run_id: str):
    """Updates MLMD."""
    cached_execution_contexts = self._get_cached_execution_contexts(
        existing_execution, new_pipeline_run_id)
    # Check if there are any previous attempts to cache and publish.
    prev_cache_executions = (
        execution_lib.get_executions_associated_with_all_contexts(
            self._mlmd, contexts=cached_execution_contexts))
    if not prev_cache_executions:
      new_execution = execution_publish_utils.register_execution(
          self._mlmd,
          execution_type=metadata_store_pb2.ExecutionType(
              id=existing_execution.type_id),
          contexts=cached_execution_contexts)
    else:
      if len(prev_cache_executions) > 1:
        logging.warning(
            'More than one previous cache executions seen when attempting '
            'reuse_node_outputs: %s', prev_cache_executions)

      if (prev_cache_executions[-1].last_known_state ==
          metadata_store_pb2.Execution.CACHED):
        return
      else:
        new_execution = prev_cache_executions[-1]

    output_artifacts = execution_lib.get_artifacts_dict(
        self._mlmd,
        existing_execution.id,
        event_type=metadata_store_pb2.Event.OUTPUT)

    execution_publish_utils.publish_cached_execution(
        self._mlmd,
        contexts=cached_execution_contexts,
        execution_id=new_execution.id,
        output_artifacts=output_artifacts)

  def reuse_node_outputs(self, node_id: str, new_pipeline_run_id: str):
    """Makes the outputs of `node_id` available to new_pipeline_run_id."""
    previous_executions = self._get_previous_execution(node_id)
    for previous_execution in previous_executions:
      self._cache_and_publish(previous_execution, new_pipeline_run_id)
