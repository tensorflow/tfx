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
import concurrent.futures
import enum
from typing import Collection, Dict, Final, List, Mapping, Optional, OrderedDict, Sequence, Set, Tuple
import uuid

from absl import logging
from tfx.dsl.compiler import compiler_utils
from tfx.dsl.compiler import constants
from tfx.orchestration import metadata
from tfx.orchestration import node_proto_view
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2

from ml_metadata.proto import metadata_store_pb2


_default_snapshot_settings = pipeline_pb2.SnapshotSettings()
_default_snapshot_settings.latest_pipeline_run_strategy.SetInParent()
_REUSE_ARTIFACT_REQUIRED = pipeline_pb2.NodeExecutionOptions.Skip.REQUIRED
_REUSE_ARTIFACT_OPTIONAL = pipeline_pb2.NodeExecutionOptions.Skip.OPTIONAL


def latest_pipeline_snapshot_settings() -> pipeline_pb2.SnapshotSettings:
  """Returns snapshot settings with latest pipeline run strategy set."""
  return _default_snapshot_settings


def set_latest_pipeline_run_strategy(
    snapshot_settings: pipeline_pb2.SnapshotSettings) -> None:
  """Sets latest pipeline run strategy in snapshot_settings."""
  snapshot_settings.latest_pipeline_run_strategy.SetInParent()


def set_base_pipeline_run_strategy(
    snapshot_settings: pipeline_pb2.SnapshotSettings, base_run_id: str) -> None:
  """Sets base pipeline run strategy in snapshot_settings."""
  snapshot_settings.base_pipeline_run_strategy.base_run_id = base_run_id


def mark_pipeline(
    pipeline: pipeline_pb2.Pipeline,
    from_nodes: Optional[Collection[str]] = None,
    to_nodes: Optional[Collection[str]] = None,
    skip_nodes: Optional[Collection[str]] = None,
    skip_snapshot_nodes: Optional[Collection[str]] = None,
    snapshot_settings: pipeline_pb2
    .SnapshotSettings = _default_snapshot_settings,
) -> pipeline_pb2.Pipeline:
  """Modifies the Pipeline IR in place, in preparation for partial run.

  This function modifies the node-level execution_options to annotate them with
  additional information needed for partial runs, such as which nodes to run,
  which nodes to skip, which node is responsible for performing the snapshot,
  as well as the snapshot settings.

  The set of nodes to be run is the set of nodes between from_nodes and to_nodes
  -- i.e., the set of nodes that are reachable by traversing downstream from
  `from_nodes` AND also reachable by traversing upstream from `to_nodes`.

  Any `reusable_nodes` will be reused in the partial run, as long as they do not
  depend on a node that is marked to run.

  Args:
    pipeline: A valid compiled Pipeline IR proto to be marked.
    from_nodes: The collection of nodes where the "sweep" starts (see detailed
      description). If None, selects all nodes.
    to_nodes: The collection of nodes where the "sweep" ends (see detailed
      description). If None, selects all nodes.
    skip_nodes: **MOST USERS DO NOT NEED THIS.** Use this to force-skip nodes
      that would otherwise have been marked to run. Note that if a node depends
      on nodes that cannot be skipped, then it is still marked to run for
      pipeline result correctness. If None, does not force-skip any node.
    skip_snapshot_nodes: **MOST USERS DO NOT NEED THIS.** Use this to force
      marking snapshot as OPTIONAL for a skipped node. Setting this field can
      be dangerous -- it can cause following partial runs to behave incorrectly
      with missing cached executions.
    snapshot_settings: Settings needed to perform the snapshot step. Defaults to
      using LATEST_PIPELINE_RUN strategy.

  Returns:
    Updated pipeline IR.

  Raises:
    ValueError: If pipeline's execution_mode is not SYNC.
    ValueError: If pipeline contains a sub-pipeline.
    ValueError: If pipeline was already marked for partial run.
    ValueError: If both from_nodes and to_nodes are empty.
    ValueError: If from_nodes/to_nodes contain node_ids not in the pipeline.
    ValueError: If pipeline is not topologically sorted.
  """
  nodes = node_proto_view.get_view_for_all_in(pipeline)
  _ensure_sync_pipeline(pipeline)
  _ensure_no_partial_run_marks(nodes)
  _ensure_not_full_run(from_nodes, to_nodes)
  _ensure_no_missing_nodes(nodes, from_nodes, to_nodes)
  _ensure_topologically_sorted(nodes)

  node_map = make_ordered_node_map(nodes)

  from_node_ids = from_nodes or node_map.keys()
  to_node_ids = to_nodes or node_map.keys()
  skip_node_ids = skip_nodes or []
  skip_snapshot_node_ids = set(skip_snapshot_nodes or [])
  nodes_to_run = compute_nodes_to_run(node_map, from_node_ids, to_node_ids,
                                      skip_node_ids)

  nodes_required_to_reuse, nodes_to_reuse = _compute_nodes_to_reuse(
      node_map, nodes_to_run, skip_snapshot_node_ids)
  nodes_requiring_snapshot = _compute_nodes_requiring_snapshot(
      node_map, nodes_to_run, nodes_to_reuse)
  snapshot_node = _pick_snapshot_node(node_map, nodes_to_run, nodes_to_reuse)
  _mark_nodes(node_map, nodes_to_run, nodes_required_to_reuse, nodes_to_reuse,
              nodes_requiring_snapshot, snapshot_node)
  pipeline.runtime_spec.snapshot_settings.CopyFrom(snapshot_settings)
  return pipeline


def snapshot(mlmd_handle: metadata.Metadata,
             pipeline: pipeline_pb2.Pipeline,
             base_run_id: Optional[str] = None) -> None:
  """Performs a snapshot.

  This operation modifies the MLMD state, so that the dependencies of
  a partial run can be resolved as if the reused artifacts were produced in the
  same pipeline run.

  Args:
    mlmd_handle: A handle to the MLMD db.
    pipeline: The marked pipeline IR.
    base_run_id: The base pipeline run ID to reuse artifacts from.

  Raises:
    ValueError: If pipeline_node has a snashot_settings field set, but the
      artifact_reuse_strategy field is not set in it.
  """
  # Avoid unnecessary snapshotting step if no node needs to reuse any artifacts.
  if not any(
      should_attempt_to_reuse_artifact(node.pipeline_node.execution_options)
      for node in pipeline.nodes):
    return

  snapshot_settings = pipeline.runtime_spec.snapshot_settings
  logging.info('snapshot_settings: %s', snapshot_settings)
  if not base_run_id:
    if snapshot_settings.HasField('base_pipeline_run_strategy'):
      base_run_id = snapshot_settings.base_pipeline_run_strategy.base_run_id
      logging.info(
          'Using base_pipeline_run_strategy with base_run_id=%s', base_run_id
      )
    elif snapshot_settings.HasField('latest_pipeline_run_strategy'):
      logging.info('Using latest_pipeline_run_strategy.')
    else:
      raise ValueError('artifact_reuse_strategy not set in SnapshotSettings.')
  logging.info('Preparing to reuse artifacts.')
  _reuse_pipeline_run_artifacts(mlmd_handle, pipeline, base_run_id=base_run_id)
  logging.info('Artifact reuse complete.')


def _pick_snapshot_node(
    node_map: Mapping[str, node_proto_view.NodeProtoView],
    nodes_to_run: Set[str],
    nodes_to_reuse: Set[str],
) -> Optional[str]:
  """Returns node_id to perform snapshot, or None if snapshot is unnecessary."""
  if not nodes_to_reuse:
    return None
  for node_id in node_map:
    if node_id in nodes_to_run:
      return node_id
  return None


def _mark_nodes(
    node_map: Mapping[str, node_proto_view.NodeProtoView],
    nodes_to_run: Set[str],
    nodes_required_to_reuse: Set[str],
    nodes_to_reuse: Set[str],
    nodes_requiring_snapshot: Set[str],
    snapshot_node: Optional[str],
):
  """Mark nodes."""
  for node_id, node in node_map.items():  # assumes topological order
    if node_id in nodes_to_run:
      node.execution_options.run.perform_snapshot = (node_id == snapshot_node)
      node.execution_options.run.depends_on_snapshot = (
          node_id in nodes_requiring_snapshot)
    elif node_id in nodes_required_to_reuse:
      node.execution_options.skip.reuse_artifacts_mode = (
          _REUSE_ARTIFACT_REQUIRED)
    elif node_id in nodes_to_reuse:
      node.execution_options.skip.reuse_artifacts_mode = (
          _REUSE_ARTIFACT_OPTIONAL)
    else:
      node.execution_options.skip.reuse_artifacts_mode = (
          pipeline_pb2.NodeExecutionOptions.Skip.NEVER)


class _Direction(enum.Enum):
  UPSTREAM = 1
  DOWNSTREAM = 2


def _ensure_sync_pipeline(pipeline: pipeline_pb2.Pipeline):
  """Raises ValueError if the pipeline's execution_mode is not SYNC."""
  if pipeline.execution_mode != pipeline_pb2.Pipeline.SYNC:
    raise ValueError(
        'Pipeline filtering is only supported for SYNC execution modes; '
        'found pipeline with execution mode: '
        f'{pipeline_pb2.Pipeline.ExecutionMode.Name(pipeline.execution_mode)}')


def _ensure_not_full_run(from_nodes: Optional[Collection[str]] = None,
                         to_nodes: Optional[Collection[str]] = None):
  """Raises ValueError if both from_nodes and to_nodes are falsy."""
  if not (from_nodes or to_nodes):
    raise ValueError('Both from_nodes and to_nodes are empty.')


def _ensure_no_partial_run_marks(
    nodes: Sequence[node_proto_view.NodeProtoView],
):
  """Raises ValueError if the pipeline is already marked for partial run."""
  for node in nodes:
    if node.execution_options.HasField('partial_run_option'):
      raise ValueError('Pipeline has already been marked for partial run.')


def _ensure_no_missing_nodes(
    nodes: Sequence[node_proto_view.NodeProtoView],
    from_nodes: Optional[Collection[str]] = None,
    to_nodes: Optional[Collection[str]] = None,
):
  """Raises ValueError if there are from_nodes/to_nodes not in the pipeline."""
  all_node_ids = set(node.node_info.id for node in nodes)
  missing_nodes = (set(from_nodes or []) | set(to_nodes or [])) - all_node_ids
  if missing_nodes:
    raise ValueError(
        f'Nodes {sorted(missing_nodes)} specified in from_nodes/to_nodes '
        f'are not present in the pipeline. Valid nodes are {all_node_ids}.')


def _ensure_topologically_sorted(
    nodes: Sequence[node_proto_view.NodeProtoView],
):
  """Raises ValueError if nodes are not topologically sorted.

  If the pipeline comes from the compiler, it should already be
  topologically sorted. This is just in case the IR proto was modified or
  created in another way.

  Args:
    nodes: The input nodes.

  Raises:
    ValueError: If the pipeline is not topologically sorted.
  """
  # Upstream check
  visited = set()
  for node in nodes:
    for upstream_node in node.upstream_nodes:
      if upstream_node not in visited:
        raise ValueError(
            'Input pipeline is not topologically sorted. '
            f'node {node.node_info.id} has upstream_node {upstream_node}, but '
            f'{upstream_node} does not appear before {node.node_info.id}')
    visited.add(node.node_info.id)
  # Downstream check
  visited.clear()
  for node in reversed(nodes):
    for downstream_node in node.downstream_nodes:
      if downstream_node not in visited:
        raise ValueError(
            'Input pipeline is not topologically sorted. '
            f'node {node.node_info.id} has downstream_node {downstream_node}, '
            f'but {downstream_node} does not appear after {node.node_info.id}')
    visited.add(node.node_info.id)


def make_ordered_node_map(
    nodes: Sequence[node_proto_view.NodeProtoView],
) -> OrderedDict[str, node_proto_view.NodeProtoView]:
  """Prepares the Pipeline proto for DAG traversal.

  Args:
    nodes: The input nodes, which must be sorted topologically.

  Returns:
    An OrderedDict that maps node_ids to PipelineNodes.
  """
  result = collections.OrderedDict()
  for node in nodes:
    node_id = node.node_info.id
    result[node_id] = node
  return result


def _traverse(
    node_map: Mapping[str, node_proto_view.NodeProtoView],
    direction: _Direction,
    start_nodes: Collection[str],
) -> Set[str]:
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


def compute_nodes_to_run(
    node_map: OrderedDict[str, node_proto_view.NodeProtoView],
    from_node_ids: Collection[str],
    to_node_ids: Collection[str],
    skip_node_ids: Collection[str],
) -> Set[str]:
  """Returns the set of nodes between from_node_ids and to_node_ids."""
  ancestors_of_to_nodes = _traverse(node_map, _Direction.UPSTREAM, to_node_ids)
  descendents_of_from_nodes = _traverse(node_map, _Direction.DOWNSTREAM,
                                        from_node_ids)
  nodes_considered_to_run = ancestors_of_to_nodes.intersection(
      descendents_of_from_nodes)
  nodes_to_run = _traverse(node_map, _Direction.DOWNSTREAM,
                           nodes_considered_to_run - set(skip_node_ids))
  return nodes_considered_to_run.intersection(nodes_to_run)


def _compute_nodes_to_reuse(
    node_map: Mapping[str, node_proto_view.NodeProtoView],
    nodes_to_run: Set[str],
    skip_snapshot_node_ids: Set[str],
) -> Tuple[Set[str], Set[str]]:
  """Returns the set of node ids whose output artifacts are to be reused.

    Only upstream nodes of nodes_to_run are required to be reused to reflect
    correct lineage.

  Args:
    node_map: Mapping of node_id to nodes.
    nodes_to_run: The set of nodes to run.
    skip_snapshot_node_ids: The set of nodes that can be skipped for
      snapshotting.

  Returns:
    Set of node ids required to be reused.
  """
  exclusion_set = _traverse(
      node_map, _Direction.DOWNSTREAM, start_nodes=nodes_to_run)
  nodes_required_to_reuse = _traverse(
      node_map, _Direction.UPSTREAM,
      start_nodes=nodes_to_run) - exclusion_set - skip_snapshot_node_ids
  nodes_to_reuse = set(node_map.keys()) - exclusion_set
  return nodes_required_to_reuse, nodes_to_reuse


def _compute_nodes_requiring_snapshot(
    node_map: Mapping[str, node_proto_view.NodeProtoView],
    nodes_to_run: Set[str],
    nodes_to_reuse: Set[str],
) -> Set[str]:
  """Returns the set of nodes to run that depend on a node to reuse."""
  result = set()
  for node_id, node in node_map.items():
    if node_id not in nodes_to_run:
      continue
    for upstream_node_id in node.upstream_nodes:
      if upstream_node_id in nodes_to_reuse:
        result.add(node_id)
        break
  return result


def _get_validated_new_run_id(pipeline: pipeline_pb2.Pipeline,
                              new_run_id: Optional[str] = None) -> str:
  """Attempts to obtain a unique new_run_id.

  Args:
    pipeline: The pipeline IR, whose runtime parameters are already resolved.
    new_run_id: The pipeline_run_id to associate those output artifacts with.
      This function will always attempt to infer the new run id from `pipeline`.
      If not found, it would use the provided `new_run_id`. If found, and
      `new_run_id` is provided, it would verify that it equals the inferred run
      id, and raise an error if they are not equal.

  Returns:
    The validated pipeline_run_id.

  Raises:
    ValueError: If `pipeline` does not contain a pipeline run id, and
      `new_run_id` is not provided.
    ValueError: If `pipeline` does contain a pipeline run id, and
      `new_run_id` is provided, but they are not equal.
  """
  inferred_new_run_id = None
  run_id_value = pipeline.runtime_spec.pipeline_run_id
  if run_id_value.HasField('field_value'):
    inferred_new_run_id = run_id_value.field_value.string_value

  if not (inferred_new_run_id or new_run_id):
    raise ValueError(
        'Unable to infer new pipeline run id. Either resolve the '
        'pipeline_run_id RuntimeParameter in `filtered_pipeline` first, or '
        'provide a `new_run_id` explicitly.')

  if new_run_id and inferred_new_run_id and new_run_id != inferred_new_run_id:
    raise ValueError(
        'Conflicting new pipeline run ids found. pipeline_run_id='
        f'{inferred_new_run_id} was inferred from `full_pipeline`, while '
        f'new_run_id={new_run_id} was explicitly provided. '
        'Consider omitting `new_run_id`, and simply use the pipeline_run_id '
        'inferred from `full_pipeline` as the new_run_id.')

  # The following OR expression will never evaluate to None, because we have
  # already checked above. However, pytype doesn't know that, so we need to cast
  # to the expression to str so that the return type is str.
  return str(inferred_new_run_id or new_run_id)


def should_attempt_to_reuse_artifact(
    execution_options: pipeline_pb2.NodeExecutionOptions):
  """Returns whether artifacts should be reused for the these execution options."""
  return execution_options.HasField('skip') and (
      execution_options.skip.reuse_artifacts or
      execution_options.skip.reuse_artifacts_mode == _REUSE_ARTIFACT_OPTIONAL or
      execution_options.skip.reuse_artifacts_mode == _REUSE_ARTIFACT_REQUIRED)


def _reuse_artifact_required(
    execution_options: pipeline_pb2.NodeExecutionOptions):
  return execution_options.HasField('skip') and (
      execution_options.skip.reuse_artifacts or
      execution_options.skip.reuse_artifacts_mode == _REUSE_ARTIFACT_REQUIRED)


def _reuse_pipeline_run_artifacts(
    metadata_handle: metadata.Metadata,
    marked_pipeline: pipeline_pb2.Pipeline,
    base_run_id: Optional[str] = None,
    new_run_id: Optional[str] = None,
):
  """Reuses the output Artifacts from a previous pipeline run.

  This computes the maximal set of nodes whose outputs can be associated with
  the new pipeline_run without creating any inconsistencies, and reuses their
  node outputs (similar to repeatedly calling `reuse_node_outputs`). It also
  puts a ParentContext into MLMD, with the `base_run_id` being the parent
  context, and the new run_id (provided by the user, or inferred from
  `pipeline`) as the child context.

  Args:
    metadata_handle: A handler to access MLMD store.
    marked_pipeline: The output of mark_pipeline function.
    base_run_id: The pipeline_run_id where the output artifacts were produced.
      Defaults to the latest previous pipeline run to use as base_run_id.
    new_run_id: The pipeline_run_id to associate those output artifacts with.
      This function will always attempt to infer the new run id from
      `full_pipeline`'s IR. If not found, it would use the provided
      `new_run_id`. If found, and `new_run_id` is provided, it would verify that
      it is the same as the inferred run id, and raise an error if they are not
      the same.

  Raises:
    ValueError: If `marked_pipeline` does not contain a pipeline run id, and
      `new_run_id` is not provided.
    ValueError: If `marked_pipeline` does contain a pipeline run id, and
      `new_run_id` is provided, but they are not the same.
  """
  validated_new_run_id = _get_validated_new_run_id(marked_pipeline, new_run_id)
  artifact_recycler = _ArtifactRecycler(
      metadata_handle,
      pipeline_name=marked_pipeline.pipeline_info.id,
      new_run_id=validated_new_run_id,
      base_run_id=base_run_id,
      new_pipeline_run_ir=marked_pipeline,
  )

  reuse_nodes = [
      node
      for node in node_proto_view.get_view_for_all_in(marked_pipeline)
      if should_attempt_to_reuse_artifact(node.execution_options)
  ]
  logging.info(
      'Reusing nodes: %s', [n.node_info.id for n in reuse_nodes]
  )
  with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {}
    for node in reuse_nodes:
      node_id = node.node_info.id
      futures[node_id] = (
          node,
          executor.submit(
              artifact_recycler.reuse_node_outputs,
              node=node
          ),
      )

    for node_id, (node, future) in futures.items():
      try:
        future.result()
      except Exception as err:  # pylint: disable=broad-except
        if _reuse_artifact_required(node.execution_options):
          # Raise error only if failed to reuse artifacts required.
          raise
        err_str = str(err)
        if 'No previous successful exe' in err_str or 'node context' in err_str:
          # This is mostly due to no previous execution of the node, so the
          # error is safe to suppress.
          logging.info(err_str)
        else:
          logging.warning('Failed to reuse artifacts for node %s. Due to %s',
                          node_id, err_str)
  artifact_recycler.put_parent_context()


class _ArtifactRecycler:
  """Allows previously-generated Artifacts to be used in a new pipeline run.

  By implementing this in a class (instead of a function), we reduce the
  number of MLMD reads when reusing the outputs of multiple nodes in the same
  pipeline run.
  """

  def __init__(
      self,
      metadata_handle: metadata.Metadata,
      pipeline_name: str,
      new_run_id: str,
      new_pipeline_run_ir: pipeline_pb2.Pipeline,
      base_run_id: Optional[str] = None,
  ):
    self._mlmd = metadata_handle
    self._pipeline_name: Final[str] = pipeline_name
    self._pipeline_context: Final[metadata_store_pb2.Context] = (
        context_lib.register_context_if_not_exists(
            self._mlmd,
            context_type_name=constants.PIPELINE_CONTEXT_TYPE_NAME,
            context_name=self._pipeline_name,
        )
    )
    self._new_run_id: Final[str] = new_run_id
    self._new_pipeline_run_context: Final[metadata_store_pb2.Context] = (
        context_lib.register_context_if_not_exists(
            self._mlmd,
            context_type_name=constants.PIPELINE_RUN_CONTEXT_TYPE_NAME,
            context_name=self._new_run_id,
            parent_contexts=[self._pipeline_context],
        )
    )

    self._base_run_context: Final[metadata_store_pb2.Context] = (
        self._get_base_pipeline_run_context(base_run_id)
    )

    self._node_context_by_name: Final[Dict[str, metadata_store_pb2.Context]] = {
        ctx.name: ctx
        for ctx in self._mlmd.store.get_contexts_by_type(
            constants.NODE_CONTEXT_TYPE_NAME
        )
    }

    self._nodes_by_id = {
        node.node_info.id: node
        for node in node_proto_view.get_view_for_all_in(new_pipeline_run_ir)
    }

  def _get_base_pipeline_run_context(
      self, base_run_id: Optional[str] = None
  ) -> metadata_store_pb2.Context:
    """Gets the latest previous pipeline_run_id."""
    child_contexts = self._mlmd.store.get_children_contexts_by_context(
        self._pipeline_context.id
    )
    if child_contexts:
      pipeline_run_type_id = self._mlmd.store.get_context_type(
          constants.PIPELINE_RUN_CONTEXT_TYPE_NAME
      ).id
      pipeline_run_contexts = {
          ctx.name: ctx
          for ctx in child_contexts
          if ctx.type_id == pipeline_run_type_id
      }
    else:
      logging.info('No child contexts found. Falling back to previous logic.')
      # The parent-child relationship between pipeline and pipeline run contexts
      # is set up after the partial run feature is available to users. For
      # existing pipelines, we need to fall back to the previous logic.
      pipeline_run_contexts = {
          run_ctx.name: run_ctx
          for run_ctx in self._mlmd.store.get_contexts_by_type(
              constants.PIPELINE_RUN_CONTEXT_TYPE_NAME
          )
      }
    if base_run_id:
      if base_run_id in pipeline_run_contexts:
        return pipeline_run_contexts[base_run_id]
      else:
        raise LookupError(f'pipeline_run_id {base_run_id} not found in MLMD.')

    pipeline_run_contexts = [
        c for c in pipeline_run_contexts.values() if c.name != self._new_run_id
    ]
    sorted_run_contexts = sorted(
        pipeline_run_contexts, key=lambda c: c.create_time_since_epoch
    )
    if not sorted_run_contexts:
      return None

    logging.info(
        'base_run_id not provided. Default to latest pipeline run: %s',
        sorted_run_contexts[-1].name,
    )
    return sorted_run_contexts[-1]

  def _get_node_context(
      self, node: node_proto_view.NodeProtoView
  ) -> list[metadata_store_pb2.Context]:
    """Returns node contexts for node.

    For subpipelines, both the end node context and subpipeline as node context
    are returned.

    Args:
      node: The node to get the contexts for.

    Returns: The node contexts for the node.

    Raises:
      LookupError: If the node context is not found.
      ValueError: If fetching contexts for a subpipeline with no parent pipeline
        ids.
    """
    contexts = []
    node_id = node.node_info.id
    # Return the end node context if we want to reuse a subpipeline. We do this
    # because nodes dependent on a subpipeline use the subpipeline's end node
    # to get their aritfacts from, so we reuse those artifacts.
    if isinstance(node, node_proto_view.ComposablePipelineProtoView):
      # TODO: b/340911977 - Once we only have subpipeline as node for input
      # context queries, we should remove the end node context.
      context_name = compiler_utils.end_node_context_name_from_subpipeline_id(
          node_id
      )
      # Subpipelines are also considered a node in the parent pipeline, so we
      # also need to add the pipeline as node context.
      parent_pipeline_ids = node.raw_proto().pipeline_info.parent_ids
      if not parent_pipeline_ids:
        raise ValueError(
            f'Subpipeline {node_id} does not have any parent pipelines.'
        )
      parent_pipeline_name = parent_pipeline_ids[-1]
      pipeline_as_node_name = compiler_utils.node_context_name(
          parent_pipeline_name, node_id
      )
      pipeline_as_node_context = self._node_context_by_name.get(
          pipeline_as_node_name
      )
      if pipeline_as_node_context is None:
        raise LookupError(
            f'node context {pipeline_as_node_name} not found in MLMD.'
        )
      contexts.append(pipeline_as_node_context)
    else:
      context_name = compiler_utils.node_context_name(
          self._pipeline_name, node_id
      )
    node_context = self._node_context_by_name.get(context_name)
    if node_context is None:
      raise LookupError(f'node context {context_name} not found in MLMD.')
    contexts.append(node_context)
    return contexts

  def _get_successful_executions(
      self, node: node_proto_view.NodeProtoView
  ) -> List[metadata_store_pb2.Execution]:
    """Gets all successful Executions of a given node in a given pipeline run.

    Args:
      node: The node whose Executions to query.

    Returns:
      All successful executions for that node at that run_id.

    Raises:
      LookupError: If no successful Execution was found.
    """
    node_contexts = self._get_node_context(node)
    node_id = node.node_info.id
    if not self._base_run_context:
      raise LookupError(
          f'No previous run is found for {node_id}. '
          'You need to have completed a pipeline run before performing a '
          'partial run with artifact reuse.'
      )

    all_associated_executions = (
        execution_lib.get_executions_associated_with_all_contexts(
            self._mlmd, contexts=[self._base_run_context] + node_contexts
        )
    )
    cache_only_succesful_executions = (
        not node.execution_options.node_success_optional
    )
    for ds in node.downstream_nodes:
      downstream = self._nodes_by_id[ds]
      # If a downstream node is upstream optional then we should also cache this
      # node's failed executions.
      if downstream.execution_options.strategy in (
          pipeline_pb2.NodeExecutionOptions.TriggerStrategy.ALL_UPSTREAM_NODES_COMPLETED,
          pipeline_pb2.NodeExecutionOptions.TriggerStrategy.LAZILY_ALL_UPSTREAM_NODES_COMPLETED,
      ):
        cache_only_succesful_executions = False
        break

    if cache_only_succesful_executions:
      prev_successful_executions = [
          e
          for e in all_associated_executions
          if execution_lib.is_execution_successful(e)
      ]
    else:
      prev_successful_executions = [
          e
          for e in all_associated_executions
          if not execution_lib.is_execution_active(e)
      ]
    if not prev_successful_executions:
      raise LookupError(
          f'No previous successful executions found for node_id {node_id} in '
          f'pipeline_run {self._base_run_context.name}'
      )

    return prev_successful_executions

  def _cache_and_publish(
      self,
      existing_executions: List[metadata_store_pb2.Execution],
      node: node_proto_view.NodeProtoView,
  ):
    """Creates and publishes cache executions."""
    if not existing_executions:
      return

    # Check if there are any previous attempts to cache and publish.
    node_contexts = self._get_node_context(node)
    cached_execution_contexts = [
        self._pipeline_context,
        self._new_pipeline_run_context,
    ] + node_contexts
    prev_cache_executions = (
        execution_lib.get_executions_associated_with_all_contexts(
            self._mlmd,
            contexts=[self._new_pipeline_run_context] + node_contexts,
        )
    )
    if not prev_cache_executions:
      new_cached_executions = []
      for e in existing_executions:
        new_cached_executions.append(
            execution_lib.prepare_execution(
                metadata_handle=self._mlmd,
                execution_type=metadata_store_pb2.ExecutionType(id=e.type_id),
                state=metadata_store_pb2.Execution.RUNNING,
                execution_name=str(uuid.uuid4()),
            )
        )
    else:
      new_cached_executions = [
          e
          for e in prev_cache_executions
          if e.last_known_state != metadata_store_pb2.Execution.CACHED
      ]
    logging.info(
        'New cached executions to be published: %s', new_cached_executions
    )
    if not new_cached_executions:
      return
    if len(new_cached_executions) != len(existing_executions):
      raise RuntimeError(
          'The number of new executions is not the same as the number of'
          ' existing executions.'
      )

    output_artifacts_maps = [
        execution_lib.get_output_artifacts(self._mlmd, e.id)
        for e in existing_executions
    ]
    execution_publish_utils.publish_cached_executions(
        self._mlmd,
        contexts=cached_execution_contexts,
        executions=new_cached_executions,
        output_artifacts_maps=output_artifacts_maps,
    )

  def put_parent_context(self):
    """Puts a ParentContext edge in MLMD."""
    if not self._base_run_context or not self._new_pipeline_run_context:
      logging.warning(
          'base run context %s or new pipeline run context %s not found.',
          self._base_run_context,
          self._new_pipeline_run_context,
      )
      return

    context_lib.put_parent_context_if_not_exists(
        self._mlmd,
        parent_id=self._base_run_context.id,
        child_id=self._new_pipeline_run_context.id,
    )

  def reuse_node_outputs(self, node: node_proto_view.NodeProtoView):
    """Makes the outputs of `node_id` available to new_pipeline_run_id."""
    previous_executions = self._get_successful_executions(node)
    self._cache_and_publish(previous_executions, node)
