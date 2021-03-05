# Copyright 2020 Google LLC. All Rights Reserved.
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
"""TaskGenerator implementation for sync pipelines."""

from typing import Callable, List, Optional, Set

from absl import logging
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import outputs_utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import topsort


class SyncPipelineTaskGenerator(task_gen.TaskGenerator):
  """Task generator for executing a sync pipeline.

  Calling `generate` is not thread-safe. Concurrent calls to `generate` should
  be explicitly serialized. Since MLMD may be updated upon call to `generate`,
  it's also not safe to call `generate` on different instances of this class
  where the instances refer to the same MLMD db and the same pipeline IR.
  """

  def __init__(self, mlmd_handle: metadata.Metadata,
               pipeline: pipeline_pb2.Pipeline,
               is_task_id_tracked_fn: Callable[[task_lib.TaskId], bool],
               ignore_node_ids: Optional[Set[str]] = None):
    """Constructs `SyncPipelineTaskGenerator`.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline: A pipeline IR proto.
      is_task_id_tracked_fn: A callable that returns `True` if a task_id is
        tracked by the task queue.
      ignore_node_ids: Set of node ids of nodes to ignore for task generation.
    """
    self._mlmd_handle = mlmd_handle
    if pipeline.execution_mode != pipeline_pb2.Pipeline.ExecutionMode.SYNC:
      raise ValueError(
          'SyncPipelineTaskGenerator should be instantiated with a pipeline '
          'proto having execution_mode `SYNC`, not `{}`'.format(
              pipeline.execution_mode))
    for node in pipeline.nodes:
      which_node = node.WhichOneof('node')
      if which_node != 'pipeline_node':
        raise ValueError(
            'All sync pipeline nodes should be of type `PipelineNode`; found: '
            '`{}`'.format(which_node))
    self._pipeline = pipeline
    self._is_task_id_tracked_fn = is_task_id_tracked_fn
    self._node_map = {
        node.pipeline_node.node_info.id: node.pipeline_node
        for node in pipeline.nodes
    }
    self._ignore_node_ids = ignore_node_ids or set()

  def generate(self) -> List[task_lib.Task]:
    """Generates tasks for executing the next executable nodes in the pipeline.

    The returned tasks must have `exec_task` populated. List may be empty if
    no nodes are ready for execution.

    Returns:
      A `list` of tasks to execute.
    """
    layers = topsort.topsorted_layers(
        [node.pipeline_node for node in self._pipeline.nodes],
        get_node_id_fn=lambda node: node.node_info.id,
        get_parent_nodes=(
            lambda node: [self._node_map[n] for n in node.upstream_nodes]),
        get_child_nodes=(
            lambda node: [self._node_map[n] for n in node.downstream_nodes]))
    result = []
    for nodes in layers:
      # Boolean that's set if there's at least one successfully executed node
      # in the current layer.
      executed_nodes = False
      for node in nodes:
        if node.node_info.id in self._ignore_node_ids:
          logging.info(
              'Ignoring node for task generation: %s',
              task_lib.NodeUid.from_pipeline_node(self._pipeline, node))
          continue
        # If a task for the node is already tracked by the task queue, it need
        # not be considered for generation again.
        if self._is_task_id_tracked_fn(
            task_lib.exec_node_task_id_from_pipeline_node(self._pipeline,
                                                          node)):
          continue
        executions = task_gen_utils.get_executions(self._mlmd_handle, node)
        if (executions and
            task_gen_utils.is_latest_execution_successful(executions)):
          executed_nodes = True
          continue
        # If all upstream nodes are executed but current node is not executed,
        # the node is deemed ready for execution.
        if self._upstream_nodes_executed(node):
          task = self._generate_task(node)
          if task:
            result.append(task)
      # If there are no executed nodes in the current layer, downstream nodes
      # need not be checked.
      if not executed_nodes:
        break
    return result

  def _generate_task(
      self, node: pipeline_pb2.PipelineNode) -> Optional[task_lib.Task]:
    """Generates a node execution task.

    If node execution is not feasible, `None` is returned.

    Args:
      node: The pipeline node for which to generate a task.

    Returns:
      Returns a `Task` or `None` if task generation is deemed infeasible.
    """
    executions = task_gen_utils.get_executions(self._mlmd_handle, node)
    result = task_gen_utils.generate_task_from_active_execution(
        self._mlmd_handle, self._pipeline, node, executions)
    if result:
      return result

    resolved_info = task_gen_utils.generate_resolved_info(
        self._mlmd_handle, node)
    if resolved_info.input_artifacts is None:
      # TODO(goutham): If the pipeline can't make progress, there should be a
      # standard mechanism to surface it to the user.
      logging.warning(
          'Task cannot be generated for node %s since no input artifacts '
          'are resolved.', node.node_info.id)
      return None

    execution = execution_publish_utils.register_execution(
        metadata_handler=self._mlmd_handle,
        execution_type=node.node_info.type,
        contexts=resolved_info.contexts,
        input_artifacts=resolved_info.input_artifacts,
        exec_properties=resolved_info.exec_properties)
    outputs_resolver = outputs_utils.OutputsResolver(
        node, self._pipeline.pipeline_info, self._pipeline.runtime_spec,
        self._pipeline.execution_mode)
    return task_lib.ExecNodeTask(
        node_uid=task_lib.NodeUid.from_pipeline_node(self._pipeline, node),
        execution=execution,
        contexts=resolved_info.contexts,
        input_artifacts=resolved_info.input_artifacts,
        exec_properties=resolved_info.exec_properties,
        output_artifacts=outputs_resolver.generate_output_artifacts(
            execution.id),
        executor_output_uri=outputs_resolver.get_executor_output_uri(
            execution.id),
        stateful_working_dir=outputs_resolver.get_stateful_working_directory(
            execution.id),
        pipeline=self._pipeline)

  # TODO(goutham): Handle pure service nodes.
  def _upstream_nodes_executed(self, node: pipeline_pb2.PipelineNode) -> bool:
    """Returns `True` if all the upstream nodes have been successfully executed."""
    upstream_nodes = [
        node for node_id, node in self._node_map.items()
        if node_id in set(node.upstream_nodes)
    ]
    if not upstream_nodes:
      return True
    for node in upstream_nodes:
      upstream_node_executions = task_gen_utils.get_executions(
          self._mlmd_handle, node)
      if not task_gen_utils.is_latest_execution_successful(
          upstream_node_executions):
        return False
    return True
