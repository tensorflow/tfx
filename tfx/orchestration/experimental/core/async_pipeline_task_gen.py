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
"""TaskGenerator implementation for async pipelines."""

import itertools
from typing import Callable, List, Optional, Set

from absl import logging
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import outputs_utils
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib

from ml_metadata.proto import metadata_store_pb2


class AsyncPipelineTaskGenerator(task_gen.TaskGenerator):
  """Task generator for executing an async pipeline.

  Calling `generate` is not thread-safe. Concurrent calls to `generate` should
  be explicitly serialized. Since MLMD may be updated upon call to `generate`,
  it's also not safe to call `generate` on different instances of this class
  where the instances refer to the same MLMD db and the same pipeline IR.
  """

  def __init__(self,
               mlmd_handle: metadata.Metadata,
               pipeline_state: pstate.PipelineState,
               is_task_id_tracked_fn: Callable[[task_lib.TaskId], bool],
               service_job_manager: service_jobs.ServiceJobManager,
               ignore_node_ids: Optional[Set[str]] = None):
    """Constructs `AsyncPipelineTaskGenerator`.

    Args:
      mlmd_handle: A handle to MLMD db.
      pipeline_state: Pipeline state.
      is_task_id_tracked_fn: A callable that returns `True` if a task_id is
        tracked by the task queue.
      service_job_manager: Used for handling service nodes in the pipeline.
      ignore_node_ids: Set of node ids of nodes to ignore for task generation.
    """
    self._mlmd_handle = mlmd_handle
    pipeline = pipeline_state.pipeline
    if pipeline.execution_mode != pipeline_pb2.Pipeline.ExecutionMode.ASYNC:
      raise ValueError(
          'AsyncPipelineTaskGenerator should be instantiated with a pipeline '
          'proto having execution mode `ASYNC`, not `{}`'.format(
              pipeline.execution_mode))
    for node in pipeline.nodes:
      which_node = node.WhichOneof('node')
      if which_node != 'pipeline_node':
        raise ValueError(
            'Sub-pipelines are not yet supported. Async pipeline should have '
            'nodes of type `PipelineNode`; found: `{}`'.format(which_node))
    self._pipeline_state = pipeline_state
    self._pipeline = pipeline
    self._is_task_id_tracked_fn = is_task_id_tracked_fn
    self._service_job_manager = service_job_manager
    self._ignore_node_ids = ignore_node_ids or set()

  def generate(self) -> List[task_lib.Task]:
    """Generates tasks for all executable nodes in the async pipeline.

    The returned tasks must have `exec_task` populated. List may be empty if no
    nodes are ready for execution.

    Returns:
      A `list` of tasks to execute.
    """
    result = []
    for node in [n.pipeline_node for n in self._pipeline.nodes]:
      node_uid = task_lib.NodeUid.from_pipeline_node(self._pipeline, node)
      node_id = node.node_info.id
      if node_id in self._ignore_node_ids:
        logging.info('Ignoring node for task generation: %s', node_uid)
        continue

      # If this is a pure service node, there is no ExecNodeTask to generate
      # but we ensure node services and check service status.
      service_status = self._ensure_node_services_if_pure(node_id)
      if service_status is not None:
        if service_status != service_jobs.ServiceStatus.RUNNING:
          result.append(self._abort_node_task(node_uid))
        continue

      # If a task for the node is already tracked by the task queue, it need
      # not be considered for generation again but we ensure node services
      # in case of a mixed service node.
      if self._is_task_id_tracked_fn(
          task_lib.exec_node_task_id_from_pipeline_node(self._pipeline, node)):
        service_status = self._ensure_node_services_if_mixed(node_id)
        if service_status is not None:
          if service_status != service_jobs.ServiceStatus.RUNNING:
            result.append(self._abort_node_task(node_uid))
        continue
      task = self._generate_task(self._mlmd_handle, node)
      if task:
        result.append(task)
    return result

  def _generate_task(
      self, metadata_handler: metadata.Metadata,
      node: pipeline_pb2.PipelineNode) -> Optional[task_lib.Task]:
    """Generates a node execution task.

    If a node execution is not feasible, `None` is returned.

    Args:
      metadata_handler: A handler to access MLMD db.
      node: The pipeline node for which to generate a task.

    Returns:
      Returns a `Task` or `None` if task generation is deemed infeasible.
    """
    executions = task_gen_utils.get_executions(metadata_handler, node)
    result = task_gen_utils.generate_task_from_active_execution(
        metadata_handler, self._pipeline, node, executions)
    if result:
      return result

    resolved_info = task_gen_utils.generate_resolved_info(
        metadata_handler, node)
    if resolved_info.input_artifacts is None or not any(
        resolved_info.input_artifacts.values()):
      logging.info(
          'Task cannot be generated for node %s since no input artifacts '
          'are resolved.', node.node_info.id)
      return None

    # If the latest successful execution had the same resolved input artifacts,
    # the component should not be triggered, so task is not generated.
    # TODO(b/170231077): This logic should be handled by the resolver when it's
    # implemented. Also, currently only the artifact ids of previous execution
    # are checked to decide if a new execution is warranted but it may also be
    # necessary to factor in the difference of execution properties.
    latest_exec = task_gen_utils.get_latest_successful_execution(executions)
    if latest_exec:
      artifact_ids_by_event_type = (
          execution_lib.get_artifact_ids_by_event_type_for_execution_id(
              metadata_handler, latest_exec.id))
      latest_exec_input_artifact_ids = artifact_ids_by_event_type.get(
          metadata_store_pb2.Event.INPUT, set())
      current_exec_input_artifact_ids = set(
          a.id
          for a in itertools.chain(*resolved_info.input_artifacts.values()))
      if latest_exec_input_artifact_ids == current_exec_input_artifact_ids:
        return None

    node_uid = task_lib.NodeUid.from_pipeline_node(self._pipeline, node)
    execution = execution_publish_utils.register_execution(
        metadata_handler=metadata_handler,
        execution_type=node.node_info.type,
        contexts=resolved_info.contexts,
        input_artifacts=resolved_info.input_artifacts,
        exec_properties=resolved_info.exec_properties)
    outputs_resolver = outputs_utils.OutputsResolver(
        node, self._pipeline.pipeline_info, self._pipeline.runtime_spec,
        self._pipeline.execution_mode)

    # For mixed service nodes, we ensure node services and check service
    # status; the node is aborted if its service jobs have failed.
    service_status = self._ensure_node_services_if_mixed(node.node_info.id)
    if service_status is not None:
      if service_status != service_jobs.ServiceStatus.RUNNING:
        return self._abort_node_task(node_uid)

    return task_lib.ExecNodeTask(
        node_uid=node_uid,
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

  def _ensure_node_services_if_pure(
      self, node_id: str) -> Optional[service_jobs.ServiceStatus]:
    """Calls `ensure_node_services` and returns status if given node is pure service node."""
    if self._service_job_manager.is_pure_service_node(self._pipeline_state,
                                                      node_id):
      return self._service_job_manager.ensure_node_services(
          self._pipeline_state, node_id)
    return None

  def _ensure_node_services_if_mixed(
      self, node_id: str) -> Optional[service_jobs.ServiceStatus]:
    """Calls `ensure_node_services` and returns status if given node is mixed service node."""
    if self._service_job_manager.is_mixed_service_node(self._pipeline_state,
                                                       node_id):
      return self._service_job_manager.ensure_node_services(
          self._pipeline_state, node_id)
    return None

  def _abort_node_task(self,
                       node_uid: task_lib.NodeUid) -> task_lib.FinalizeNodeTask:
    """Returns task to abort the node execution."""
    logging.error('Required service node not running or healthy, node uid: %s',
                  node_uid)
    return task_lib.FinalizeNodeTask(
        node_uid=node_uid,
        status=status_lib.Status(
            code=status_lib.Code.ABORTED,
            message=(f'Aborting node execution as the associated service '
                     f'job is not running or healthy; problematic node '
                     f'uid: {node_uid}')))
