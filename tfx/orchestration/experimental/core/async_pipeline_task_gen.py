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
from typing import Callable, Dict, List, Optional

from absl import logging
from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration import node_proto_view
from tfx.orchestration.experimental.core import mlmd_state
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

  def __init__(self, mlmd_handle: metadata.Metadata,
               is_task_id_tracked_fn: Callable[[task_lib.TaskId], bool],
               service_job_manager: service_jobs.ServiceJobManager):
    """Constructs `AsyncPipelineTaskGenerator`.

    Args:
      mlmd_handle: A handle to MLMD db.
      is_task_id_tracked_fn: A callable that returns `True` if a task_id is
        tracked by the task queue.
      service_job_manager: Used for handling service nodes in the pipeline.
    """
    self._mlmd_handle = mlmd_handle
    self._is_task_id_tracked_fn = is_task_id_tracked_fn
    self._service_job_manager = service_job_manager

  def generate(self,
               pipeline_state: pstate.PipelineState) -> List[task_lib.Task]:
    """Generates tasks for all executable nodes in the async pipeline.

    The returned tasks must have `exec_task` populated. List may be empty if no
    nodes are ready for execution.

    Args:
      pipeline_state: The `PipelineState` object associated with the pipeline
        for which to generate tasks.

    Returns:
      A `list` of tasks to execute.
    """
    return _Generator(self._mlmd_handle, pipeline_state,
                      self._is_task_id_tracked_fn, self._service_job_manager)()


class _Generator:
  """Generator implementation class for AsyncPipelineTaskGenerator."""

  def __init__(self, mlmd_handle: metadata.Metadata,
               pipeline_state: pstate.PipelineState,
               is_task_id_tracked_fn: Callable[[task_lib.TaskId], bool],
               service_job_manager: service_jobs.ServiceJobManager):
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

  def __call__(self) -> List[task_lib.Task]:
    result = []
    for node in [node_proto_view.get_view(n) for n in self._pipeline.nodes]:
      node_uid = task_lib.NodeUid.from_node(self._pipeline, node)
      node_id = node.node_info.id

      with self._pipeline_state:
        node_state = self._pipeline_state.get_node_state(node_uid)
        if node_state.state in (pstate.NodeState.STOPPING,
                                pstate.NodeState.STOPPED,
                                pstate.NodeState.PAUSING,
                                pstate.NodeState.PAUSED):
          logging.info('Ignoring node in state \'%s\' for task generation: %s',
                       node_state.state, node_uid)
          continue

      # If this is a pure service node, there is no ExecNodeTask to generate
      # but we ensure node services and check service status.
      service_status = self._ensure_node_services_if_pure(node_id)
      if service_status is not None:
        if service_status != service_jobs.ServiceStatus.RUNNING:
          error_msg = f'associated service job failed; node uid: {node_uid}'
          result.append(
              task_lib.UpdateNodeStateTask(
                  node_uid=node_uid,
                  state=pstate.NodeState.FAILED,
                  status=status_lib.Status(
                      code=status_lib.Code.ABORTED, message=error_msg)))
        elif node_state.state != pstate.NodeState.RUNNING:
          result.append(
              task_lib.UpdateNodeStateTask(
                  node_uid=node_uid, state=pstate.NodeState.RUNNING))
        continue

      # For mixed service nodes, we ensure node services and check service
      # status; the node is aborted if its service jobs have failed.
      service_status = self._ensure_node_services_if_mixed(node.node_info.id)
      if service_status is not None:
        if service_status != service_jobs.ServiceStatus.RUNNING:
          error_msg = f'associated service job failed; node uid: {node_uid}'
          result.append(
              task_lib.UpdateNodeStateTask(
                  node_uid=node_uid,
                  state=pstate.NodeState.FAILED,
                  status=status_lib.Status(
                      code=status_lib.Code.ABORTED, message=error_msg)))
          continue

      # If a task for the node is already tracked by the task queue, it need
      # not be considered for generation again.
      if self._is_task_id_tracked_fn(
          task_lib.exec_node_task_id_from_node(self._pipeline, node)):
        continue

      result.extend(self._generate_tasks_for_node(self._mlmd_handle, node))
    return result

  def _generate_tasks_for_node(
      self, metadata_handler: metadata.Metadata,
      node: node_proto_view.NodeProtoView) -> List[task_lib.Task]:
    """Generates a node execution task.

    If a node execution is not feasible, `None` is returned.

    Args:
      metadata_handler: A handler to access MLMD db.
      node: The pipeline node for which to generate a task.

    Returns:
      Returns a `Task` or `None` if task generation is deemed infeasible.
    """
    result = []
    node_uid = task_lib.NodeUid.from_node(self._pipeline, node)

    # Gets the oldest active execution. If the oldest active execution exists,
    # generates a task from it.
    # TODO(b/239858201) Too many executions may have performance issue, it is
    # better to limit the number of executions.
    executions = task_gen_utils.get_executions(metadata_handler, node)
    oldest_active_execution = task_gen_utils.get_oldest_active_execution(
        executions)
    if oldest_active_execution:
      with mlmd_state.mlmd_execution_atomic_op(
          mlmd_handle=self._mlmd_handle,
          execution_id=oldest_active_execution.id) as execution:
        execution.last_known_state = metadata_store_pb2.Execution.RUNNING
      result.append(
          task_lib.UpdateNodeStateTask(
              node_uid=node_uid, state=pstate.NodeState.RUNNING))
      result.append(
          task_gen_utils.generate_task_from_execution(self._mlmd_handle,
                                                      self._pipeline, node,
                                                      oldest_active_execution))
      return result

    resolved_info = task_gen_utils.generate_resolved_info(
        metadata_handler, node)

    # Note that some nodes e.g. ImportSchemaGen don't have inputs, and for those
    # nodes it is okay that there are no resolved input artifacts.
    if ((resolved_info is None or not resolved_info.input_and_params or
         resolved_info.input_and_params[0] is None or
         resolved_info.input_and_params[0].input_artifacts is None) or
        (node.inputs.inputs and
         not any(resolved_info.input_and_params[0].input_artifacts.values()))):
      logging.info(
          'Task cannot be generated for node %s since no input artifacts '
          'are resolved.', node.node_info.id)
      return result

    successful_executions = [
        e for e in executions if execution_lib.is_execution_successful(e)
    ]
    execution_identifiers = []
    for execution in successful_executions:
      execution_identifier = {}
      artifact_ids_by_event_type = (
          execution_lib.get_artifact_ids_by_event_type_for_execution_id(
              metadata_handler, execution.id))
      # TODO(b/250075208) Support notrigger, trigger_by, etc.
      execution_identifier['artifact_ids'] = artifact_ids_by_event_type.get(
          metadata_store_pb2.Event.INPUT, set())
      execution_identifiers.append(execution_identifier)

    unprocessed_inputs = []
    for input_and_param in resolved_info.input_and_params:
      current_exec_input_artifact_ids = set(
          a.id
          for a in itertools.chain(*input_and_param.input_artifacts.values()))

      for execution_identifier in execution_identifiers:
        if (execution_identifier['artifact_ids'] ==
            current_exec_input_artifact_ids):
          break
      else:
        unprocessed_inputs.append(input_and_param)

    if not unprocessed_inputs:
      result.append(
          task_lib.UpdateNodeStateTask(
              node_uid=node_uid, state=pstate.NodeState.STARTED))
      return result

    # Don't need to register all executions in one transaction. If the
    # orchestractor is interrupted in the middle, in the next iteration, the
    # orchstrator is still able to register the rest of executions.
    new_executioins = []
    for input_and_param in unprocessed_inputs:
      new_executioins.append(
          execution_publish_utils.register_execution(
              metadata_handler=metadata_handler,
              execution_type=node.node_info.type,
              contexts=resolved_info.contexts,
              input_artifacts=input_and_param.input_artifacts,
              exec_properties=input_and_param.exec_properties,
              last_known_state=metadata_store_pb2.Execution.NEW))

    # Selects the first artifacts and create a exec task.
    input_and_param = unprocessed_inputs[0]
    execution = new_executioins[0]
    # Selects the first execution and marks it as RUNNING.
    with mlmd_state.mlmd_execution_atomic_op(
        mlmd_handle=self._mlmd_handle, execution_id=execution.id) as execution:
      execution.last_known_state = metadata_store_pb2.Execution.RUNNING

    outputs_resolver = outputs_utils.OutputsResolver(
        node, self._pipeline.pipeline_info, self._pipeline.runtime_spec,
        self._pipeline.execution_mode)
    output_artifacts = outputs_resolver.generate_output_artifacts(execution.id)
    outputs_utils.make_output_dirs(output_artifacts)
    result.append(
        task_lib.UpdateNodeStateTask(
            node_uid=node_uid, state=pstate.NodeState.RUNNING))
    result.append(
        task_lib.ExecNodeTask(
            node_uid=node_uid,
            execution_id=execution.id,
            contexts=resolved_info.contexts,
            input_artifacts=input_and_param.input_artifacts,
            exec_properties=input_and_param.exec_properties,
            output_artifacts=output_artifacts,
            executor_output_uri=outputs_resolver.get_executor_output_uri(
                execution.id),
            stateful_working_dir=outputs_resolver
            .get_stateful_working_directory(execution.id),
            tmp_dir=outputs_resolver.make_tmp_dir(execution.id),
            pipeline=self._pipeline))
    return result

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


def _exec_properties_match(
    exec_props1: Dict[str, types.ExecPropertyTypes],
    exec_props2: Dict[str, types.ExecPropertyTypes]) -> bool:
  """Returns True if exec properties match."""

  def _filter_out_internal_keys(
      props: Dict[str, types.ExecPropertyTypes]
  ) -> Dict[str, types.ExecPropertyTypes]:
    return {
        k: v for k, v in props.items() if not execution_lib.is_internal_key(k)
    }

  exec_props1 = _filter_out_internal_keys(exec_props1)
  exec_props2 = _filter_out_internal_keys(exec_props2)
  return exec_props1 == exec_props2
