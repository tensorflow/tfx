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

import sys
import traceback
from typing import Callable, List, Optional

from absl import logging
from tfx.orchestration import metadata
from tfx.orchestration import node_proto_view
from tfx.orchestration.experimental.core import constants
from tfx.orchestration.experimental.core import event_observer
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration import mlmd_connection_manager as mlmd_cm
from tfx.orchestration.portable.input_resolution import exceptions
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

  def __init__(self, mlmd_connection_manager: mlmd_cm.MLMDConnectionManager,
               is_task_id_tracked_fn: Callable[[task_lib.TaskId], bool],
               service_job_manager: service_jobs.ServiceJobManager):
    """Constructs `AsyncPipelineTaskGenerator`.

    Args:
      mlmd_connection_manager: A `MLMDConnectionManager` instance to manager
        multiple mlmd connections.
      is_task_id_tracked_fn: A callable that returns `True` if a task_id is
        tracked by the task queue.
      service_job_manager: Used for handling service nodes in the pipeline.
    """
    self._mlmd_connection_manager = mlmd_connection_manager
    self._is_task_id_tracked_fn = is_task_id_tracked_fn
    self._service_job_manager = service_job_manager

  def generate(
      self, pipeline_state: pstate.PipelineState
  ) -> List[task_lib.Task]:
    """Generates tasks for all executable nodes in the async pipeline.

    The returned tasks must have `exec_task` populated. List may be empty if no
    nodes are ready for execution.

    Args:
      pipeline_state: The `PipelineState` object associated with the pipeline
        for which to generate tasks.

    Returns:
      A `list` of tasks to execute.
    """
    return _Generator(self._mlmd_connection_manager, pipeline_state,
                      self._is_task_id_tracked_fn, self._service_job_manager)()


class _Generator:
  """Generator implementation class for AsyncPipelineTaskGenerator."""

  def __init__(self, mlmd_connection_manager: mlmd_cm.MLMDConnectionManager,
               pipeline_state: pstate.PipelineState,
               is_task_id_tracked_fn: Callable[[task_lib.TaskId], bool],
               service_job_manager: service_jobs.ServiceJobManager):
    self._mlmd_connection_manager = mlmd_connection_manager
    self._mlmd_handle = mlmd_connection_manager.primary_mlmd_handle
    pipeline = pipeline_state.pipeline
    if pipeline.execution_mode != pipeline_pb2.Pipeline.ExecutionMode.ASYNC:
      raise ValueError(
          'AsyncPipelineTaskGenerator should be instantiated with a pipeline '
          'proto having execution mode `ASYNC`, not `{}`'.format(
              pipeline.execution_mode))
    self._pipeline_state = pipeline_state
    self._pipeline = pipeline
    self._is_task_id_tracked_fn = is_task_id_tracked_fn
    self._service_job_manager = service_job_manager

  def __call__(self) -> List[task_lib.Task]:
    result = []
    for node in [node_proto_view.get_view(n) for n in self._pipeline.nodes]:
      node_uid = task_lib.NodeUid.from_node(self._pipeline, node)
      node_id = node.node_info.id

      logging.info(
          '[AsyncPipelineTaskGenerator._generate_tasks_for_node] generating'
          ' tasks for node %s',
          node_id,
      )

      with self._pipeline_state:
        node_state = self._pipeline_state.get_node_state(node_uid)
        if node_state.state in (pstate.NodeState.STOPPING,
                                pstate.NodeState.STOPPED,
                                pstate.NodeState.FAILED):
          logging.info('Ignoring node in state \'%s\' for task generation: %s',
                       node_state.state, node_uid)
          continue

      # If this is a pure service node, there is no ExecNodeTask to generate
      # but we ensure node services and check service status.
      service_status = self._ensure_node_services_if_pure(
          node_id, node_state.backfill_token
      )
      if service_status is not None:
        if (
            node_state.backfill_token
            and service_status.code == service_jobs.ServiceStatusCode.SUCCESS
        ):
          # Transitions ExampleGen node to STOPPED state and service job to
          # STATE_STOPPED when backfill completes.
          logging.info(
              'Stopping ExampleGen: %s ; Backfill with token: %s completed',
              node_id,
              node_state.backfill_token,
          )
          result.append(
              task_lib.UpdateNodeStateTask(
                  node_uid=node_uid,
                  state=pstate.NodeState.STOPPED,
                  backfill_token='',
              )
          )
          # The service job already completes with success but we still need to
          # update the in-memory state.
          self._service_job_manager.stop_node_services(
              self._pipeline_state, node_id
          )
        elif service_status.code != service_jobs.ServiceStatusCode.RUNNING:
          error_msg = f'service job failed; error message: {service_status.msg}'
          result.append(
              task_lib.UpdateNodeStateTask(
                  node_uid=node_uid,
                  state=pstate.NodeState.FAILED,
                  status=status_lib.Status(
                      code=status_lib.Code.UNKNOWN, message=error_msg
                  ),
                  backfill_token='',
              )
          )
        elif node_state.state != pstate.NodeState.RUNNING:
          result.append(
              task_lib.UpdateNodeStateTask(
                  node_uid=node_uid,
                  state=pstate.NodeState.RUNNING,
                  backfill_token=node_state.backfill_token,
              )
          )
        continue

      # For mixed service nodes, we ensure node services and check service
      # status; the node is aborted if its service jobs have failed.
      service_status = self._ensure_node_services_if_mixed(node.node_info.id)
      if service_status is not None:
        if service_status.code != service_jobs.ServiceStatusCode.RUNNING:
          error_msg = (
              f'associated service job failed; node uid: {node_uid}; error'
              f' message: {service_status.msg}'
          )
          result.append(
              task_lib.UpdateNodeStateTask(
                  node_uid=node_uid,
                  state=pstate.NodeState.FAILED,
                  status=status_lib.Status(
                      code=status_lib.Code.UNKNOWN, message=error_msg)))
          continue

      # If a task for the node is already tracked by the task queue, it need
      # not be considered for generation again.
      if self._is_task_id_tracked_fn(
          task_lib.exec_node_task_id_from_node(self._pipeline, node)):
        continue

      tasks = self._generate_tasks_for_node(
          self._mlmd_handle, node, node_state.backfill_token
      )
      logging.info(
          '[AsyncPipelineTaskGenerator._generate_tasks_for_node] generated'
          ' tasks for node %s: %s',
          node.node_info.id,
          [t.task_id for t in tasks],
      )
      result.extend(tasks)
    return result

  def _generate_tasks_for_node(
      self,
      metadata_handle: metadata.Metadata,
      node: node_proto_view.NodeProtoView,
      backfill_token: str,
  ) -> List[task_lib.Task]:
    """Generates a node execution task.

    If a node execution is not feasible, `None` is returned.

    Args:
      metadata_handle: A handler to access MLMD db.
      node: The pipeline node for which to generate a task.
      backfill_token: Backfill token, if applicable.

    Returns:
      Returns a `Task` or `None` if task generation is deemed infeasible.
    """
    result = []
    node_uid = task_lib.NodeUid.from_node(self._pipeline, node)

    # Gets the active executions. If the active executions exist, generates a
    # task from the oldest active execution.
    active_executions = task_gen_utils.get_executions(
        metadata_handle,
        node,
        additional_filters=['last_known_state IN (NEW, RUNNING)'],
    )
    next_active_execution_to_run = (
        task_gen_utils.get_next_active_execution_to_run(active_executions)
    )
    if next_active_execution_to_run:
      if backfill_token:
        if (
            next_active_execution_to_run.custom_properties[
                constants.BACKFILL_TOKEN_CUSTOM_PROPERTY_KEY
            ].string_value
            != backfill_token
        ):
          logging.warning(
              (
                  'Node %s is in backfill mode, but there are active executions'
                  ' that are not for backfill token %s. Oldest active execution'
                  ' was: %s. Aborting backfill and setting node to STOPPED'
                  ' state'
              ),
              node.node_info.id,
              backfill_token,
              next_active_execution_to_run,
          )
          result.append(
              task_lib.UpdateNodeStateTask(
                  node_uid=node_uid,
                  state=pstate.NodeState.STOPPED,
                  status=status_lib.Status(
                      code=status_lib.Code.FAILED_PRECONDITION,
                      message=(
                          f'Node {node.node_info.id} has active executions that'
                          f' are not for backfill token {backfill_token}.'
                          ' Oldest active execution was'
                          f' {next_active_execution_to_run}'
                      ),
                  ),
                  backfill_token='',
              )
          )
          return result

      with mlmd_state.mlmd_execution_atomic_op(
          mlmd_handle=self._mlmd_handle,
          execution_id=next_active_execution_to_run.id,
          on_commit=event_observer.make_notify_execution_state_change_fn(
              node_uid
          ),
      ) as execution:
        execution.last_known_state = metadata_store_pb2.Execution.RUNNING
      result.append(
          task_lib.UpdateNodeStateTask(
              node_uid=node_uid,
              state=pstate.NodeState.RUNNING,
              backfill_token=backfill_token,
          )
      )
      result.append(
          task_gen_utils.generate_task_from_execution(
              self._mlmd_handle,
              self._pipeline,
              node,
              next_active_execution_to_run,
          )
      )
      return result

    with self._pipeline_state:
      node_state = self._pipeline_state.get_node_state(node_uid)
    if not backfill_token and node_state.state != pstate.NodeState.STARTED:
      # If there is no active execution, change the node state to STARTED.
      result.append(
          task_lib.UpdateNodeStateTask(
              node_uid=node_uid,
              state=pstate.NodeState.STARTED,
              backfill_token=backfill_token,
          )
      )

    if backfill_token and (
        newest_executions := task_gen_utils.get_executions(
            metadata_handle, node, limit=1
        )
    ):
      newest_execution = newest_executions[0]
      # If we are backfilling, we only want to do input resolution once,
      # and register the executions once. To check if we've already registered
      # the executions, we check for the existence of executions with the
      # backfill token. Note that this can be incorrect in rare cases until
      # b/266014070 is resolved.
      if (
          newest_execution.custom_properties[
              constants.BACKFILL_TOKEN_CUSTOM_PROPERTY_KEY
          ].string_value
          == backfill_token
      ):
        logging.info(
            'Backfill of node %s is complete. Setting node to STOPPED state',
            node.node_info.id,
        )
        result.append(
            task_lib.UpdateNodeStateTask(
                node_uid=node_uid,
                state=pstate.NodeState.STOPPED,
                backfill_token='',
            )
        )
        return result

    try:
      resolved_info = task_gen_utils.generate_resolved_info(
          mlmd_handle_like=self._mlmd_connection_manager,
          node=node,
          pipeline=self._pipeline,
          skip_errors=[exceptions.InsufficientInputError],
      )
    except exceptions.InputResolutionError:
      error_msg = (
          f'failure to resolve inputs; node uid: {node_uid}; '
          f'error: {traceback.format_exception(*sys.exc_info(), limit=0)}'
      )
      if backfill_token:
        logging.exception(
            'InputResolutionError raised when resolving input artifacts for'
            ' node %s during backfill. Setting node to FAILED state with status'
            ' code FAILED_PRECONDITION.',
            node.node_info.id,
        )
        result.append(
            task_lib.UpdateNodeStateTask(
                node_uid=node_uid,
                state=pstate.NodeState.FAILED,
                status=status_lib.Status(
                    code=status_lib.Code.FAILED_PRECONDITION,
                    message=(
                        f'Backfill of node {node.node_info.id} failed'
                        f' Error: {error_msg}'
                    ),
                ),
                backfill_token='',
            )
        )
      else:
        logging.exception(
            'InputResolutionError raised when resolving input artifacts for'
            ' node %s. Setting node to STARTED state with status code'
            ' UNAVAILABLE.',
            node.node_info.id,
        )
        result.append(
            task_lib.UpdateNodeStateTask(
                node_uid=node_uid,
                state=pstate.NodeState.STARTED,
                status=status_lib.Status(
                    code=status_lib.Code.UNAVAILABLE, message=error_msg
                ),
            )
        )
      return result

    # Note that some nodes e.g. ImportSchemaGen don't have inputs, and for those
    # nodes it is okay that there are no resolved input artifacts.
    if ((resolved_info is None or not resolved_info.input_and_params or
         resolved_info.input_and_params[0] is None or
         resolved_info.input_and_params[0].input_artifacts is None) or
        (node.inputs.inputs and
         not any(resolved_info.input_and_params[0].input_artifacts.values()))):
      if backfill_token:
        error_msg = (
            f'Backfill of node {node.node_info.id} resvoled no input artifacts'
        )
        logging.info(
            (
                'Backfill of node %s resolved no input artifacts. Setting node'
                ' to STOPPED state with status code FAIL_PRECONDITION.'
                ' Error: %s'
            ),
            node.node_info.id,
            error_msg,
        )
        result.append(
            task_lib.UpdateNodeStateTask(
                node_uid=node_uid,
                state=pstate.NodeState.STOPPED,
                status=status_lib.Status(
                    code=status_lib.Code.FAILED_PRECONDITION,
                    message=error_msg,
                ),
                backfill_token='',
            )
        )
      else:
        logging.info(
            'No input artifacts resolved for node %s. Setting node to STARTED'
            ' state with OK status.',
            node.node_info.id,
        )
        result.append(
            task_lib.UpdateNodeStateTask(
                node_uid=node_uid,
                state=pstate.NodeState.STARTED,
                status=status_lib.Status(
                    code=status_lib.Code.OK,
                    message=(
                        'Waiting for new input artifacts to be processed.'
                        ' Non-triggering input or insufficient number of'
                        ' artifacts will not trigger new execution.'
                    ),
                ),
            )
        )

      return result

    # Copys artifact types of the external artifacts to local db, in idempotent
    # manner. Idempotency is guaranteed by the artifact type name.
    # The external artifacts will be copies to local db when we register
    # executions. Idempotency is guaranteed by external_id.
    updated_external_artifacts = []
    for input_and_params in resolved_info.input_and_params:
      for artifacts in input_and_params.input_artifacts.values():
        updated_external_artifacts.extend(
            task_gen_utils.update_external_artifact_type(
                self._mlmd_handle, artifacts
            )
        )
    if updated_external_artifacts:
      logging.info(
          'Updated external artifacts: %s',
          [a.id for a in updated_external_artifacts],
      )

    if backfill_token:
      # For backfills, ignore all previous executions.
      unprocessed_inputs = resolved_info.input_and_params
    else:
      unprocessed_inputs = task_gen_utils.get_unprocessed_inputs(
          metadata_handle, resolved_info, node
      )
    if not unprocessed_inputs:
      return result

    for input_and_param in unprocessed_inputs:
      if backfill_token:
        input_and_param.exec_properties[
            constants.BACKFILL_TOKEN_CUSTOM_PROPERTY_KEY
        ] = backfill_token

    execution_state_change_fn = (
        event_observer.make_notify_execution_state_change_fn(node_uid)
    )
    executions = task_gen_utils.register_executions(
        metadata_handle=metadata_handle,
        execution_type=node.node_info.type,
        contexts=resolved_info.contexts,
        input_and_params=unprocessed_inputs,
    )

    for execution in executions:
      execution_state_change_fn(None, execution)

    result.extend(
        task_gen_utils.generate_tasks_from_one_input(
            metadata_handle=metadata_handle,
            node=node,
            execution=executions[0],
            input_and_param=unprocessed_inputs[0],
            contexts=resolved_info.contexts,
            pipeline=self._pipeline,
            execution_node_state=pstate.NodeState.RUNNING,
            backfill_token=backfill_token,
            execution_commit_fn=execution_state_change_fn,
        )
    )
    return result

  def _ensure_node_services_if_pure(
      self, node_id: str, backfill_token: str
  ) -> Optional[service_jobs.ServiceStatus]:
    """Calls `ensure_node_services` and returns status if given node is pure service node."""
    if self._service_job_manager.is_pure_service_node(self._pipeline_state,
                                                      node_id):
      return self._service_job_manager.ensure_node_services(
          self._pipeline_state, node_id, backfill_token
      )
    return None

  def _ensure_node_services_if_mixed(
      self, node_id: str) -> Optional[service_jobs.ServiceStatus]:
    """Calls `ensure_node_services` and returns status if given node is mixed service node."""
    if self._service_job_manager.is_mixed_service_node(self._pipeline_state,
                                                       node_id):
      return self._service_job_manager.ensure_node_services(
          self._pipeline_state, node_id)
    return None
