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
"""Utils for publishing execution results."""
from __future__ import annotations

from typing import Optional

from absl import logging
from tfx.dsl.io import fileio
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import component_generated_alert_pb2
from tfx.orchestration.experimental.core import constants
from tfx.orchestration.experimental.core import event_observer
from tfx.orchestration.experimental.core import garbage_collection
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_scheduler as ts
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import execution_result_pb2
from tfx.utils import status as status_lib
from tfx.utils import typing_utils

from ml_metadata import proto


def publish_execution_results_for_task(mlmd_handle: metadata.Metadata,
                                       task: task_lib.ExecNodeTask,
                                       result: ts.TaskSchedulerResult) -> None:
  """Publishes execution results to MLMD for task."""

  def _update_state(
      status: status_lib.Status,
      execution_result: Optional[execution_result_pb2.ExecutionResult] = None
  ) -> None:
    assert status.code != status_lib.Code.OK
    _remove_temporary_task_dirs(tmp_dir=task.tmp_dir)
    if status.code == status_lib.Code.CANCELLED and execution_result is None:
      # Mark the execution as cancelled only if the task was cancelled by the
      # task scheduler, and not by the executor.
      logging.info('Cancelling execution (id: %s); task id: %s; status: %s',
                   task.execution_id, task.task_id, status)
      execution_state = proto.Execution.CANCELED
    else:
      logging.info(
          'Aborting execution (id: %s) due to error (code: %s); task id: %s',
          task.execution_id, status.code, task.task_id)
      execution_state = proto.Execution.FAILED
    _update_execution_state_in_mlmd(
        mlmd_handle=mlmd_handle,
        node_uid=task.node_uid,
        execution_id=task.execution_id,
        new_state=execution_state,
        error_code=status.code,
        error_msg=status.message,
        execution_result=execution_result)

  if result.status.code != status_lib.Code.OK:
    _update_state(result.status)
    return

  if isinstance(result.output, ts.ExecutorNodeOutput):
    executor_output = result.output.executor_output
    if executor_output is not None:
      if executor_output.execution_result.code != status_lib.Code.OK:
        _update_state(
            status_lib.Status(
                code=executor_output.execution_result.code,
                message=executor_output.execution_result.result_message),
            executor_output.execution_result)
        return
    _remove_temporary_task_dirs(
        stateful_working_dir=task.stateful_working_dir, tmp_dir=task.tmp_dir)
    # TODO(b/262040844): Instead of directly using the context manager here, we
    # should consider creating and using wrapper functions.
    with mlmd_state.evict_from_cache(task.execution_id):
      _, execution = execution_publish_utils.publish_succeeded_execution(
          mlmd_handle,
          execution_id=task.execution_id,
          contexts=task.contexts,
          output_artifacts=task.output_artifacts,
          executor_output=executor_output)
    garbage_collection.run_garbage_collection_for_node(mlmd_handle,
                                                       task.node_uid,
                                                       task.get_node())
    if constants.COMPONENT_GENERATED_ALERTS_KEY in execution.custom_properties:
      alerts_proto = component_generated_alert_pb2.ComponentGeneratedAlertList()
      execution.custom_properties[
          constants.COMPONENT_GENERATED_ALERTS_KEY
      ].proto_value.Unpack(alerts_proto)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline=task.pipeline)

      for alert in alerts_proto.component_generated_alert_list:
        alert_event = event_observer.ComponentGeneratedAlert(
            execution=execution,
            pipeline_uid=pipeline_uid,
            pipeline_run=pipeline_uid.pipeline_run_id,
            node_id=task.node_uid.node_id,
            alert_body=alert.alert_body,
            alert_name=alert.alert_name,
        )
        event_observer.notify(alert_event)

  elif isinstance(result.output, ts.ImporterNodeOutput):
    output_artifacts = result.output.output_artifacts
    _remove_temporary_task_dirs(
        stateful_working_dir=task.stateful_working_dir, tmp_dir=task.tmp_dir)
    # TODO(b/262040844): Instead of directly using the context manager here, we
    # should consider creating and using wrapper functions.
    with mlmd_state.evict_from_cache(task.execution_id):
      execution_publish_utils.publish_succeeded_execution(
          mlmd_handle,
          execution_id=task.execution_id,
          contexts=task.contexts,
          output_artifacts=output_artifacts)
  elif isinstance(result.output, ts.ResolverNodeOutput):
    resolved_input_artifacts = result.output.resolved_input_artifacts
    # TODO(b/262040844): Instead of directly using the context manager here, we
    # should consider creating and using wrapper functions.
    with mlmd_state.evict_from_cache(task.execution_id):
      execution_publish_utils.publish_internal_execution(
          mlmd_handle,
          execution_id=task.execution_id,
          contexts=task.contexts,
          output_artifacts=resolved_input_artifacts)
  else:
    raise TypeError(f'Unable to process task scheduler result: {result}')


def publish_execution_results(
    mlmd_handle: metadata.Metadata,
    executor_output: execution_result_pb2.ExecutorOutput,
    execution_info: data_types.ExecutionInfo,
    contexts: list[proto.Context]) -> Optional[typing_utils.ArtifactMultiMap]:
  """Publishes execution result to MLMD for single component run."""
  if executor_output.execution_result.code != status_lib.Code.OK:
    if executor_output.execution_result.code == status_lib.Code.CANCELLED:
      execution_state = proto.Execution.CANCELED
    else:
      execution_state = proto.Execution.FAILED
    _remove_temporary_task_dirs(tmp_dir=execution_info.tmp_dir)
    node_uid = task_lib.NodeUid(
        pipeline_uid=task_lib.PipelineUid.from_pipeline_id_and_run_id(
            pipeline_id=execution_info.pipeline_info.id,
            pipeline_run_id=execution_info.pipeline_run_id),
        node_id=execution_info.pipeline_node.node_info.id)
    _update_execution_state_in_mlmd(
        mlmd_handle=mlmd_handle,
        node_uid=node_uid,
        execution_id=execution_info.execution_id,
        new_state=execution_state,
        error_code=executor_output.execution_result.code,
        error_msg=executor_output.execution_result.result_message,
        execution_result=executor_output.execution_result)
    return
  _remove_temporary_task_dirs(
      stateful_working_dir=execution_info.stateful_working_dir,
      tmp_dir=execution_info.tmp_dir)
  # TODO(b/262040844): Instead of directly using the context manager here, we
  # should consider creating and using wrapper functions.
  with mlmd_state.evict_from_cache(execution_info.execution_id):
    output_dict, _ = execution_publish_utils.publish_succeeded_execution(
        mlmd_handle,
        execution_id=execution_info.execution_id,
        contexts=contexts,
        output_artifacts=execution_info.output_dict,
        executor_output=executor_output)
    return output_dict


def _update_execution_state_in_mlmd(
    mlmd_handle: metadata.Metadata,
    node_uid: task_lib.NodeUid,
    execution_id: int,
    new_state: proto.Execution.State,
    error_code: int,
    error_msg: str,
    execution_result: Optional[execution_result_pb2.ExecutionResult] = None,
) -> None:
  """Updates the execution state and sets execution_result if provided."""
  with mlmd_state.mlmd_execution_atomic_op(
      mlmd_handle,
      execution_id,
      on_commit=event_observer.make_notify_execution_state_change_fn(
          node_uid)) as execution:
    execution.last_known_state = new_state
    data_types_utils.set_metadata_value(
        execution.custom_properties[constants.EXECUTION_ERROR_CODE_KEY],
        error_code,
    )
    if error_msg:
      data_types_utils.set_metadata_value(
          execution.custom_properties[constants.EXECUTION_ERROR_MSG_KEY],
          error_msg)
    if execution_result:
      execution_lib.set_execution_result(execution_result, execution)


def _remove_temporary_task_dirs(stateful_working_dir: str = '',
                                tmp_dir: str = '') -> None:
  """Removes temporary directories created for the task."""
  if stateful_working_dir:
    try:
      fileio.rmtree(stateful_working_dir)
    except fileio.NotFoundError:
      logging.warning('stateful_working_dir %s not found, ignoring.',
                      stateful_working_dir)
  if tmp_dir:
    try:
      fileio.rmtree(tmp_dir)
    except fileio.NotFoundError:
      logging.warning(
          'tmp_dir %s not found while attempting to delete, ignoring.')
