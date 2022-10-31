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
from typing import List, Optional

from absl import logging
from tfx.dsl.io import fileio
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import constants
from tfx.orchestration.experimental.core import garbage_collection
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import pipeline_state
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_scheduler as ts
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import outputs_utils
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
    _remove_output_dirs(task)
    _remove_task_dirs(
        stateful_working_dir=task.stateful_working_dir,
        tmp_dir=task.tmp_dir,
        executor_output_uri=task.executor_output_uri)
    if status.code == status_lib.Code.CANCELLED:
      logging.info('Cancelling execution (id: %s); task id: %s; status: %s',
                   task.execution_id, task.task_id, status)
      execution_state = proto.Execution.CANCELED
    else:
      logging.info(
          'Aborting execution (id: %s) due to error (code: %s); task id: %s',
          task.execution_id, status.code, task.task_id)
      execution_state = proto.Execution.FAILED
    _update_execution_state_in_mlmd(mlmd_handle, task.execution_id,
                                    execution_state, status.message,
                                    execution_result)
    pipeline_state.record_state_change_time()

  if result.status.code != status_lib.Code.OK:
    _update_state(result.status)
    return

  # TODO(b/182316162): Unify publisher handing so that post-execution artifact
  # logic is more cleanly handled.
  outputs_utils.tag_output_artifacts_with_version(task.output_artifacts)
  if isinstance(result.output, ts.ExecutorNodeOutput):
    executor_output = result.output.executor_output
    if executor_output is not None:
      if executor_output.execution_result.code != status_lib.Code.OK:
        _update_state(
            status_lib.Status(
                # We should not reuse "execution_result.code" because it may be
                # CANCELLED, in which case we should still fail the execution.
                code=status_lib.Code.ABORTED,
                message=executor_output.execution_result.result_message),
            executor_output.execution_result)
        return
      # TODO(b/182316162): Unify publisher handing so that post-execution
      # artifact logic is more cleanly handled.
      outputs_utils.tag_executor_output_with_version(executor_output)
    _remove_task_dirs(
        stateful_working_dir=task.stateful_working_dir,
        tmp_dir=task.tmp_dir,
        executor_output_uri=task.executor_output_uri)
    execution_publish_utils.publish_succeeded_execution(
        mlmd_handle,
        execution_id=task.execution_id,
        contexts=task.contexts,
        output_artifacts=task.output_artifacts,
        executor_output=executor_output)
    garbage_collection.run_garbage_collection_for_node(mlmd_handle,
                                                       task.node_uid,
                                                       task.get_node())
  elif isinstance(result.output, ts.ImporterNodeOutput):
    output_artifacts = result.output.output_artifacts
    # TODO(b/182316162): Unify publisher handing so that post-execution artifact
    # logic is more cleanly handled.
    outputs_utils.tag_output_artifacts_with_version(output_artifacts)
    _remove_task_dirs(
        stateful_working_dir=task.stateful_working_dir,
        tmp_dir=task.tmp_dir,
        executor_output_uri=task.executor_output_uri)
    execution_publish_utils.publish_succeeded_execution(
        mlmd_handle,
        execution_id=task.execution_id,
        contexts=task.contexts,
        output_artifacts=output_artifacts)
  elif isinstance(result.output, ts.ResolverNodeOutput):
    resolved_input_artifacts = result.output.resolved_input_artifacts
    execution_publish_utils.publish_internal_execution(
        mlmd_handle,
        execution_id=task.execution_id,
        contexts=task.contexts,
        output_artifacts=resolved_input_artifacts)
  else:
    raise TypeError(f'Unable to process task scheduler result: {result}')

  pipeline_state.record_state_change_time()


def publish_execution_results(
    mlmd_handle: metadata.Metadata,
    executor_output: execution_result_pb2.ExecutorOutput,
    execution_info: data_types.ExecutionInfo,
    contexts: List[proto.Context]) -> Optional[typing_utils.ArtifactMultiMap]:
  """Publishes execution result to MLMD for single component run."""
  outputs_utils.tag_output_artifacts_with_version(execution_info.output_dict)
  if executor_output.execution_result.code != status_lib.Code.OK:
    outputs_utils.remove_output_dirs(execution_info.output_dict)
    _remove_task_dirs(
        stateful_working_dir=execution_info.stateful_working_dir,
        tmp_dir=execution_info.tmp_dir,
        executor_output_uri=execution_info.execution_output_uri)
    _update_execution_state_in_mlmd(
        mlmd_handle=mlmd_handle,
        execution_id=execution_info.execution_id,
        new_state=proto.Execution.FAILED,
        error_msg=executor_output.execution_result.result_message,
        execution_result=executor_output.execution_result)
    return
  # TODO(b/182316162): Unify publisher handing so that post-execution
  # artifact logic is more cleanly handled.
  outputs_utils.tag_executor_output_with_version(executor_output)
  _remove_task_dirs(
      stateful_working_dir=execution_info.stateful_working_dir,
      tmp_dir=execution_info.tmp_dir,
      executor_output_uri=execution_info.execution_output_uri)
  return execution_publish_utils.publish_succeeded_execution(
      mlmd_handle,
      execution_id=execution_info.execution_id,
      contexts=contexts,
      output_artifacts=execution_info.output_dict,
      executor_output=executor_output)


def _update_execution_state_in_mlmd(
    mlmd_handle: metadata.Metadata,
    execution_id: int,
    new_state: proto.Execution.State,
    error_msg: str,
    execution_result: Optional[execution_result_pb2.ExecutionResult] = None
) -> None:
  """Updates the execution state and sets execution_result if provided."""
  with mlmd_state.mlmd_execution_atomic_op(mlmd_handle,
                                           execution_id) as execution:
    execution.last_known_state = new_state
    if error_msg:
      data_types_utils.set_metadata_value(
          execution.custom_properties[constants.EXECUTION_ERROR_MSG_KEY],
          error_msg)
    if execution_result:
      execution_lib.set_execution_result(execution_result, execution)


def _remove_output_dirs(task: task_lib.ExecNodeTask) -> None:
  outputs_utils.remove_output_dirs(task.output_artifacts)


def _remove_task_dirs(stateful_working_dir: str = '',
                      tmp_dir: str = '',
                      executor_output_uri: str = '') -> None:
  """Removes directories created for the task."""
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
  if executor_output_uri:
    try:
      fileio.remove(executor_output_uri)
    except fileio.NotFoundError:
      logging.warning(
          'Skipping deletion of executor_output_uri (file not found): %s',
          executor_output_uri)
