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
"""A task scheduler for subpipeline."""

import copy
import time
from typing import Optional

from absl import flags
from absl import logging
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import pipeline_ops
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_scheduler
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib

# TODO(b/242089808): Merge the polling intervals with other places.
_POLLING_INTERVAL_SECS = flags.DEFINE_float(
    'subpipeline_scheduler_polling_interval_secs', 10.0,
    'Default polling interval for subpipeline task scheduler.')


class SubPipelineTaskScheduler(
    task_scheduler.TaskScheduler[task_lib.ExecNodeTask]):
  """A task scheduler for subpipeline."""

  def __init__(self, mlmd_handle: metadata.Metadata,
               pipeline: pipeline_pb2.Pipeline, task: task_lib.ExecNodeTask):
    super().__init__(mlmd_handle, pipeline, task)
    pipeline_node = self.task.get_node()
    self._sub_pipeline = copy.deepcopy(
        _subpipeline_ir_rewrite(pipeline_node.raw_proto()))
    self._pipeline_uid = task_lib.PipelineUid.from_pipeline(self._sub_pipeline)
    self._pipeline_run_id = (
        self._sub_pipeline.runtime_spec.pipeline_run_id.field_value.string_value
    )

  def _get_pipeline_view(self) -> Optional[pstate.PipelineView]:
    try:
      return pstate.PipelineView.load(
          self.mlmd_handle,
          self._pipeline_uid.pipeline_id,
          pipeline_run_id=self._pipeline_run_id)
    except status_lib.StatusNotOkError:
      return None

  def schedule(self) -> task_scheduler.TaskSchedulerResult:
    if not self._get_pipeline_view():
      try:
        pipeline_ops.initiate_pipeline_start(self.mlmd_handle,
                                             self._sub_pipeline, None, None)
      except status_lib.StatusNotOkError as e:
        return task_scheduler.TaskSchedulerResult(status=e.status())

    while True:
      view = self._get_pipeline_view()
      if view:
        if execution_lib.is_execution_successful(view.execution):
          return task_scheduler.TaskSchedulerResult(
              status=status_lib.Status(code=status_lib.Code.OK))
        if execution_lib.is_execution_failed(
            view.execution) or execution_lib.is_execution_canceled(
                view.execution):
          return task_scheduler.TaskSchedulerResult(
              status=status_lib.Status(
                  code=status_lib.Code.UNKNOWN,
                  message='Subpipeline execution is cancelled or failed.'))
      else:
        return task_scheduler.TaskSchedulerResult(
            status=status_lib.Status(
                code=status_lib.Code.INTERNAL,
                message='Failed to find the state of a subpipeline run.'))

      logging.info('Waiting %s secs for subpipeline %s to finish.',
                   _POLLING_INTERVAL_SECS.value, self._pipeline_uid.pipeline_id)
      time.sleep(_POLLING_INTERVAL_SECS.value)

    # Should not reach here.
    raise RuntimeError(
        f'Subpipeline {self._pipeline_uid.pipeline_i} scheduling failed.')

  def cancel(self) -> None:
    pipeline_ops.stop_pipeline(self.mlmd_handle, self._pipeline_uid)


def _subpipeline_ir_rewrite(
    pipeline: pipeline_pb2.Pipeline) -> pipeline_pb2.Pipeline:
  """Rewrites the subpipeline IR so that it can be run independently.

  Clears the upstream nodes of PipelineBegin node and downstream nodes of
  PipelineEnd node.

  Args:
    pipeline: Original subpipeline IR that is produced by compiler.

  Returns:
    An updated subpipeline IR that can be run independently.
  """
  pipeline.nodes[0].pipeline_node.ClearField('upstream_nodes')
  pipeline.nodes[-1].pipeline_node.ClearField('downstream_nodes')
  return pipeline
