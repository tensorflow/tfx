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
from typing import Callable, Optional

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
    self._sub_pipeline = subpipeline_ir_rewrite(pipeline_node.raw_proto(),
                                                task.execution_id)
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


def _visit_pipeline_nodes_recursively(
    p: pipeline_pb2.Pipeline, visitor: Callable[[pipeline_pb2.PipelineNode],
                                                None]):
  """Helper function to visit every node inside a possibly nested pipeline."""
  for pipeline_or_node in p.nodes:
    if pipeline_or_node.WhichOneof('node') == 'pipeline_node':
      visitor(pipeline_or_node.pipeline_node)
    else:
      _visit_pipeline_nodes_recursively(pipeline_or_node.sub_pipeline, visitor)


def _update_pipeline_run_id(pipeline: pipeline_pb2.Pipeline, execution_id: int):
  """Rewrites pipeline run id in a given pipeline IR."""
  old_pipeline_run_id = pipeline.runtime_spec.pipeline_run_id.field_value.string_value
  new_pipeline_run_id = old_pipeline_run_id + f'_{execution_id}'

  def _node_updater(node: pipeline_pb2.PipelineNode):
    for context_spec in node.contexts.contexts:
      if (context_spec.type.name == 'pipeline_run' and
          context_spec.name.field_value.string_value == old_pipeline_run_id):
        context_spec.name.field_value.string_value = new_pipeline_run_id
    for input_spec in node.inputs.inputs.values():
      for channel in input_spec.channels:
        for context_query in channel.context_queries:
          if (context_query.type.name == 'pipeline_run' and
              context_query.name.field_value.string_value
              == old_pipeline_run_id):
            context_query.name.field_value.string_value = new_pipeline_run_id

  _visit_pipeline_nodes_recursively(pipeline, _node_updater)
  pipeline.runtime_spec.pipeline_run_id.field_value.string_value = new_pipeline_run_id


def subpipeline_ir_rewrite(original_ir: pipeline_pb2.Pipeline,
                           execution_id: int) -> pipeline_pb2.Pipeline:
  """Rewrites the subpipeline IR so that it can be run independently.

  Args:
    original_ir: Original subpipeline IR that is produced by compiler.
    execution_id: The ID of Subpipeline task scheduler Execution. It is used to
      generated a new pipeline run id.

  Returns:
    An updated subpipeline IR that can be run independently.
  """
  pipeline = copy.deepcopy(original_ir)
  pipeline.nodes[0].pipeline_node.ClearField('upstream_nodes')
  pipeline.nodes[-1].pipeline_node.ClearField('downstream_nodes')
  _update_pipeline_run_id(pipeline, execution_id)
  return pipeline
