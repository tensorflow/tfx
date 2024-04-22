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
import threading
from typing import Callable, Optional

from absl import flags
from absl import logging
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import pipeline_ops
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_scheduler
from tfx.orchestration.portable.mlmd import context_lib
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib

from ml_metadata.proto import metadata_store_pb2
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
    self._cancel = threading.Event()
    if task.cancel_type:
      self._cancel.set()

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
    except status_lib.StatusNotOkError as e:
      logging.info(
          'Unable to load run %s for %s, probably new run. %s',
          self._pipeline_run_id,
          self._pipeline_uid.pipeline_id,
          e,
      )
      return None

  def _put_begin_node_execution(self):
    """Inserts an execution for the subpipeline begin node into MLMD.

    The new begin node execution is just forwarding the inputs to this
    subpipeline, which is possible via treaing the begin node as a Resolver,
    however because the begin node *actually* has tasks generated for it twice,
    once in the outer pipeline where the begin node is a pipeline-as-node, and
    once in the inner pipeline as a node, we don't want to regenerate tasks.

    Specifically, injecting the execution here is *required* for using ForEach,
    so that the multiple executions are only taken care of in the outer
    pipeline, and the inner pipeline only ever sees one artifact at a time from
    ForEach.
    """
    input_artifacts = self.task.input_artifacts
    begin_node = self._sub_pipeline.nodes[0].pipeline_node
    begin_node_execution = execution_lib.prepare_execution(
        metadata_handle=self.mlmd_handle,
        execution_type=begin_node.node_info.type,
        state=metadata_store_pb2.Execution.State.COMPLETE,
        exec_properties={'injected_begin_node_execution': True},
    )
    contexts = context_lib.prepare_contexts(
        metadata_handle=self.mlmd_handle,
        node_contexts=begin_node.contexts,
    )
    execution_lib.put_execution(
        metadata_handle=self.mlmd_handle,
        execution=begin_node_execution,
        contexts=contexts,
        input_artifacts=input_artifacts,
        output_artifacts=input_artifacts,
        output_event_type=metadata_store_pb2.Event.Type.INTERNAL_OUTPUT,
    )

  def _set_pipeline_execution_outputs(self):
    end_node = self._sub_pipeline.nodes[-1].pipeline_node
    end_node_contexts = context_lib.prepare_contexts(
        self.mlmd_handle, end_node.contexts
    )
    [end_node_execution] = (
        execution_lib.get_executions_associated_with_all_contexts(
            self.mlmd_handle, end_node_contexts
        )
    )
    pipeline_outputs = execution_lib.get_output_artifacts(
        self.mlmd_handle, end_node_execution.id
    )
    [pipeline_as_node_execution] = self.mlmd_handle.store.get_executions_by_id(
        [self.task.execution_id]
    )
    execution_lib.put_execution(
        metadata_handle=self.mlmd_handle,
        execution=pipeline_as_node_execution,
        contexts=self.task.contexts,
        output_artifacts=pipeline_outputs,
        output_event_type=metadata_store_pb2.Event.Type.OUTPUT,
    )

  def schedule(self) -> task_scheduler.TaskSchedulerResult:
    view = None
    if  self._cancel.is_set() or(view := self._get_pipeline_view()) is not None:
      logging.info(
          'Cancel was set OR pipeline view was not none, skipping start,'
          ' cancel.is_set(): %s, view exists: %s',
          self._cancel.is_set(),
          view is not None,
      )
    else:
      try:
        # Only create a begin node execution if we need to start the pipeline.
        # If we don't need to start the pipeline this likely means the pipeline
        # was already started so the execution should already exist.
        self._put_begin_node_execution()
        logging.info('[Subpipeline Task Scheduler]: start subpipeline.')
        pipeline_ops.initiate_pipeline_start(self.mlmd_handle,
                                             self._sub_pipeline, None, None)
      except status_lib.StatusNotOkError as e:
        return task_scheduler.TaskSchedulerResult(status=e.status())

    while not self._cancel.wait(_POLLING_INTERVAL_SECS.value):
      view = self._get_pipeline_view()
      if view:
        if execution_lib.is_execution_successful(view.execution):
          self._set_pipeline_execution_outputs()
          return task_scheduler.TaskSchedulerResult(
              status=status_lib.Status(code=status_lib.Code.OK))
        if execution_lib.is_execution_failed(view.execution):
          return task_scheduler.TaskSchedulerResult(
              status=status_lib.Status(
                  code=status_lib.Code.ABORTED,
                  message='Subpipeline execution is failed.'))
        if execution_lib.is_execution_canceled(view.execution):
          return task_scheduler.TaskSchedulerResult(
              status=status_lib.Status(
                  code=status_lib.Code.CANCELLED,
                  message='Subpipeline execution is cancelled.',
              )
          )
      else:
        return task_scheduler.TaskSchedulerResult(
            status=status_lib.Status(
                code=status_lib.Code.INTERNAL,
                message=(
                    'Failed to find the subpipeline run with run id: '
                    f'{self._pipeline_run_id}.'
                ),
            )
        )

    view = self._get_pipeline_view()
    if view and execution_lib.is_execution_active(view.execution):
      logging.info(
          '[Subpipeline Task Scheduler]: stopping subpipeline %s',
          self._pipeline_uid,
      )
      pipeline_ops.stop_pipeline(self.mlmd_handle, self._pipeline_uid)
      logging.info(
          '[Subpipeline Task Scheduler]: subpipeline stopped %s',
          self._pipeline_uid,
      )
    return task_scheduler.TaskSchedulerResult(
        status=status_lib.Status(code=status_lib.Code.CANCELLED)
    )

  def cancel(self, cancel_task: task_lib.CancelTask) -> None:
    self._cancel.set()


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
