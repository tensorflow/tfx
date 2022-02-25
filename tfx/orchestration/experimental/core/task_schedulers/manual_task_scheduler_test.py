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
"""Tests for tfx.orchestration.experimental.core.task_schedulers.manual_task_scheduler."""

import os
import threading
import time
import typing
import uuid

import tensorflow as tf
from tfx.dsl.compiler import constants
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import sync_pipeline_task_gen as sptg
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import task_scheduler as ts
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.experimental.core.task_schedulers import manual_task_scheduler
from tfx.orchestration.experimental.core.testing import test_manual_node
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.utils import status as status_lib


class ManualTaskSchedulerTest(test_utils.TfxTest):

  def setUp(self):
    super().setUp()

    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())

    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self._mlmd_connection = metadata.Metadata(
        connection_config=connection_config)

    self._pipeline = self._make_pipeline(pipeline_root, str(uuid.uuid4()))
    self._manual_node = self._pipeline.nodes[0].pipeline_node

  def _make_pipeline(self, pipeline_root, pipeline_run_id):
    pipeline = test_manual_node.create_pipeline()
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline, {
            constants.PIPELINE_ROOT_PARAMETER_NAME: pipeline_root,
            constants.PIPELINE_RUN_ID_PARAMETER_NAME: pipeline_run_id,
        })
    return pipeline

  def test_manual_task_scheduler(self):
    task_queue = tq.TaskQueue()

    [manual_task] = test_utils.run_generator_and_test(
        test_case=self,
        mlmd_connection=self._mlmd_connection,
        generator_class=sptg.SyncPipelineTaskGenerator,
        pipeline=self._pipeline,
        task_queue=task_queue,
        use_task_queue=True,
        service_job_manager=None,
        num_initial_executions=0,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._manual_node],
        ignore_update_node_state_tasks=True)

    ts_result = []

    def start_scheduler(ts_result):
      with self._mlmd_connection as m:
        ts_result.append(
            manual_task_scheduler.ManualTaskScheduler(
                mlmd_handle=m, pipeline=self._pipeline,
                task=manual_task).schedule())

    # Marks the execution as COMPLETE.
    def resume_node():
      task = typing.cast(task_lib.ExecNodeTask, manual_task)
      with mlmd_state.mlmd_execution_atomic_op(
          mlmd_handle=self._mlmd_connection,
          execution_id=task.execution_id) as execution:
        completed_state = manual_task_scheduler.ManualNodeState(
            state=manual_task_scheduler.ManualNodeState.COMPLETED)
        completed_state.set_mlmd_value(
            execution.custom_properties.get_or_create(
                manual_task_scheduler.NODE_STATE_PROPERTY_KEY))

    # Shortens the polling interval during test.
    manual_task_scheduler._POLLING_INTERVAL_SECS = 1

    # Starts task scheduler and keeps polling for the node state.
    # The scheduler should be blocked (ts_result has nothing)
    # because the node state stays in WAITING.
    threading.Thread(target=start_scheduler, args=(ts_result,)).start()
    self.assertEqual(len(ts_result), 0)
    time.sleep(manual_task_scheduler._POLLING_INTERVAL_SECS * 10)
    self.assertEqual(len(ts_result), 0)

    # Changes node state to COMPLETED in another thread.
    threading.Thread(target=resume_node).start()
    # Waits for the state change to propagate through.
    time.sleep(manual_task_scheduler._POLLING_INTERVAL_SECS * 10)
    self.assertEqual(len(ts_result), 1)
    self.assertEqual(status_lib.Code.OK, ts_result[0].status.code)
    self.assertIsInstance(ts_result[0].output, ts.ExecutorNodeOutput)


if __name__ == '__main__':
  tf.test.main()
