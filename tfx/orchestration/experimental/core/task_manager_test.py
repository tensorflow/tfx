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
"""Tests for tfx.orchestration.experimental.core.task_manager."""

import functools
import threading

from absl import logging
from absl.testing.absltest import mock
import tensorflow as tf
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_manager as tm
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import task_scheduler as ts
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.portable import test_utils as tu
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2


def _test_exec_node_task(node_id, pipeline_id, pipeline_run_id=None):
  node_uid = task_lib.NodeUid(
      pipeline_id=pipeline_id, pipeline_run_id=pipeline_run_id, node_id=node_id)
  return test_utils.create_exec_node_task(node_uid)


def _test_cancel_node_task(node_id, pipeline_id, pipeline_run_id=None):
  node_uid = task_lib.NodeUid(
      pipeline_id=pipeline_id, pipeline_run_id=pipeline_run_id, node_id=node_id)
  return task_lib.CancelNodeTask(node_uid=node_uid)


class _Collector:

  def __init__(self):
    self._lock = threading.Lock()
    self.scheduled_tasks = []
    self.cancelled_tasks = []

  def add_scheduled_task(self, task):
    with self._lock:
      self.scheduled_tasks.append(task)

  def add_cancelled_task(self, task):
    with self._lock:
      self.cancelled_tasks.append(task)


class _FakeTaskScheduler(ts.TaskScheduler):

  def __init__(self, block_nodes, collector, **kwargs):
    super(_FakeTaskScheduler, self).__init__(**kwargs)
    # For these nodes, `schedule` will block until `cancel` is called.
    self._block_nodes = block_nodes
    self._collector = collector
    self._stop_event = threading.Event()

  def schedule(self):
    logging.info('_FakeTaskScheduler: scheduling task: %s', self.task)
    self._collector.add_scheduled_task(self.task)
    if self.task.node_uid.node_id in self._block_nodes:
      self._stop_event.wait()
    return ts.TaskSchedulerResult(
        schedule_status=0,
        executor_output=execution_result_pb2.ExecutorOutput())

  def cancel(self):
    logging.info('_FakeTaskScheduler: cancelling task: %s', self.task)
    self._collector.add_cancelled_task(self.task)
    self._stop_event.set()


class TaskManagerTest(tu.TfxTest):

  @mock.patch.object(tm, '_publish_execution_results')
  def test_queue_multiplexing(self, mock_publish):
    # Create a pipeline IR containing deployment config for testing.
    deployment_config = pipeline_pb2.IntermediateDeploymentConfig()
    executor_spec = pipeline_pb2.ExecutorSpec.PythonClassExecutorSpec(
        class_path='trainer.TrainerExecutor')
    deployment_config.executor_specs['Trainer'].Pack(executor_spec)
    deployment_config.executor_specs['Transform'].Pack(executor_spec)
    deployment_config.executor_specs['Evaluator'].Pack(executor_spec)
    pipeline = pipeline_pb2.Pipeline()
    pipeline.deployment_config.Pack(deployment_config)

    collector = _Collector()

    # Register a bunch of fake task schedulers.
    # Register fake task scheduler.
    ts.TaskSchedulerRegistry.register(
        deployment_config.executor_specs['Trainer'].type_url,
        functools.partial(
            _FakeTaskScheduler,
            block_nodes={'Trainer', 'Transform'},
            collector=collector))

    task_queue = tq.TaskQueue()

    # Enqueue some tasks.
    trainer_exec_task = _test_exec_node_task('Trainer', 'test-pipeline')
    task_queue.enqueue(trainer_exec_task)
    task_queue.enqueue(_test_cancel_node_task('Trainer', 'test-pipeline'))

    with tm.TaskManager(
        mock.Mock(),
        pipeline,
        task_queue,
        max_active_task_schedulers=1000,
        max_dequeue_wait_secs=0.1,
        process_all_queued_tasks_before_exit=True):
      # Enqueue more tasks after task manager starts.
      transform_exec_task = _test_exec_node_task('Transform', 'test-pipeline')
      task_queue.enqueue(transform_exec_task)
      evaluator_exec_task = _test_exec_node_task('Evaluator', 'test-pipeline')
      task_queue.enqueue(evaluator_exec_task)
      task_queue.enqueue(_test_cancel_node_task('Transform', 'test-pipeline'))

    # Ensure that all exec and cancellation tasks were processed correctly.
    self.assertCountEqual(
        [trainer_exec_task, transform_exec_task, evaluator_exec_task],
        collector.scheduled_tasks)
    self.assertCountEqual([trainer_exec_task, transform_exec_task],
                          collector.cancelled_tasks)
    mock_publish.assert_has_calls([
        mock.call(mock.ANY, pipeline, trainer_exec_task, mock.ANY),
        mock.call(mock.ANY, pipeline, transform_exec_task, mock.ANY),
        mock.call(mock.ANY, pipeline, evaluator_exec_task, mock.ANY)
    ],
                                  any_order=True)


if __name__ == '__main__':
  tf.test.main()
