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

import contextlib
import functools
import os
import threading

from absl import logging
from absl.testing.absltest import mock
import tensorflow as tf
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import async_pipeline_task_gen as asptg
from tfx.orchestration.experimental.core import constants
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import status as status_lib
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_manager as tm
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import task_scheduler as ts
from tfx.orchestration.experimental.core import test_utils
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import test_case_utils as tu

from ml_metadata.proto import metadata_store_pb2


def _test_exec_node_task(node_id, pipeline_id, pipeline=None):
  node_uid = task_lib.NodeUid(
      pipeline_uid=task_lib.PipelineUid(pipeline_id=pipeline_id),
      node_id=node_id)
  return test_utils.create_exec_node_task(node_uid, pipeline=pipeline)


def _test_cancel_node_task(node_id, pipeline_id):
  node_uid = task_lib.NodeUid(
      pipeline_uid=task_lib.PipelineUid(pipeline_id=pipeline_id),
      node_id=node_id)
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
        status=status_lib.Status(code=status_lib.Code.OK),
        executor_output=execution_result_pb2.ExecutorOutput())

  def cancel(self):
    logging.info('_FakeTaskScheduler: cancelling task: %s', self.task)
    self._collector.add_cancelled_task(self.task)
    self._stop_event.set()


class TaskManagerTest(tu.TfxTest):

  def setUp(self):
    super(TaskManagerTest, self).setUp()

    # Create a pipeline IR containing deployment config for testing.
    deployment_config = pipeline_pb2.IntermediateDeploymentConfig()
    executor_spec = pipeline_pb2.ExecutorSpec.PythonClassExecutorSpec(
        class_path='trainer.TrainerExecutor')
    deployment_config.executor_specs['Trainer'].Pack(executor_spec)
    deployment_config.executor_specs['Transform'].Pack(executor_spec)
    deployment_config.executor_specs['Evaluator'].Pack(executor_spec)
    pipeline = pipeline_pb2.Pipeline()
    pipeline.pipeline_info.id = 'test-pipeline'
    pipeline.deployment_config.Pack(deployment_config)

    ts.TaskSchedulerRegistry.clear()

    self._deployment_config = deployment_config
    self._pipeline = pipeline
    self._type_url = deployment_config.executor_specs['Trainer'].type_url

  @contextlib.contextmanager
  def _task_manager(self, task_queue):
    with tm.TaskManager(
        mock.Mock(),
        task_queue,
        max_active_task_schedulers=1000,
        max_dequeue_wait_secs=0.1,
        process_all_queued_tasks_before_exit=True) as task_manager:
      yield task_manager

  @mock.patch.object(tm, '_publish_execution_results')
  def test_task_handling(self, mock_publish):
    collector = _Collector()

    # Register a fake task scheduler.
    ts.TaskSchedulerRegistry.register(
        self._type_url,
        functools.partial(
            _FakeTaskScheduler,
            block_nodes={'Trainer', 'Transform'},
            collector=collector))

    task_queue = tq.TaskQueue()

    # Enqueue some tasks.
    trainer_exec_task = _test_exec_node_task(
        'Trainer', 'test-pipeline', pipeline=self._pipeline)
    task_queue.enqueue(trainer_exec_task)
    task_queue.enqueue(_test_cancel_node_task('Trainer', 'test-pipeline'))

    with self._task_manager(task_queue) as task_manager:
      # Enqueue more tasks after task manager starts.
      transform_exec_task = _test_exec_node_task(
          'Transform', 'test-pipeline', pipeline=self._pipeline)
      task_queue.enqueue(transform_exec_task)
      evaluator_exec_task = _test_exec_node_task(
          'Evaluator', 'test-pipeline', pipeline=self._pipeline)
      task_queue.enqueue(evaluator_exec_task)
      task_queue.enqueue(_test_cancel_node_task('Transform', 'test-pipeline'))

    self.assertTrue(task_manager.done())
    self.assertIsNone(task_manager.exception())

    # Ensure that all exec and cancellation tasks were processed correctly.
    self.assertCountEqual(
        [trainer_exec_task, transform_exec_task, evaluator_exec_task],
        collector.scheduled_tasks)
    self.assertCountEqual([trainer_exec_task, transform_exec_task],
                          collector.cancelled_tasks)
    mock_publish.assert_has_calls([
        mock.call(
            mlmd_handle=mock.ANY, task=trainer_exec_task, result=mock.ANY),
        mock.call(
            mlmd_handle=mock.ANY, task=transform_exec_task, result=mock.ANY),
        mock.call(
            mlmd_handle=mock.ANY, task=evaluator_exec_task, result=mock.ANY),
    ],
                                  any_order=True)

  @mock.patch.object(tm, '_publish_execution_results')
  def test_exceptions_are_surfaced(self, mock_publish):

    def _publish(**kwargs):
      task = kwargs['task']
      assert task_lib.is_exec_node_task(task)
      if task.node_uid.node_id == 'Transform':
        raise ValueError('test error')
      return mock.DEFAULT

    mock_publish.side_effect = _publish

    collector = _Collector()

    # Register a fake task scheduler.
    ts.TaskSchedulerRegistry.register(
        self._type_url,
        functools.partial(
            _FakeTaskScheduler, block_nodes={}, collector=collector))

    task_queue = tq.TaskQueue()

    with self._task_manager(task_queue) as task_manager:
      transform_task = _test_exec_node_task(
          'Transform', 'test-pipeline', pipeline=self._pipeline)
      trainer_task = _test_exec_node_task(
          'Trainer', 'test-pipeline', pipeline=self._pipeline)
      task_queue.enqueue(transform_task)
      task_queue.enqueue(trainer_task)

    self.assertTrue(task_manager.done())
    exception = task_manager.exception()
    self.assertIsNotNone(exception)
    self.assertIsInstance(exception, tm.TasksProcessingError)
    self.assertLen(exception.errors, 1)
    self.assertEqual('test error', str(exception.errors[0]))

    self.assertCountEqual([transform_task, trainer_task],
                          collector.scheduled_tasks)
    mock_publish.assert_has_calls([
        mock.call(mlmd_handle=mock.ANY, task=transform_task, result=mock.ANY),
        mock.call(mlmd_handle=mock.ANY, task=trainer_task, result=mock.ANY),
    ],
                                  any_order=True)


class _FakeComponentScheduler(ts.TaskScheduler):

  def __init__(self, return_result, exception, **kwargs):
    super(_FakeComponentScheduler, self).__init__(**kwargs)
    self.exception = exception
    self.return_result = return_result

  def schedule(self):
    if self.exception:
      raise self.exception
    return self.return_result

  def cancel(self):
    pass


def _make_executor_output(task, code=status_lib.Code.OK, msg=''):
  assert task_lib.is_exec_node_task(task)
  executor_output = execution_result_pb2.ExecutorOutput()
  for key, artifacts in task.output_artifacts.items():
    for artifact in artifacts:
      executor_output.output_artifacts[key].artifacts.add().CopyFrom(
          artifact.mlmd_artifact)
  executor_output.execution_result.code = code
  executor_output.execution_result.result_message = msg
  return executor_output


class TaskManagerE2ETest(tu.TfxTest):
  """Test end-to-end from task generation to publication of results to MLMD."""

  def setUp(self):
    super(TaskManagerE2ETest, self).setUp()
    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())

    # Makes sure multiple connections within a test always connect to the same
    # MLMD instance.
    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    self._metadata_path = metadata_path
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self._mlmd_connection = metadata.Metadata(
        connection_config=connection_config)

    # Sets up the pipeline.
    pipeline = pipeline_pb2.Pipeline()
    self.load_proto_from_text(
        os.path.join(
            os.path.dirname(__file__), 'testdata', 'async_pipeline.pbtxt'),
        pipeline)

    # Extracts components.
    self._example_gen = pipeline.nodes[0].pipeline_node
    self._transform = pipeline.nodes[1].pipeline_node
    self._trainer = pipeline.nodes[2].pipeline_node

    # Pack deployment config for testing.
    deployment_config = pipeline_pb2.IntermediateDeploymentConfig()
    executor_spec = pipeline_pb2.ExecutorSpec.PythonClassExecutorSpec(
        class_path='fake.ClassPath')
    deployment_config.executor_specs[self._trainer.node_info.id].Pack(
        executor_spec)
    deployment_config.executor_specs[self._transform.node_info.id].Pack(
        executor_spec)
    self._type_url = deployment_config.executor_specs[
        self._trainer.node_info.id].type_url
    pipeline.deployment_config.Pack(deployment_config)
    self._pipeline = pipeline
    self._pipeline_info = pipeline.pipeline_info
    self._pipeline_runtime_spec = pipeline.runtime_spec
    self._pipeline_runtime_spec.pipeline_root.field_value.string_value = (
        pipeline_root)

    ts.TaskSchedulerRegistry.clear()
    self._task_queue = tq.TaskQueue()

    # Run fake example-gen to prepare downstreams component triggers.
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)

    # Task generator should produce a task to run transform.
    with self._mlmd_connection as m:
      pipeline_state = pstate.PipelineState.new(m, self._pipeline)
      tasks = asptg.AsyncPipelineTaskGenerator(
          m, pipeline_state, self._task_queue.contains_task_id,
          service_jobs.DummyServiceJobManager()).generate()
    self.assertLen(tasks, 1)
    task = tasks[0]
    self.assertEqual('my_transform', task.node_uid.node_id)

    # Task generator should produce a task to run transform.
    with self._mlmd_connection as m:
      pipeline_state = pstate.PipelineState.new(m, self._pipeline)
      tasks = asptg.AsyncPipelineTaskGenerator(
          m, pipeline_state, self._task_queue.contains_task_id,
          service_jobs.DummyServiceJobManager()).generate()
    self.assertLen(tasks, 1)
    self._task = tasks[0]
    self.assertEqual('my_transform', self._task.node_uid.node_id)
    self._task_queue.enqueue(self._task)

    # There should be 1 active execution in MLMD.
    with self._mlmd_connection as m:
      executions = m.store.get_executions()
    active_executions = [
        e for e in executions
        if e.last_known_state == metadata_store_pb2.Execution.RUNNING
    ]
    self.assertLen(active_executions, 1)

    # Active execution id.
    self._execution_id = active_executions[0].id

  def _register_task_scheduler(self, return_result, exception=None):
    ts.TaskSchedulerRegistry.register(
        self._type_url,
        functools.partial(
            _FakeComponentScheduler,
            return_result=return_result,
            exception=exception))

  def _run_task_manager(self):
    with self._mlmd_connection as m:
      with tm.TaskManager(
          m,
          self._task_queue,
          1000,
          max_dequeue_wait_secs=0.1,
          process_all_queued_tasks_before_exit=True) as task_manager:
        pass
    return task_manager

  def _get_execution(self):
    with self._mlmd_connection as m:
      executions = m.store.get_executions_by_id([self._execution_id])
    return executions[0]

  def test_successful_execution(self):
    # Register a fake task scheduler that returns a successful execution result
    # and `OK` task scheduler status.
    self._register_task_scheduler(
        ts.TaskSchedulerResult(
            status=status_lib.Status(code=status_lib.Code.OK),
            executor_output=_make_executor_output(self._task, code=0)))
    task_manager = self._run_task_manager()
    self.assertTrue(task_manager.done())
    self.assertIsNone(task_manager.exception())

    # Check that the task was processed and MLMD execution marked successful.
    self.assertTrue(self._task_queue.is_empty())
    execution = self._get_execution()
    self.assertEqual(metadata_store_pb2.Execution.COMPLETE,
                     execution.last_known_state)

  def test_scheduler_failure(self):
    # Register a fake task scheduler that returns a failure status.
    self._register_task_scheduler(
        ts.TaskSchedulerResult(
            status=status_lib.Status(
                code=status_lib.Code.ABORTED, message='foobar error'),
            executor_output=None))
    task_manager = self._run_task_manager()
    self.assertTrue(task_manager.done())
    self.assertIsNone(task_manager.exception())

    # Check that the task was processed and MLMD execution marked failed.
    self.assertTrue(self._task_queue.is_empty())
    execution = self._get_execution()
    self.assertEqual(metadata_store_pb2.Execution.FAILED,
                     execution.last_known_state)
    self.assertEqual(
        'foobar error',
        data_types_utils.get_metadata_value(
            execution.custom_properties[constants.EXECUTION_ERROR_MSG_KEY]))

  def test_executor_failure(self):
    # Register a fake task scheduler that returns success but the executor
    # was cancelled.
    self._register_task_scheduler(
        ts.TaskSchedulerResult(
            status=status_lib.Status(code=status_lib.Code.OK),
            executor_output=_make_executor_output(
                self._task,
                code=status_lib.Code.FAILED_PRECONDITION,
                msg='foobar error')))
    task_manager = self._run_task_manager()
    self.assertTrue(task_manager.done())
    self.assertIsNone(task_manager.exception())

    # Check that the task was processed and MLMD execution marked failed.
    self.assertTrue(self._task_queue.is_empty())
    execution = self._get_execution()
    self.assertEqual(metadata_store_pb2.Execution.FAILED,
                     execution.last_known_state)
    self.assertEqual(
        'foobar error',
        data_types_utils.get_metadata_value(
            execution.custom_properties[constants.EXECUTION_ERROR_MSG_KEY]))

  def test_scheduler_raises_exception(self):
    # Register a fake task scheduler that raises an exception in `schedule`.
    self._register_task_scheduler(None, exception=ValueError('test exception'))
    task_manager = self._run_task_manager()
    self.assertTrue(task_manager.done())
    self.assertIsNone(task_manager.exception())

    # Check that the task was processed and MLMD execution marked failed.
    self.assertTrue(self._task_queue.is_empty())
    execution = self._get_execution()
    self.assertEqual(metadata_store_pb2.Execution.FAILED,
                     execution.last_known_state)


if __name__ == '__main__':
  tf.test.main()
