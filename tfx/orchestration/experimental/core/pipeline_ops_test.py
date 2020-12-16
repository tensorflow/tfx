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
"""Tests for tfx.orchestration.experimental.core.pipeline_ops."""

import base64
import copy
import os
import threading
import time

from absl.testing.absltest import mock
import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import async_pipeline_task_gen
from tfx.orchestration.experimental.core import pipeline_ops
from tfx.orchestration.experimental.core import status as status_lib
from tfx.orchestration.experimental.core import sync_pipeline_task_gen
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import test_case_utils as tu

from ml_metadata.proto import metadata_store_pb2


def _test_pipeline(pipeline_id,
                   execution_mode: pipeline_pb2.Pipeline.ExecutionMode = (
                       pipeline_pb2.Pipeline.ASYNC)):
  pipeline = pipeline_pb2.Pipeline()
  pipeline.pipeline_info.id = pipeline_id
  pipeline.execution_mode = execution_mode
  return pipeline


class PipelineOpsTest(tu.TfxTest):

  def setUp(self):
    super(PipelineOpsTest, self).setUp()
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

  def test_initiate_pipeline_start(self):
    with self._mlmd_connection as m:
      # Initiate a pipeline start.
      pipeline1 = _test_pipeline('pipeline1')
      pipeline_ops.initiate_pipeline_start(m, pipeline1)

      # Initiate another pipeline start.
      pipeline2 = _test_pipeline('pipeline2')
      pipeline_ops.initiate_pipeline_start(m, pipeline2)

      # No error raised => context/execution types exist.
      m.store.get_context_type(pipeline_ops._ORCHESTRATOR_RESERVED_ID)
      m.store.get_execution_type(pipeline_ops._ORCHESTRATOR_RESERVED_ID)

      # Verify MLMD state.
      contexts = m.store.get_contexts_by_type(
          pipeline_ops._ORCHESTRATOR_RESERVED_ID)
      self.assertLen(contexts, 2)
      self.assertCountEqual([
          pipeline_ops._orchestrator_context_name(
              task_lib.PipelineUid.from_pipeline(pipeline1)),
          pipeline_ops._orchestrator_context_name(
              task_lib.PipelineUid.from_pipeline(pipeline2))
      ], [c.name for c in contexts])

      for context in contexts:
        executions = m.store.get_executions_by_context(context.id)
        self.assertLen(executions, 1)
        self.assertEqual(metadata_store_pb2.Execution.NEW,
                         executions[0].last_known_state)
        retrieved_pipeline = pipeline_pb2.Pipeline()
        retrieved_pipeline.ParseFromString(
            base64.b64decode(executions[0].properties[
                pipeline_ops._PIPELINE_IR].string_value))
        expected_pipeline_id = (
            pipeline_ops._pipeline_uid_from_context(context).pipeline_id)
        self.assertEqual(
            _test_pipeline(expected_pipeline_id), retrieved_pipeline)

  def test_initiate_pipeline_start_new_execution(self):
    with self._mlmd_connection as m:
      pipeline1 = _test_pipeline('pipeline1')
      pipeline_ops.initiate_pipeline_start(m, pipeline1)

      # Error if attempted to initiate when old one is active.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.initiate_pipeline_start(m, pipeline1)
      self.assertEqual(status_lib.Code.ALREADY_EXISTS,
                       exception_context.exception.code)

      # Fine to initiate after the previous one is inactive.
      executions = m.store.get_executions_by_type(
          pipeline_ops._ORCHESTRATOR_RESERVED_ID)
      self.assertLen(executions, 1)
      executions[0].last_known_state = metadata_store_pb2.Execution.COMPLETE
      m.store.put_executions(executions)
      execution = pipeline_ops.initiate_pipeline_start(m, pipeline1)
      self.assertEqual(metadata_store_pb2.Execution.NEW,
                       execution.last_known_state)

      # Verify MLMD state.
      contexts = m.store.get_contexts_by_type(
          pipeline_ops._ORCHESTRATOR_RESERVED_ID)
      self.assertLen(contexts, 1)
      self.assertEqual(
          pipeline_ops._orchestrator_context_name(
              task_lib.PipelineUid.from_pipeline(pipeline1)), contexts[0].name)
      executions = m.store.get_executions_by_context(contexts[0].id)
      self.assertLen(executions, 2)
      self.assertCountEqual([
          metadata_store_pb2.Execution.COMPLETE,
          metadata_store_pb2.Execution.NEW
      ], [e.last_known_state for e in executions])

  def test_initiate_pipeline_stop(self):
    with self._mlmd_connection as m:
      pipeline1 = _test_pipeline('pipeline1')
      pipeline_ops.initiate_pipeline_start(m, pipeline1)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline1)
      pipeline_ops._initiate_pipeline_stop(m, pipeline_uid)

      # Verify MLMD state.
      executions = m.store.get_executions_by_type(
          pipeline_ops._ORCHESTRATOR_RESERVED_ID)
      self.assertLen(executions, 1)
      execution = executions[0]
      self.assertEqual(
          1,
          execution.custom_properties[pipeline_ops._STOP_INITIATED].int_value)

  def test_stop_pipeline_non_existent(self):
    with self._mlmd_connection as m:
      # Stop pipeline without creating one.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.stop_pipeline(
            m, task_lib.PipelineUid(pipeline_id='foo', pipeline_run_id=None))
      self.assertEqual(status_lib.Code.NOT_FOUND,
                       exception_context.exception.code)

      # Initiate pipeline start and mark it completed.
      pipeline1 = _test_pipeline('pipeline1')
      execution = pipeline_ops.initiate_pipeline_start(m, pipeline1)
      pipeline_uid = task_lib.PipelineUid.from_pipeline(pipeline1)
      pipeline_ops._initiate_pipeline_stop(m, pipeline_uid)
      execution.last_known_state = metadata_store_pb2.Execution.COMPLETE
      m.store.put_executions([execution])

      # Try to initiate stop again.
      with self.assertRaises(status_lib.StatusNotOkError) as exception_context:
        pipeline_ops.stop_pipeline(m, pipeline_uid)
      self.assertEqual(status_lib.Code.ALREADY_EXISTS,
                       exception_context.exception.code)

  def test_stop_pipeline_wait_for_inactivation(self):
    with self._mlmd_connection as m:
      pipeline1 = _test_pipeline('pipeline1')
      execution = pipeline_ops.initiate_pipeline_start(m, pipeline1)

      def _inactivate(execution):
        time.sleep(2.0)
        with pipeline_ops._PIPELINE_OPS_LOCK:
          execution.last_known_state = metadata_store_pb2.Execution.COMPLETE
          m.store.put_executions([execution])

      thread = threading.Thread(
          target=_inactivate, args=(copy.deepcopy(execution),))
      thread.start()

      pipeline_ops.stop_pipeline(
          m, task_lib.PipelineUid.from_pipeline(pipeline1), timeout_secs=5.0)

      thread.join()

  def test_stop_pipeline_wait_for_inactivation_timeout(self):
    with self._mlmd_connection as m:
      pipeline1 = _test_pipeline('pipeline1')
      pipeline_ops.initiate_pipeline_start(m, pipeline1)

      with self.assertRaisesRegex(
          status_lib.StatusNotOkError,
          'Timed out.*waiting for execution inactivation.'
      ) as exception_context:
        pipeline_ops.stop_pipeline(
            m, task_lib.PipelineUid.from_pipeline(pipeline1), timeout_secs=1.0)
      self.assertEqual(status_lib.Code.DEADLINE_EXCEEDED,
                       exception_context.exception.code)

  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  def test_generate_tasks_async_active_pipelines(self, mock_async_task_gen,
                                                 mock_sync_task_gen):
    with self._mlmd_connection as m:
      # One active pipeline.
      pipeline1 = _test_pipeline('pipeline1')
      pipeline_ops.initiate_pipeline_start(m, pipeline1)

      # Another active pipeline (with previously completed execution).
      pipeline2 = _test_pipeline('pipeline2')
      execution2 = pipeline_ops.initiate_pipeline_start(m, pipeline2)
      execution2.last_known_state = metadata_store_pb2.Execution.COMPLETE
      m.store.put_executions([execution2])
      execution2 = pipeline_ops.initiate_pipeline_start(m, pipeline2)

      # Inactive pipelines should be ignored.
      pipeline3 = _test_pipeline('pipeline3')
      execution3 = pipeline_ops.initiate_pipeline_start(m, pipeline3)
      execution3.last_known_state = metadata_store_pb2.Execution.COMPLETE
      m.store.put_executions([execution3])

      # For active pipelines pipeline1 and pipeline2, there are a couple of
      # active executions.
      def _exec_node_tasks():
        for pipeline_id in ('pipeline1', 'pipeline2'):
          yield [
              test_utils.create_exec_node_task(
                  node_uid=task_lib.NodeUid(
                      pipeline_uid=task_lib.PipelineUid(
                          pipeline_id=pipeline_id, pipeline_run_id=None),
                      node_id='Transform')),
              test_utils.create_exec_node_task(
                  node_uid=task_lib.NodeUid(
                      pipeline_uid=task_lib.PipelineUid(
                          pipeline_id=pipeline_id, pipeline_run_id=None),
                      node_id='Trainer'))
          ]

      mock_async_task_gen.return_value.generate.side_effect = _exec_node_tasks()

      task_queue = tq.TaskQueue()
      pipeline_ops.generate_tasks(m, task_queue)

      self.assertEqual(2, mock_async_task_gen.return_value.generate.call_count)
      mock_sync_task_gen.assert_not_called()

      # Verify that tasks are enqueued in the expected order.
      for node_id in ('Transform', 'Trainer'):
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertTrue(task_lib.is_exec_node_task(task))
        self.assertEqual(node_id, task.node_uid.node_id)
        self.assertEqual('pipeline1', task.node_uid.pipeline_uid.pipeline_id)
      for node_id in ('Transform', 'Trainer'):
        task = task_queue.dequeue()
        task_queue.task_done(task)
        self.assertTrue(task_lib.is_exec_node_task(task))
        self.assertEqual(node_id, task.node_uid.node_id)
        self.assertEqual('pipeline2', task.node_uid.pipeline_uid.pipeline_id)
      self.assertTrue(task_queue.is_empty())

  @mock.patch.object(sync_pipeline_task_gen, 'SyncPipelineTaskGenerator')
  @mock.patch.object(async_pipeline_task_gen, 'AsyncPipelineTaskGenerator')
  @mock.patch.object(task_gen_utils, 'generate_task_from_active_execution')
  def test_stop_initiated_async_pipelines(self, mock_gen_task_from_active,
                                          mock_async_task_gen,
                                          mock_sync_task_gen):
    with self._mlmd_connection as m:
      pipeline1 = _test_pipeline('pipeline1')
      pipeline1.nodes.add().pipeline_node.node_info.id = 'Transform'
      pipeline1.nodes.add().pipeline_node.node_info.id = 'Trainer'
      pipeline1.nodes.add().pipeline_node.node_info.id = 'Evaluator'
      pipeline_ops.initiate_pipeline_start(m, pipeline1)
      pipeline1_execution = pipeline_ops._initiate_pipeline_stop(
          m, task_lib.PipelineUid.from_pipeline(pipeline1))

      task_queue = tq.TaskQueue()

      # For the stop-initiated pipeline, "Transform" execution task is in queue,
      # "Trainer" has an active execution in MLMD but no task in queue,
      # "Evaluator" has no active execution.
      task_queue.enqueue(
          test_utils.create_exec_node_task(
              node_uid=task_lib.NodeUid(
                  pipeline_uid=task_lib.PipelineUid(
                      pipeline_id='pipeline1', pipeline_run_id=None),
                  node_id='Transform')))
      transform_task = task_queue.dequeue()  # simulates task being processed
      mock_gen_task_from_active.side_effect = [
          test_utils.create_exec_node_task(
              node_uid=task_lib.NodeUid(
                  pipeline_uid=task_lib.PipelineUid(
                      pipeline_id='pipeline1', pipeline_run_id=None),
                  node_id='Trainer'),
              is_cancelled=True), None, None, None, None
      ]

      pipeline_ops.generate_tasks(m, task_queue)

      # There are no active pipelines so these shouldn't be called.
      mock_async_task_gen.assert_not_called()
      mock_sync_task_gen.assert_not_called()

      # Simulate finishing the "Transform" ExecNodeTask.
      task_queue.task_done(transform_task)

      # CancelNodeTask for the "Transform" ExecNodeTask should be next.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(task_lib.is_cancel_node_task(task))
      self.assertEqual('Transform', task.node_uid.node_id)

      # ExecNodeTask for "Trainer" is next.
      task = task_queue.dequeue()
      task_queue.task_done(task)
      self.assertTrue(task_lib.is_exec_node_task(task))
      self.assertEqual('Trainer', task.node_uid.node_id)

      self.assertTrue(task_queue.is_empty())

      mock_gen_task_from_active.assert_has_calls([
          mock.call(
              m,
              pipeline1,
              pipeline1.nodes[1].pipeline_node,
              mock.ANY,
              is_cancelled=True),
          mock.call(
              m,
              pipeline1,
              pipeline1.nodes[2].pipeline_node,
              mock.ANY,
              is_cancelled=True)
      ])
      self.assertEqual(2, mock_gen_task_from_active.call_count)

      # Pipeline execution should continue to be active since active node
      # executions were found in the last call to `generate_tasks`.
      [execution] = m.store.get_executions_by_id([pipeline1_execution.id])
      self.assertTrue(execution_lib.is_execution_active(execution))

      # Call `generate_tasks` again; this time there are no more active node
      # executions so the pipeline should be marked as cancelled.
      pipeline_ops.generate_tasks(m, task_queue)
      self.assertTrue(task_queue.is_empty())
      [execution] = m.store.get_executions_by_id([pipeline1_execution.id])
      self.assertEqual(metadata_store_pb2.Execution.CANCELED,
                       execution.last_known_state)

  def test_to_status_not_ok_error_decorator(self):

    @pipeline_ops._to_status_not_ok_error
    def fn1():
      raise RuntimeError('test error 1')

    @pipeline_ops._to_status_not_ok_error
    def fn2():
      raise status_lib.StatusNotOkError(
          code=status_lib.Code.ALREADY_EXISTS, message='test error 2')

    with self.assertRaisesRegex(status_lib.StatusNotOkError,
                                'test error 1') as ctxt:
      fn1()
    self.assertEqual(status_lib.Code.UNKNOWN, ctxt.exception.code)

    with self.assertRaisesRegex(status_lib.StatusNotOkError,
                                'test error 2') as ctxt:
      fn2()
    self.assertEqual(status_lib.Code.ALREADY_EXISTS, ctxt.exception.code)


if __name__ == '__main__':
  tf.test.main()
