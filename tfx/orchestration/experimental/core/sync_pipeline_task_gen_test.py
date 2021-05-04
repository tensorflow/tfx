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
"""Tests for tfx.orchestration.experimental.core.sync_pipeline_task_gen."""

import os
import uuid

from absl.testing import parameterized
from absl.testing.absltest import mock
import tensorflow as tf
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import constants
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import sync_pipeline_task_gen as sptg
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import test_utils as otu
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib
from tfx.utils import test_case_utils as tu

from ml_metadata.proto import metadata_store_pb2


class SyncPipelineTaskGeneratorTest(tu.TfxTest, parameterized.TestCase):

  def setUp(self):
    super(SyncPipelineTaskGeneratorTest, self).setUp()
    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())
    self._pipeline_root = pipeline_root

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
    pipeline = self._make_pipeline(self._pipeline_root, str(uuid.uuid4()))
    self._pipeline = pipeline

    # Extracts components.
    self._example_gen = otu.get_node(pipeline, 'my_example_gen')
    self._stats_gen = otu.get_node(pipeline, 'my_statistics_gen')
    self._schema_gen = otu.get_node(pipeline, 'my_schema_gen')
    self._transform = otu.get_node(pipeline, 'my_transform')
    self._example_validator = otu.get_node(pipeline, 'my_example_validator')
    self._trainer = otu.get_node(pipeline, 'my_trainer')

    self._task_queue = tq.TaskQueue()

    self._mock_service_job_manager = mock.create_autospec(
        service_jobs.ServiceJobManager, instance=True)

    self._mock_service_job_manager.is_pure_service_node.side_effect = (
        lambda _, node_id: node_id == self._example_gen.node_info.id)
    self._mock_service_job_manager.is_mixed_service_node.side_effect = (
        lambda _, node_id: node_id == self._transform.node_info.id)

    def _default_ensure_node_services(unused_pipeline_state, node_id):
      self.assertIn(
          node_id,
          (self._example_gen.node_info.id, self._transform.node_info.id))
      return service_jobs.ServiceStatus.SUCCESS

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _default_ensure_node_services)

  def _make_pipeline(self, pipeline_root, pipeline_run_id):
    pipeline = pipeline_pb2.Pipeline()
    self.load_proto_from_text(
        os.path.join(
            os.path.dirname(__file__), 'testdata', 'sync_pipeline.pbtxt'),
        pipeline)
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline, {
            'pipeline_root': pipeline_root,
            'pipeline_run_id': pipeline_run_id,
        })
    return pipeline

  def _finish_node_execution(self, use_task_queue, exec_node_task):
    """Simulates successful execution of a node."""
    otu.fake_execute_node(self._mlmd_connection, exec_node_task)
    if use_task_queue:
      dequeued_task = self._task_queue.dequeue()
      self._task_queue.task_done(dequeued_task)
      self.assertEqual(exec_node_task.task_id, dequeued_task.task_id)

  def _generate_and_test(self,
                         use_task_queue,
                         num_initial_executions,
                         num_tasks_generated,
                         num_new_executions,
                         num_active_executions,
                         pipeline=None,
                         expected_exec_nodes=None):
    """Generates tasks and tests the effects."""
    return otu.run_generator_and_test(
        self,
        self._mlmd_connection,
        sptg.SyncPipelineTaskGenerator,
        pipeline or self._pipeline,
        self._task_queue,
        use_task_queue,
        self._mock_service_job_manager,
        num_initial_executions=num_initial_executions,
        num_tasks_generated=num_tasks_generated,
        num_new_executions=num_new_executions,
        num_active_executions=num_active_executions,
        expected_exec_nodes=expected_exec_nodes)

  @parameterized.parameters(False, True)
  def test_tasks_generated_when_upstream_done(self, use_task_queue):
    """Tests that tasks are generated when upstream is done.

    Args:
      use_task_queue: If task queue is enabled, new tasks are only generated if
        a task with the same task_id does not already exist in the queue.
        `use_task_queue=False` is useful to test the case of task generation
        when task queue is empty (for eg: due to orchestrator restart).
    """
    # Simulate that ExampleGen has already completed successfully.
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1, 1)

    # Generate once. Stats-gen task should be generated.
    [stats_gen_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=1,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._stats_gen])

    self._mock_service_job_manager.ensure_node_services.assert_called_with(
        mock.ANY, self._example_gen.node_info.id)
    self._mock_service_job_manager.reset_mock()

    # Finish stats-gen execution.
    self._finish_node_execution(use_task_queue, stats_gen_task)

    # Schema-gen should execute next.
    [schema_gen_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=2,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._schema_gen])

    # Finish schema-gen execution.
    self._finish_node_execution(use_task_queue, schema_gen_task)

    # Transform and ExampleValidator should both execute next.
    [example_validator_task, transform_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=3,
        num_tasks_generated=2,
        num_new_executions=2,
        num_active_executions=2,
        expected_exec_nodes=[self._example_validator, self._transform])

    # Transform is a "mixed service node".
    self._mock_service_job_manager.ensure_node_services.assert_called_once_with(
        mock.ANY, self._transform.node_info.id)
    self._mock_service_job_manager.reset_mock()

    # Finish example-validator execution.
    self._finish_node_execution(use_task_queue, example_validator_task)

    # Since transform hasn't finished, trainer will not be triggered yet.
    tasks = self._generate_and_test(
        use_task_queue,
        num_initial_executions=5,
        num_tasks_generated=0 if use_task_queue else 1,
        num_new_executions=0,
        num_active_executions=1,
        expected_exec_nodes=[] if use_task_queue else [self._transform])
    if not use_task_queue:
      transform_task = tasks[0]

    # Finish transform execution.
    self._finish_node_execution(use_task_queue, transform_task)

    # Now all trainer upstream nodes are done, so trainer will be triggered.
    [trainer_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=5,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._trainer])

    # Finish trainer execution.
    self._finish_node_execution(use_task_queue, trainer_task)

    # No more components to execute, FinalizePipelineTask should be generated.
    [finalize_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=6,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0)
    self.assertTrue(task_lib.is_finalize_pipeline_task(finalize_task))
    self.assertEqual(status_lib.Code.OK, finalize_task.status.code)
    if use_task_queue:
      self.assertTrue(self._task_queue.is_empty())

  def test_service_job_running(self):
    """Tests task generation when example-gen service job is still running."""

    def _ensure_node_services(unused_pipeline_state, node_id):
      self.assertEqual('my_example_gen', node_id)
      return service_jobs.ServiceStatus.RUNNING

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _ensure_node_services)
    tasks = self._generate_and_test(
        True,
        num_initial_executions=0,
        num_tasks_generated=0,
        num_new_executions=0,
        num_active_executions=0)
    self.assertEmpty(tasks)

  def test_service_job_failed(self):
    """Tests task generation when example-gen service job fails."""

    def _ensure_node_services(unused_pipeline_state, node_id):
      self.assertEqual('my_example_gen', node_id)
      return service_jobs.ServiceStatus.FAILED

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _ensure_node_services)
    [finalize_task] = self._generate_and_test(
        True,
        num_initial_executions=0,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0)
    self.assertTrue(task_lib.is_finalize_pipeline_task(finalize_task))
    self.assertEqual(status_lib.Code.ABORTED, finalize_task.status.code)
    self.assertRegexMatch(finalize_task.status.message, ['service job failed'])

  @parameterized.parameters(False, True)
  def test_node_failed(self, use_task_queue):
    """Tests task generation when a node registers a failed execution."""
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1, 1)

    def _ensure_node_services(unused_pipeline_state, node_id):
      self.assertEqual(self._example_gen.node_info.id, node_id)
      return service_jobs.ServiceStatus.SUCCESS

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _ensure_node_services)

    [stats_gen_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=1,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1)
    self.assertEqual(
        task_lib.NodeUid.from_pipeline_node(self._pipeline, self._stats_gen),
        stats_gen_task.node_uid)
    stats_gen_exec = stats_gen_task.execution

    # Fail stats-gen execution.
    stats_gen_exec.last_known_state = metadata_store_pb2.Execution.FAILED
    data_types_utils.set_metadata_value(
        stats_gen_exec.custom_properties[constants.EXECUTION_ERROR_MSG_KEY],
        'foobar error')
    with self._mlmd_connection as m:
      m.store.put_executions([stats_gen_exec])
    if use_task_queue:
      task = self._task_queue.dequeue()
      self._task_queue.task_done(task)

    # Test generation of FinalizePipelineTask.
    [finalize_task] = self._generate_and_test(
        True,
        num_initial_executions=2,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0)
    self.assertTrue(task_lib.is_finalize_pipeline_task(finalize_task))
    self.assertEqual(status_lib.Code.ABORTED, finalize_task.status.code)
    self.assertRegexMatch(finalize_task.status.message, ['foobar error'])

  def test_cached_execution(self):
    """Tests that cached execution is used if one is available."""

    # Fake ExampleGen run.
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1, 1)

    # Invoking generator should produce an ExecNodeTask for StatsGen.
    [stats_gen_task] = self._generate_and_test(
        False,
        num_initial_executions=1,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1)
    self.assertEqual('my_statistics_gen', stats_gen_task.node_uid.node_id)

    # Finish StatsGen execution.
    otu.fake_execute_node(self._mlmd_connection, stats_gen_task)

    # Prepare another pipeline with a new pipeline_run_id.
    pipeline_run_id = str(uuid.uuid4())
    new_pipeline = self._make_pipeline(self._pipeline_root, pipeline_run_id)
    stats_gen = otu.get_node(new_pipeline, 'my_statistics_gen')

    # Invoking generator for the new pipeline should result in:
    # 1. StatsGen execution succeeds with state "CACHED" but no ExecNodeTask
    #    generated.
    # 2. An ExecNodeTask is generated for SchemaGen (component downstream of
    #    StatsGen) with an active execution in MLMD.
    [schema_gen_task] = self._generate_and_test(
        False,
        pipeline=new_pipeline,
        num_initial_executions=2,
        num_tasks_generated=1,
        num_new_executions=2,
        num_active_executions=1)
    self.assertEqual('my_schema_gen', schema_gen_task.node_uid.node_id)

    # Check that StatsGen execution is successful in state "CACHED".
    with self._mlmd_connection as m:
      executions = task_gen_utils.get_executions(m, stats_gen)
      self.assertLen(executions, 1)
      execution = executions[0]
      self.assertTrue(execution_lib.is_execution_successful(execution))
      self.assertEqual(metadata_store_pb2.Execution.CACHED,
                       execution.last_known_state)


if __name__ == '__main__':
  tf.test.main()
