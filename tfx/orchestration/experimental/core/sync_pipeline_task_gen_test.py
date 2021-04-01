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
from tfx.orchestration.experimental.core import status as status_lib
from tfx.orchestration.experimental.core import sync_pipeline_task_gen as sptg
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import test_utils as otu
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import test_case_utils as tu

from ml_metadata.proto import metadata_store_pb2


def _get_node(pipeline, node_id):
  for node in pipeline.nodes:
    if node.pipeline_node.node_info.id == node_id:
      return node.pipeline_node
  raise ValueError(f'could not find {node_id}')


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
    pipeline = pipeline_pb2.Pipeline()
    self.load_proto_from_text(
        os.path.join(
            os.path.dirname(__file__), 'testdata', 'sync_pipeline.pbtxt'),
        pipeline)
    self._pipeline_run_id = str(uuid.uuid4())
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline, {
            'pipeline_root': pipeline_root,
            'pipeline_run_id': self._pipeline_run_id
        })
    self._pipeline = pipeline

    # Extracts components.
    self._example_gen = _get_node(pipeline, 'my_example_gen')
    self._stats_gen = _get_node(pipeline, 'my_statistics_gen')
    self._schema_gen = _get_node(pipeline, 'my_schema_gen')
    self._transform = _get_node(pipeline, 'my_transform')
    self._example_validator = _get_node(pipeline, 'my_example_validator')
    self._trainer = _get_node(pipeline, 'my_trainer')

    self._task_queue = tq.TaskQueue()

    self._mock_service_job_manager = mock.create_autospec(
        service_jobs.ServiceJobManager, instance=True)

    self._mock_service_job_manager.is_pure_service_node.side_effect = (
        lambda _, node_id: node_id == self._example_gen.node_info.id)
    self._mock_service_job_manager.is_mixed_service_node.side_effect = (
        lambda _, node_id: node_id == self._transform.node_info.id)

  def _verify_exec_node_task(self, node, execution_id, task):
    otu.verify_exec_node_task(self, self._pipeline, node, execution_id, task)

  def _dequeue_and_test(self, use_task_queue, node, execution_id):
    if use_task_queue:
      task = self._task_queue.dequeue()
      self._task_queue.task_done(task)
      self._verify_exec_node_task(node, execution_id, task)

  def _finish_node_execution(self, use_task_queue, node, execution):
    """Simulates successful execution of a node."""
    otu.fake_component_output(self._mlmd_connection, node, execution)
    self._dequeue_and_test(use_task_queue, node, execution.id)

  def _generate_and_test(self, use_task_queue, num_initial_executions,
                         num_tasks_generated, num_new_executions,
                         num_active_executions):
    """Generates tasks and tests the effects."""
    return otu.run_generator_and_test(
        self,
        self._mlmd_connection,
        sptg.SyncPipelineTaskGenerator,
        self._pipeline,
        self._task_queue,
        use_task_queue,
        self._mock_service_job_manager,
        num_initial_executions=num_initial_executions,
        num_tasks_generated=num_tasks_generated,
        num_new_executions=num_new_executions,
        num_active_executions=num_active_executions)

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

    def _ensure_node_services(unused_pipeline_state, node_id):
      self.assertIn(
          node_id,
          (self._example_gen.node_info.id, self._transform.node_info.id))
      return service_jobs.ServiceStatus.SUCCESS

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _ensure_node_services)

    # Generate once. Stats-gen task should be generated.
    tasks, active_executions = self._generate_and_test(
        use_task_queue,
        num_initial_executions=1,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1)
    execution_id = active_executions[0].id
    self._verify_exec_node_task(self._stats_gen, execution_id, tasks[0])

    self._mock_service_job_manager.ensure_node_services.assert_called_with(
        mock.ANY, self._example_gen.node_info.id)
    self._mock_service_job_manager.reset_mock()

    # Finish stats-gen execution.
    self._finish_node_execution(use_task_queue, self._stats_gen,
                                active_executions[0])

    # Schema-gen should execute next.
    tasks, active_executions = self._generate_and_test(
        use_task_queue,
        num_initial_executions=2,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1)
    execution_id = active_executions[0].id
    self._verify_exec_node_task(self._schema_gen, execution_id, tasks[0])

    # Finish schema-gen execution.
    self._finish_node_execution(use_task_queue, self._schema_gen,
                                active_executions[0])

    # Transform and ExampleValidator should both execute next.
    tasks, active_executions = self._generate_and_test(
        use_task_queue,
        num_initial_executions=3,
        num_tasks_generated=2,
        num_new_executions=2,
        num_active_executions=2)
    self._verify_exec_node_task(self._example_validator,
                                active_executions[0].id, tasks[0])
    transform_exec = active_executions[1]
    self._verify_exec_node_task(self._transform, transform_exec.id, tasks[1])

    # Transform is a "mixed service node".
    self._mock_service_job_manager.ensure_node_services.assert_called_once_with(
        mock.ANY, self._transform.node_info.id)
    self._mock_service_job_manager.reset_mock()

    # Finish example-validator execution.
    self._finish_node_execution(use_task_queue, self._example_validator,
                                active_executions[0])

    # Since transform hasn't finished, trainer will not be triggered yet.
    tasks, active_executions = self._generate_and_test(
        use_task_queue,
        num_initial_executions=5,
        num_tasks_generated=0 if use_task_queue else 1,
        num_new_executions=0,
        num_active_executions=1)
    if not use_task_queue:
      self._verify_exec_node_task(self._transform, active_executions[0].id,
                                  tasks[0])

    # Finish transform execution.
    self._finish_node_execution(use_task_queue, self._transform, transform_exec)

    # Now all trainer upstream nodes are done, so trainer will be triggered.
    tasks, active_executions = self._generate_and_test(
        use_task_queue,
        num_initial_executions=5,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1)
    self._verify_exec_node_task(self._trainer, active_executions[0].id,
                                tasks[0])

    # Finish trainer execution.
    self._finish_node_execution(use_task_queue, self._trainer,
                                active_executions[0])

    # No more components to execute, FinalizePipelineTask should be generated.
    tasks, _ = self._generate_and_test(
        use_task_queue,
        num_initial_executions=6,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0)
    self.assertLen(tasks, 1)
    self.assertTrue(task_lib.is_finalize_pipeline_task(tasks[0]))
    self.assertEqual(status_lib.Code.OK, tasks[0].status.code)
    if use_task_queue:
      self.assertTrue(self._task_queue.is_empty())

  def test_service_job_running(self):
    """Tests task generation when example-gen service job is still running."""

    def _ensure_node_services(unused_pipeline_state, node_id):
      self.assertEqual('my_example_gen', node_id)
      return service_jobs.ServiceStatus.RUNNING

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _ensure_node_services)
    tasks, _ = self._generate_and_test(
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
    tasks, _ = self._generate_and_test(
        True,
        num_initial_executions=0,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0)
    self.assertLen(tasks, 1)
    self.assertTrue(task_lib.is_finalize_pipeline_task(tasks[0]))
    self.assertEqual(status_lib.Code.ABORTED, tasks[0].status.code)
    self.assertRegexMatch(tasks[0].status.message, ['service job failed'])

  @parameterized.parameters(False, True)
  def test_node_failed(self, use_task_queue):
    """Tests task generation when a node registers a failed execution."""
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1, 1)

    def _ensure_node_services(unused_pipeline_state, node_id):
      self.assertEqual(self._example_gen.node_info.id, node_id)
      return service_jobs.ServiceStatus.SUCCESS

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _ensure_node_services)

    tasks, active_executions = self._generate_and_test(
        use_task_queue,
        num_initial_executions=1,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1)
    self.assertEqual(
        task_lib.NodeUid.from_pipeline_node(self._pipeline, self._stats_gen),
        tasks[0].node_uid)
    stats_gen_exec = active_executions[0]

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
    tasks, _ = self._generate_and_test(
        True,
        num_initial_executions=2,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0)
    self.assertLen(tasks, 1)
    self.assertTrue(task_lib.is_finalize_pipeline_task(tasks[0]))
    self.assertEqual(status_lib.Code.ABORTED, tasks[0].status.code)
    self.assertRegexMatch(tasks[0].status.message, ['foobar error'])


if __name__ == '__main__':
  tf.test.main()
