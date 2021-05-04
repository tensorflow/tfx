# Lint as: python2, python3
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
"""Tests for tfx.orchestration.experimental.core.async_pipeline_task_gen."""

import os

from absl.testing import parameterized
from absl.testing.absltest import mock
import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import async_pipeline_task_gen as asptg
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import test_utils as otu
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib
from tfx.utils import test_case_utils as tu


class AsyncPipelineTaskGeneratorTest(tu.TfxTest, parameterized.TestCase):

  def setUp(self):
    super(AsyncPipelineTaskGeneratorTest, self).setUp()
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
            os.path.dirname(__file__), 'testdata', 'async_pipeline.pbtxt'),
        pipeline)
    self._pipeline = pipeline
    self._pipeline_info = pipeline.pipeline_info
    self._pipeline_runtime_spec = pipeline.runtime_spec
    self._pipeline_runtime_spec.pipeline_root.field_value.string_value = (
        pipeline_root)

    # Extracts components.
    self._example_gen = pipeline.nodes[0].pipeline_node
    self._transform = pipeline.nodes[1].pipeline_node
    self._trainer = pipeline.nodes[2].pipeline_node

    self._task_queue = tq.TaskQueue()

    self._mock_service_job_manager = mock.create_autospec(
        service_jobs.ServiceJobManager, instance=True)

    def _is_pure_service_node(unused_pipeline_state, node_id):
      return node_id == self._example_gen.node_info.id

    def _is_mixed_service_node(unused_pipeline_state, node_id):
      return node_id == self._transform.node_info.id

    self._mock_service_job_manager.is_pure_service_node.side_effect = (
        _is_pure_service_node)
    self._mock_service_job_manager.is_mixed_service_node.side_effect = (
        _is_mixed_service_node)

    def _default_ensure_node_services(unused_pipeline_state, node_id):
      self.assertIn(
          node_id,
          (self._example_gen.node_info.id, self._transform.node_info.id))
      return service_jobs.ServiceStatus.RUNNING

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _default_ensure_node_services)

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
                         expected_exec_nodes=None,
                         ignore_node_ids=None):
    """Generates tasks and tests the effects."""
    return otu.run_generator_and_test(
        self,
        self._mlmd_connection,
        asptg.AsyncPipelineTaskGenerator,
        self._pipeline,
        self._task_queue,
        use_task_queue,
        self._mock_service_job_manager,
        num_initial_executions=num_initial_executions,
        num_tasks_generated=num_tasks_generated,
        num_new_executions=num_new_executions,
        num_active_executions=num_active_executions,
        expected_exec_nodes=expected_exec_nodes,
        ignore_node_ids=ignore_node_ids)

  @parameterized.parameters(0, 1)
  def test_no_tasks_generated_when_no_inputs(self, min_count):
    """Tests no tasks are generated when there are no inputs, regardless of min_count."""
    for node in self._pipeline.nodes:
      for v in node.pipeline_node.inputs.inputs.values():
        v.min_count = min_count

    with self._mlmd_connection as m:
      pipeline_state = pstate.PipelineState.new(m, self._pipeline)
      task_gen = asptg.AsyncPipelineTaskGenerator(
          m,
          pipeline_state,
          lambda _: False,
          service_jobs.DummyServiceJobManager(),
          ignore_node_ids=set([self._example_gen.node_info.id]))
      tasks = task_gen.generate()
      self.assertEmpty(tasks, 'Expected no task generation when no inputs.')
      self.assertEmpty(
          m.store.get_executions(),
          'There must not be any registered executions since no tasks were '
          'generated.')

  @parameterized.parameters(False, True)
  def test_task_generation(self, use_task_queue):
    """Tests async pipeline task generation.

    Args:
      use_task_queue: If task queue is enabled, new tasks are only generated if
        a task with the same task_id does not already exist in the queue.
        `use_task_queue=False` is useful to test the case of task generation
        when task queue is empty (for eg: due to orchestrator restart).
    """
    # Simulate that ExampleGen has already completed successfully.
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1, 1)

    # Generate once.
    [transform_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=1,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._transform])

    self._mock_service_job_manager.ensure_node_services.assert_has_calls([
        mock.call(mock.ANY, self._example_gen.node_info.id),
        mock.call(mock.ANY, self._transform.node_info.id)
    ])

    # No new effects if generate called again.
    tasks = self._generate_and_test(
        use_task_queue,
        num_initial_executions=2,
        num_tasks_generated=0 if use_task_queue else 1,
        num_new_executions=0,
        num_active_executions=1,
        expected_exec_nodes=[] if use_task_queue else [self._transform])
    if not use_task_queue:
      transform_task = tasks[0]

    # Mark transform execution complete.
    self._finish_node_execution(use_task_queue, transform_task)

    # Trainer execution task should be generated next.
    [trainer_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=2,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._trainer])

    # Mark the trainer execution complete.
    self._finish_node_execution(use_task_queue, trainer_task)

    # No more tasks should be generated as there are no new inputs.
    self._generate_and_test(
        use_task_queue,
        num_initial_executions=3,
        num_tasks_generated=0,
        num_new_executions=0,
        num_active_executions=0)

    # Fake another ExampleGen run.
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1, 1)

    # Both transform and trainer tasks should be generated as they both find
    # new inputs.
    [transform_task, trainer_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=4,
        num_tasks_generated=2,
        num_new_executions=2,
        num_active_executions=2,
        expected_exec_nodes=[self._transform, self._trainer])

    # Re-generation will produce the same tasks when task queue disabled.
    tasks = self._generate_and_test(
        use_task_queue,
        num_initial_executions=6,
        num_tasks_generated=0 if use_task_queue else 2,
        num_new_executions=0,
        num_active_executions=2,
        expected_exec_nodes=[]
        if use_task_queue else [self._transform, self._trainer])
    if not use_task_queue:
      transform_task = tasks[0]
      trainer_task = tasks[1]

    # Mark transform execution complete.
    self._finish_node_execution(use_task_queue, transform_task)

    # Mark the trainer execution complete.
    self._finish_node_execution(use_task_queue, trainer_task)

    # Trainer should be triggered again due to transform producing new output.
    [trainer_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=6,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._trainer])

    # Finally, no new tasks once trainer completes.
    self._finish_node_execution(use_task_queue, trainer_task)
    self._generate_and_test(
        use_task_queue,
        num_initial_executions=7,
        num_tasks_generated=0,
        num_new_executions=0,
        num_active_executions=0)
    if use_task_queue:
      self.assertTrue(self._task_queue.is_empty())

  @parameterized.parameters(False, True)
  def test_task_generation_ignore_nodes(self, ignore_transform):
    """Tests nodes can be ignored while generating tasks."""
    # Simulate that ExampleGen has already completed successfully.
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1, 1)

    # Generate once.
    num_initial_executions = 1
    if ignore_transform:
      num_tasks_generated = 0
      num_new_executions = 0
      num_active_executions = 0
      ignore_node_ids = set([self._transform.node_info.id])
    else:
      num_tasks_generated = 1
      num_new_executions = 1
      num_active_executions = 1
      ignore_node_ids = None
    tasks = self._generate_and_test(
        True,
        num_initial_executions=num_initial_executions,
        num_tasks_generated=num_tasks_generated,
        num_new_executions=num_new_executions,
        num_active_executions=num_active_executions,
        ignore_node_ids=ignore_node_ids)
    if ignore_transform:
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
    self.assertTrue(task_lib.is_finalize_node_task(finalize_task))
    self.assertEqual(status_lib.Code.ABORTED, finalize_task.status.code)


if __name__ == '__main__':
  tf.test.main()
