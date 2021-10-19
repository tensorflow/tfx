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
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.experimental.core.testing import test_async_pipeline
from tfx.utils import status as status_lib


class AsyncPipelineTaskGeneratorTest(test_utils.TfxTest,
                                     parameterized.TestCase):

  def setUp(self):
    super().setUp()
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
    pipeline = test_async_pipeline.create_pipeline()
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
    test_utils.fake_execute_node(self._mlmd_connection, exec_node_task)
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
                         expected_exec_nodes=None):
    """Generates tasks and tests the effects."""
    return test_utils.run_generator_and_test(
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
        expected_exec_nodes=expected_exec_nodes)

  @parameterized.parameters(0, 1)
  def test_no_tasks_generated_when_no_inputs(self, min_count):
    """Tests no tasks are generated when there are no inputs, regardless of min_count."""
    for node in self._pipeline.nodes:
      for v in node.pipeline_node.inputs.inputs.values():
        v.min_count = min_count

    with self._mlmd_connection as m:
      pipeline_state = test_utils.get_or_create_pipeline_state(
          m, self._pipeline)
      task_gen = asptg.AsyncPipelineTaskGenerator(
          m, lambda _: False, service_jobs.DummyServiceJobManager())
      tasks = task_gen.generate(pipeline_state)
      self.assertEmpty(tasks, 'Expected no task generation when no inputs.')
      self.assertEmpty(
          test_utils.get_non_orchestrator_executions(m),
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
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)

    # Generate once.
    [update_example_gen_task, update_transform_task,
     exec_transform_task] = self._generate_and_test(
         use_task_queue,
         num_initial_executions=1,
         num_tasks_generated=3,
         num_new_executions=1,
         num_active_executions=1,
         expected_exec_nodes=[self._transform])
    self.assertTrue(task_lib.is_update_node_state_task(update_example_gen_task))
    self.assertEqual(pstate.NodeState.RUNNING, update_example_gen_task.state)
    self.assertTrue(task_lib.is_update_node_state_task(update_transform_task))
    self.assertEqual(pstate.NodeState.RUNNING, update_transform_task.state)
    self.assertTrue(task_lib.is_exec_node_task(exec_transform_task))

    self._mock_service_job_manager.ensure_node_services.assert_has_calls([
        mock.call(mock.ANY, self._example_gen.node_info.id),
        mock.call(mock.ANY, self._transform.node_info.id)
    ])

    # No new effects if generate called again.
    tasks = self._generate_and_test(
        use_task_queue,
        num_initial_executions=2,
        num_tasks_generated=1 if use_task_queue else 3,
        num_new_executions=0,
        num_active_executions=1,
        expected_exec_nodes=[] if use_task_queue else [self._transform])
    if not use_task_queue:
      exec_transform_task = tasks[2]

    # Mark transform execution complete.
    self._finish_node_execution(use_task_queue, exec_transform_task)

    # Trainer execution task should be generated next.
    [
        update_example_gen_task, update_transform_task, update_trainer_task,
        exec_trainer_task
    ] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=2,
        num_tasks_generated=4,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._trainer])
    self.assertTrue(task_lib.is_update_node_state_task(update_example_gen_task))
    self.assertEqual(pstate.NodeState.RUNNING, update_example_gen_task.state)
    self.assertTrue(task_lib.is_update_node_state_task(update_transform_task))
    self.assertEqual(pstate.NodeState.STARTED, update_transform_task.state)
    self.assertTrue(task_lib.is_update_node_state_task(update_trainer_task))
    self.assertEqual(pstate.NodeState.RUNNING, update_trainer_task.state)
    self.assertTrue(task_lib.is_exec_node_task(exec_trainer_task))

    # Mark the trainer execution complete.
    self._finish_node_execution(use_task_queue, exec_trainer_task)

    # Only UpdateNodeStateTask are generated as there are no new inputs.
    tasks = self._generate_and_test(
        use_task_queue,
        num_initial_executions=3,
        num_tasks_generated=3,
        num_new_executions=0,
        num_active_executions=0)
    for task in tasks:
      self.assertTrue(task_lib.is_update_node_state_task(task))
      self.assertEqual(pstate.NodeState.RUNNING, update_example_gen_task.state)

    # Fake another ExampleGen run.
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)

    # Both transform and trainer tasks should be generated as they both find
    # new inputs.
    [
        update_example_gen_task, update_transform_task, exec_transform_task,
        update_trainer_task, exec_trainer_task
    ] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=4,
        num_tasks_generated=5,
        num_new_executions=2,
        num_active_executions=2,
        expected_exec_nodes=[self._transform, self._trainer])
    self.assertTrue(task_lib.is_update_node_state_task(update_example_gen_task))
    self.assertEqual(pstate.NodeState.RUNNING, update_example_gen_task.state)
    self.assertTrue(task_lib.is_update_node_state_task(update_transform_task))
    self.assertEqual(pstate.NodeState.RUNNING, update_transform_task.state)
    self.assertTrue(task_lib.is_exec_node_task(exec_transform_task))
    self.assertTrue(task_lib.is_update_node_state_task(update_trainer_task))
    self.assertEqual(pstate.NodeState.RUNNING, update_trainer_task.state)
    self.assertTrue(task_lib.is_exec_node_task(exec_trainer_task))

    # Re-generation will produce the same tasks when task queue disabled.
    tasks = self._generate_and_test(
        use_task_queue,
        num_initial_executions=6,
        num_tasks_generated=1 if use_task_queue else 5,
        num_new_executions=0,
        num_active_executions=2,
        expected_exec_nodes=[]
        if use_task_queue else [self._transform, self._trainer])
    if not use_task_queue:
      self.assertTrue(task_lib.is_update_node_state_task(tasks[0]))
      self.assertEqual(pstate.NodeState.RUNNING, update_example_gen_task.state)
      self.assertTrue(task_lib.is_update_node_state_task(tasks[1]))
      self.assertEqual(pstate.NodeState.RUNNING, update_example_gen_task.state)
      self.assertTrue(task_lib.is_exec_node_task(tasks[2]))
      self.assertTrue(task_lib.is_update_node_state_task(tasks[3]))
      self.assertEqual(pstate.NodeState.RUNNING, update_example_gen_task.state)
      self.assertTrue(task_lib.is_exec_node_task(tasks[4]))
      exec_transform_task = tasks[2]
      exec_trainer_task = tasks[4]
    else:
      self.assertTrue(task_lib.is_update_node_state_task(tasks[0]))
      self.assertEqual(pstate.NodeState.RUNNING, update_example_gen_task.state)

    # Mark transform execution complete.
    self._finish_node_execution(use_task_queue, exec_transform_task)

    # Mark the trainer execution complete.
    self._finish_node_execution(use_task_queue, exec_trainer_task)

    # Trainer should be triggered again due to transform producing new output.
    [
        update_example_gen_task, update_transform_task, update_trainer_task,
        exec_trainer_task
    ] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=6,
        num_tasks_generated=4,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._trainer])
    self.assertTrue(task_lib.is_update_node_state_task(update_example_gen_task))
    self.assertEqual(pstate.NodeState.RUNNING, update_example_gen_task.state)
    self.assertTrue(task_lib.is_update_node_state_task(update_transform_task))
    self.assertEqual(pstate.NodeState.STARTED, update_transform_task.state)
    self.assertTrue(task_lib.is_update_node_state_task(update_trainer_task))
    self.assertEqual(pstate.NodeState.RUNNING, update_trainer_task.state)
    self.assertTrue(task_lib.is_exec_node_task(exec_trainer_task))

    # Finally, no new tasks once trainer completes.
    self._finish_node_execution(use_task_queue, exec_trainer_task)
    [update_example_gen_task, update_transform_task,
     update_trainer_task] = self._generate_and_test(
         use_task_queue,
         num_initial_executions=7,
         num_tasks_generated=3,
         num_new_executions=0,
         num_active_executions=0)
    self.assertTrue(task_lib.is_update_node_state_task(update_example_gen_task))
    self.assertEqual(pstate.NodeState.RUNNING, update_example_gen_task.state)
    self.assertTrue(task_lib.is_update_node_state_task(update_transform_task))
    self.assertEqual(pstate.NodeState.STARTED, update_transform_task.state)
    self.assertTrue(task_lib.is_update_node_state_task(update_trainer_task))
    self.assertEqual(pstate.NodeState.STARTED, update_trainer_task.state)

    if use_task_queue:
      self.assertTrue(self._task_queue.is_empty())

  @parameterized.parameters(False, True)
  def test_task_generation_when_node_stopped(self, stop_transform):
    """Tests stopped nodes are ignored when generating tasks."""
    # Simulate that ExampleGen has already completed successfully.
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)

    # Generate once.
    num_initial_executions = 1
    if stop_transform:
      num_tasks_generated = 1
      num_new_executions = 0
      num_active_executions = 0
      with self._mlmd_connection as m:
        pipeline_state = test_utils.get_or_create_pipeline_state(
            m, self._pipeline)
        with pipeline_state:
          with pipeline_state.node_state_update_context(
              task_lib.NodeUid.from_pipeline_node(
                  self._pipeline, self._transform)) as node_state:
            node_state.update(pstate.NodeState.STOPPING,
                              status_lib.Status(code=status_lib.Code.CANCELLED))
    else:
      num_tasks_generated = 3
      num_new_executions = 1
      num_active_executions = 1
    tasks = self._generate_and_test(
        True,
        num_initial_executions=num_initial_executions,
        num_tasks_generated=num_tasks_generated,
        num_new_executions=num_new_executions,
        num_active_executions=num_active_executions)
    self.assertLen(tasks, num_tasks_generated)

    if stop_transform:
      self.assertTrue(task_lib.is_update_node_state_task(tasks[0]))
      self.assertEqual(pstate.NodeState.RUNNING, tasks[0].state)
    else:
      self.assertTrue(task_lib.is_update_node_state_task(tasks[0]))
      self.assertEqual(pstate.NodeState.RUNNING, tasks[0].state)
      self.assertTrue(task_lib.is_update_node_state_task(tasks[1]))
      self.assertEqual(pstate.NodeState.RUNNING, tasks[1].state)
      self.assertTrue(task_lib.is_exec_node_task(tasks[2]))

  def test_service_job_failed(self):
    """Tests task generation when example-gen service job fails."""

    def _ensure_node_services(unused_pipeline_state, node_id):
      self.assertEqual('my_example_gen', node_id)
      return service_jobs.ServiceStatus.FAILED

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _ensure_node_services)
    [update_task] = self._generate_and_test(
        True,
        num_initial_executions=0,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0)
    self.assertTrue(task_lib.is_update_node_state_task(update_task))
    self.assertEqual(status_lib.Code.ABORTED, update_task.status.code)


if __name__ == '__main__':
  tf.test.main()
