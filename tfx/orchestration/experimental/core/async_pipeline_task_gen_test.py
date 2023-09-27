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
from tfx.orchestration import node_proto_view
from tfx.orchestration.experimental.core import async_pipeline_task_gen as asptg
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.experimental.core.testing import test_async_pipeline
from tfx.orchestration import mlmd_connection_manager as mlmd_cm
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
    self._mlmd_cm = mlmd_cm.MLMDConnectionManager.sqlite(metadata_path)
    self.enter_context(self._mlmd_cm)
    self._mlmd_connection = self._mlmd_cm.primary_mlmd_handle

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
    self._mock_service_job_manager.stop_node_services.return_value = True

    def _default_ensure_node_services(
        unused_pipeline_state, node_id, unused_backfill_token=''
    ):
      self.assertIn(
          node_id,
          (self._example_gen.node_info.id, self._transform.node_info.id))
      return service_jobs.ServiceStatus.RUNNING

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _default_ensure_node_services)

  def _finish_node_execution(
      self, use_task_queue, exec_node_task, success=True
  ):
    """Simulates successful execution of a node."""
    test_utils.fake_execute_node(
        self._mlmd_connection, exec_node_task, None, success
    )
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
                         ignore_update_node_state_tasks=False):
    """Generates tasks and tests the effects."""
    return test_utils.run_generator_and_test(
        self,
        self._mlmd_cm,
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
        ignore_update_node_state_tasks=ignore_update_node_state_tasks)

  @parameterized.parameters(0, 1)
  def test_tasks_generation_when_no_inputs(self, min_count):
    """Tests no tasks generated when no inputs, regardless of min_count."""

    for node in self._pipeline.nodes:
      for v in node.pipeline_node.inputs.inputs.values():
        v.min_count = min_count

    # Note that "example gen" tasks will be generated since it has no declared
    # inputs, so it is okay to execute it even when there are no inputs.
    [update_example_gen_task] = self._generate_and_test(
        use_task_queue=False,
        num_initial_executions=0,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0,
        expected_exec_nodes=[])
    self.assertIsInstance(update_example_gen_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_example_gen_task.state)

  @parameterized.parameters(False, True)
  @mock.patch.object(task_gen_utils, 'update_external_artifact_type')
  def test_task_generation(self, use_task_queue,
                           mock_update_external_artifact_type):
    """Tests async pipeline task generation.

    Args:
      use_task_queue: If task queue is enabled, new tasks are only generated if
        a task with the same task_id does not already exist in the queue.
        `use_task_queue=False` is useful to test the case of task generation
        when task queue is empty (for eg: due to orchestrator restart).
      mock_update_external_artifact_type: mock object to the function
        task_gen_utils.update_external_artifact_type
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
    self.assertIsInstance(update_example_gen_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_example_gen_task.state)
    self.assertIsInstance(update_transform_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_transform_task.state)
    self.assertIsInstance(exec_transform_task, task_lib.ExecNodeTask)

    self._mock_service_job_manager.ensure_node_services.assert_has_calls([
        mock.call(mock.ANY, self._example_gen.node_info.id, ''),
        mock.call(mock.ANY, self._transform.node_info.id),
    ])

    # No new effects if generate called again.
    tasks = self._generate_and_test(
        use_task_queue,
        num_initial_executions=2,
        num_tasks_generated=0 if use_task_queue else 2,
        num_new_executions=0,
        num_active_executions=1,
        expected_exec_nodes=[] if use_task_queue else [self._transform])
    if not use_task_queue:
      exec_transform_task = tasks[1]

    # Mark transform execution complete.
    self._finish_node_execution(use_task_queue, exec_transform_task)

    # Trainer execution task should be generated next.
    [update_transform_task, update_trainer_task,
     exec_trainer_task] = self._generate_and_test(
         use_task_queue,
         num_initial_executions=2,
         num_tasks_generated=3,
         num_new_executions=1,
         num_active_executions=1,
         expected_exec_nodes=[self._trainer])
    self.assertIsInstance(update_transform_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.STARTED, update_transform_task.state)
    self.assertIsInstance(update_trainer_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_trainer_task.state)
    self.assertIsInstance(exec_trainer_task, task_lib.ExecNodeTask)

    # Mark the trainer execution complete.
    self._finish_node_execution(use_task_queue, exec_trainer_task)

    # Trainer is completed, its state should be updated to STARTED.
    [update_trainer_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=3,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0)
    self.assertIsInstance(update_trainer_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.STARTED, update_trainer_task.state)

    # Fake another ExampleGen run.
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)

    # Both transform and trainer tasks should be generated as they both find
    # new inputs.
    [
        update_transform_task, exec_transform_task, update_trainer_task,
        exec_trainer_task
    ] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=4,
        num_tasks_generated=4,
        num_new_executions=2,
        num_active_executions=2,
        expected_exec_nodes=[self._transform, self._trainer])
    self.assertIsInstance(update_transform_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_transform_task.state)
    self.assertIsInstance(exec_transform_task, task_lib.ExecNodeTask)
    self.assertIsInstance(update_trainer_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_trainer_task.state)
    self.assertIsInstance(exec_trainer_task, task_lib.ExecNodeTask)

    # Re-generation will produce the same tasks when task queue disabled.
    tasks = self._generate_and_test(
        use_task_queue,
        num_initial_executions=6,
        num_tasks_generated=0 if use_task_queue else 4,
        num_new_executions=0,
        num_active_executions=2,
        expected_exec_nodes=[]
        if use_task_queue else [self._transform, self._trainer])
    if not use_task_queue:
      self.assertIsInstance(tasks[0], task_lib.UpdateNodeStateTask)
      self.assertIsInstance(tasks[1], task_lib.ExecNodeTask)
      self.assertIsInstance(tasks[2], task_lib.UpdateNodeStateTask)
      self.assertIsInstance(tasks[3], task_lib.ExecNodeTask)
      exec_transform_task = tasks[1]
      exec_trainer_task = tasks[3]

    # Mark transform execution complete.
    self._finish_node_execution(use_task_queue, exec_transform_task)

    # Mark the trainer execution complete.
    self._finish_node_execution(use_task_queue, exec_trainer_task)

    # Trainer should be triggered again due to transform producing new output.
    [
        update_transform_task, update_trainer_task_1, update_trainer_task_2,
        exec_trainer_task
    ] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=6,
        num_tasks_generated=4,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._trainer])
    self.assertIsInstance(update_transform_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.STARTED, update_transform_task.state)
    self.assertIsInstance(update_trainer_task_1, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.STARTED, update_trainer_task_1.state)
    self.assertIsInstance(update_trainer_task_2, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_trainer_task_2.state)
    self.assertIsInstance(exec_trainer_task, task_lib.ExecNodeTask)

    # Finally, update Trainer's state to STARTED.
    self._finish_node_execution(use_task_queue, exec_trainer_task)
    [update_trainer_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=7,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0)
    self.assertIsInstance(update_trainer_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.STARTED, update_trainer_task.state)

    if use_task_queue:
      self.assertTrue(self._task_queue.is_empty())

    mock_update_external_artifact_type.assert_called()

  @parameterized.parameters(False, True)
  def test_task_generation_for_each(self, use_task_queue):
    """Tests async pipeline task generation.

    Args:
      use_task_queue: If task queue is enabled, new tasks are only generated if
        a task with the same task_id does not already exist in the queue.
        `use_task_queue=False` is useful to test the case of task generation
        when task queue is empty (for eg: due to orchestrator restart).
    """
    # Simulate that ExampleGen run twice for 2 spans.
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 2,
                                    1)

    # Generate once, two executions for Transform is generated.
    [update_example_gen_task, update_transform_task,
     exec_transform_task] = self._generate_and_test(
         use_task_queue,
         num_initial_executions=2,
         num_tasks_generated=3,
         num_new_executions=2,
         num_active_executions=2,
         expected_exec_nodes=[self._transform])
    self.assertIsInstance(update_example_gen_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_example_gen_task.state)
    self.assertIsInstance(update_transform_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_transform_task.state)
    self.assertIsInstance(exec_transform_task, task_lib.ExecNodeTask)

    self._mock_service_job_manager.ensure_node_services.assert_has_calls([
        mock.call(mock.ANY, self._example_gen.node_info.id, ''),
        mock.call(mock.ANY, self._transform.node_info.id),
    ])

    # Mark one of the Transform executions complete.
    self._finish_node_execution(use_task_queue, exec_transform_task)

    # Generate again, an execution for Trainer is generated.
    [
        update_transform_task, exec_transform_task, update_trainer_task,
        exec_trainer_task
    ] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=4,
        num_tasks_generated=4,
        num_new_executions=1,
        num_active_executions=2,
        expected_exec_nodes=[self._transform, self._trainer])
    self.assertIsInstance(update_transform_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_transform_task.state)
    self.assertIsInstance(exec_transform_task, task_lib.ExecNodeTask)
    self.assertIsInstance(update_trainer_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_trainer_task.state)
    self.assertIsInstance(exec_trainer_task, task_lib.ExecNodeTask)

    # Mark the Transform execution complete.
    self._finish_node_execution(use_task_queue, exec_transform_task)
    # Mark the Trainer execution complete.
    self._finish_node_execution(use_task_queue, exec_trainer_task)

    # Generate again, another execution for Trainer is generated.
    [
        update_transform_task, update_trainer_task_1, update_trainer_task_2,
        exec_trainer_task
    ] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=5,
        num_tasks_generated=4,
        num_new_executions=1,
        num_active_executions=1)
    self.assertIsInstance(update_transform_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.STARTED, update_transform_task.state)
    self.assertIsInstance(update_trainer_task_1, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.STARTED, update_trainer_task_1.state)
    self.assertIsInstance(update_trainer_task_2, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_trainer_task_2.state)
    self.assertIsInstance(exec_trainer_task, task_lib.ExecNodeTask)

    # Mark the trainer execution complete.
    self._finish_node_execution(use_task_queue, exec_trainer_task)

    # Finally, update Trainer's state to STARTED.
    [update_trainer_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=6,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0)
    self.assertIsInstance(update_trainer_task, task_lib.UpdateNodeStateTask)
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
              task_lib.NodeUid.from_node(self._pipeline,
                                         self._transform)) as node_state:
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
      self.assertIsInstance(tasks[0], task_lib.UpdateNodeStateTask)
      self.assertEqual(pstate.NodeState.RUNNING, tasks[0].state)
    else:
      self.assertIsInstance(tasks[0], task_lib.UpdateNodeStateTask)
      self.assertEqual(pstate.NodeState.RUNNING, tasks[0].state)
      self.assertIsInstance(tasks[1], task_lib.UpdateNodeStateTask)
      self.assertEqual(pstate.NodeState.RUNNING, tasks[1].state)
      self.assertIsInstance(tasks[2], task_lib.ExecNodeTask)

  def test_service_job_failed(self):
    """Tests task generation when example-gen service job fails."""

    def _ensure_node_services(
        unused_pipeline_state, node_id, unused_backfill_token=''
    ):
      if node_id == 'my_example_gen':
        return service_jobs.ServiceStatus.FAILED

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _ensure_node_services)
    [update_task] = self._generate_and_test(
        True,
        num_initial_executions=0,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0)
    self.assertIsInstance(update_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(status_lib.Code.UNKNOWN, update_task.status.code)

  def test_mix_service_job_failed(self):
    """Tests task generation when my_transform mix service job fails."""

    def _ensure_node_services(
        unused_pipeline_state, node_id, unused_backfill_token=''
    ):
      if node_id == 'my_example_gen':
        return service_jobs.ServiceStatus.RUNNING
      if node_id == 'my_transform':
        return service_jobs.ServiceStatus.FAILED

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _ensure_node_services)
    [example_gen_update_task, transform_update_task] = self._generate_and_test(
        True,
        num_initial_executions=0,
        num_tasks_generated=2,
        num_new_executions=0,
        num_active_executions=0)
    self.assertIsInstance(example_gen_update_task, task_lib.UpdateNodeStateTask)
    self.assertIsInstance(transform_update_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(status_lib.Code.UNKNOWN, transform_update_task.status.code)

  def test_backfill(self):
    """Tests async pipeline task generation for backfill."""
    use_task_queue = True
    # Simulate that ExampleGen has already completed successfully.
    test_utils.fake_example_gen_run(
        self._mlmd_connection, self._example_gen, 1, 1
    )

    # Generate once.
    [update_example_gen_task, update_transform_task, exec_transform_task] = (
        self._generate_and_test(
            use_task_queue,
            num_initial_executions=1,
            num_tasks_generated=3,
            num_new_executions=1,
            num_active_executions=1,
            expected_exec_nodes=[self._transform],
        )
    )
    self.assertIsInstance(update_example_gen_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_example_gen_task.state)
    self.assertIsInstance(update_transform_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_transform_task.state)
    self.assertIsInstance(exec_transform_task, task_lib.ExecNodeTask)

    self._mock_service_job_manager.ensure_node_services.assert_has_calls([
        mock.call(mock.ANY, self._example_gen.node_info.id, ''),
        mock.call(mock.ANY, self._transform.node_info.id),
    ])

    # Mark transform execution complete.
    self._finish_node_execution(use_task_queue, exec_transform_task)

    # Trainer execution task should be generated next.
    [update_transform_task, update_trainer_task, exec_trainer_task] = (
        self._generate_and_test(
            use_task_queue,
            num_initial_executions=2,
            num_tasks_generated=3,
            num_new_executions=1,
            num_active_executions=1,
            expected_exec_nodes=[self._trainer],
        )
    )
    self.assertIsInstance(update_transform_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.STARTED, update_transform_task.state)
    self.assertIsInstance(update_trainer_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_trainer_task.state)
    self.assertIsInstance(exec_trainer_task, task_lib.ExecNodeTask)

    # Mark the trainer execution complete.
    self._finish_node_execution(use_task_queue, exec_trainer_task)

    # Only UpdateNodeStateTask are generated as there are no new inputs.
    [update_trainer_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=3,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0,
    )
    self.assertIsInstance(update_trainer_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.STARTED, update_trainer_task.state)

    # Put Transform in backfill mode.
    with pstate.PipelineState.load(
        self._mlmd_connection,
        task_lib.PipelineUid.from_pipeline(self._pipeline),
    ) as pipeline_state:
      transform_node = task_lib.NodeUid.from_node(
          self._pipeline, node_proto_view.get_view(self._transform)
      )
      with pipeline_state.node_state_update_context(
          transform_node
      ) as node_state:
        node_state.update(
            pstate.NodeState.STARTING,
            backfill_token='backfill-20221215-180505-123456',
        )

    # Transform tasks should be generated as it will start a backfill.
    # Trainer will just be updated to STARTED state, since there are no new
    # inputs.
    [
        update_transform_to_started_task,
        update_transform_to_running_task,
        exec_transform_task,
    ] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=3,
        num_tasks_generated=3,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._transform],
    )
    self.assertIsInstance(
        update_transform_to_started_task, task_lib.UpdateNodeStateTask
    )
    self.assertEqual(
        pstate.NodeState.STARTED, update_transform_to_started_task.state
    )
    self.assertEqual(
        'backfill-20221215-180505-123456',
        update_transform_to_started_task.backfill_token,
    )
    self.assertIsInstance(
        update_transform_to_running_task, task_lib.UpdateNodeStateTask
    )
    self.assertEqual(
        pstate.NodeState.RUNNING, update_transform_to_running_task.state
    )
    self.assertEqual(
        'backfill-20221215-180505-123456',
        update_transform_to_running_task.backfill_token,
    )
    self.assertIsInstance(exec_transform_task, task_lib.ExecNodeTask)

    # Mark transform execution complete.
    self._finish_node_execution(use_task_queue, exec_transform_task)

    # Transform should be stopped, since the backfill is complete.
    # Trainer should be triggered again due to transform producing new output.
    [update_transform_task, update_trainer_task, exec_trainer_task] = (
        self._generate_and_test(
            use_task_queue,
            num_initial_executions=4,
            num_tasks_generated=3,
            num_new_executions=1,
            num_active_executions=1,
            expected_exec_nodes=[self._trainer],
        )
    )
    self.assertIsInstance(update_transform_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.STOPPED, update_transform_task.state)
    self.assertIsInstance(update_trainer_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.RUNNING, update_trainer_task.state)
    self.assertIsInstance(exec_trainer_task, task_lib.ExecNodeTask)

    # Trainer completes, goes back into STARTED state.
    self._finish_node_execution(use_task_queue, exec_trainer_task)
    [update_trainer_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=5,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0,
    )
    self.assertIsInstance(update_trainer_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.STARTED, update_trainer_task.state)

    # Put Transform in backfill mode with the same token as before.
    with pstate.PipelineState.load(
        self._mlmd_connection,
        task_lib.PipelineUid.from_pipeline(self._pipeline),
    ) as pipeline_state:
      transform_node = task_lib.NodeUid.from_node(
          self._pipeline, node_proto_view.get_view(self._transform)
      )
      with pipeline_state.node_state_update_context(
          transform_node
      ) as node_state:
        node_state.update(
            pstate.NodeState.STARTING,
            backfill_token='backfill-20221215-180505-123456',
        )

    # Transform should stop immediately, since it sees the previous backfill
    # execution.
    [update_transform_to_started_task, update_transform_to_stopped_task] = (
        self._generate_and_test(
            use_task_queue,
            num_initial_executions=5,
            num_tasks_generated=2,
            num_new_executions=0,
            num_active_executions=0,
        )
    )
    self.assertIsInstance(
        update_transform_to_started_task, task_lib.UpdateNodeStateTask
    )
    self.assertEqual(
        pstate.NodeState.STARTED, update_transform_to_started_task.state
    )
    self.assertIsInstance(
        update_transform_to_stopped_task, task_lib.UpdateNodeStateTask
    )
    self.assertEqual(
        pstate.NodeState.STOPPED, update_transform_to_stopped_task.state
    )

    # Put Transform in backfill mode with a new token.
    with pstate.PipelineState.load(
        self._mlmd_connection,
        task_lib.PipelineUid.from_pipeline(self._pipeline),
    ) as pipeline_state:
      transform_node = task_lib.NodeUid.from_node(
          self._pipeline, node_proto_view.get_view(self._transform)
      )
      with pipeline_state.node_state_update_context(
          transform_node
      ) as node_state:
        node_state.update(
            pstate.NodeState.STARTING,
            backfill_token='backfill-20221215-192233-234567',
        )

    # Transform tasks should be generated as it will start a new backfill.
    [
        update_transform_to_started_task,
        update_transform_to_running_task,
        exec_transform_task,
    ] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=5,
        num_tasks_generated=3,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._transform],
    )
    self.assertIsInstance(
        update_transform_to_started_task, task_lib.UpdateNodeStateTask
    )
    self.assertEqual(
        pstate.NodeState.STARTED, update_transform_to_started_task.state
    )
    self.assertEqual(
        'backfill-20221215-192233-234567',
        update_transform_to_started_task.backfill_token,
    )
    self.assertIsInstance(
        update_transform_to_running_task, task_lib.UpdateNodeStateTask
    )
    self.assertEqual(
        pstate.NodeState.RUNNING, update_transform_to_running_task.state
    )
    self.assertEqual(
        'backfill-20221215-192233-234567',
        update_transform_to_running_task.backfill_token,
    )
    self.assertIsInstance(exec_transform_task, task_lib.ExecNodeTask)

    # Mark transform execution complete, but FAILED.
    self._finish_node_execution(
        use_task_queue, exec_transform_task, success=False
    )

    # In backfill mode, we don't retry failed executions, so Transform should
    # be stopped, since the backfill is complete.
    [update_transform_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=6,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0,
        expected_exec_nodes=[],
    )
    self.assertIsInstance(update_transform_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(pstate.NodeState.STOPPED, update_transform_task.state)

  def test_backfill_pure_service_node(self):
    backfill_token = 'backfill-20230227-180505-123456'
    test_utils.get_or_create_pipeline_state(
        self._mlmd_connection, self._pipeline
    )
    # Put ExampleGen in backfill mode.
    with pstate.PipelineState.load(
        self._mlmd_connection,
        task_lib.PipelineUid.from_pipeline(self._pipeline),
    ) as pipeline_state:
      example_gen_node = task_lib.NodeUid.from_node(
          self._pipeline, node_proto_view.get_view(self._example_gen)
      )
      with pipeline_state.node_state_update_context(
          example_gen_node
      ) as node_state:
        node_state.update(
            pstate.NodeState.STARTING,
            backfill_token=backfill_token,
        )
    # Generate a RUNNING task for ExampleGen backfill.
    [running_example_gen_task] = self._generate_and_test(
        use_task_queue=False,
        num_initial_executions=0,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0,
        expected_exec_nodes=[],
    )

    self.assertIsInstance(
        running_example_gen_task, task_lib.UpdateNodeStateTask
    )
    self.assertEqual(running_example_gen_task.state, pstate.NodeState.RUNNING)
    self.assertEqual(running_example_gen_task.backfill_token, backfill_token)
    self._mock_service_job_manager.ensure_node_services.assert_has_calls([
        mock.call(
            mock.ANY,
            self._example_gen.node_info.id,
            backfill_token,
        ),
    ])

    # Mark ExampleGen backfill service job as COMPLETED.
    def _backfill_completes(
        unused_pipeline_state, node_id, unused_backfill_token=''
    ):
      if node_id == self._example_gen.node_info.id:
        return service_jobs.ServiceStatus.SUCCESS

    self._mock_service_job_manager.reset_mock()
    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _backfill_completes
    )

    # Generate a STOPPED task after ExampleGen backfill completes.
    [stopped_example_gen_task] = self._generate_and_test(
        use_task_queue=False,
        num_initial_executions=0,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0,
        expected_exec_nodes=[],
    )
    self.assertIsInstance(
        stopped_example_gen_task, task_lib.UpdateNodeStateTask
    )
    self.assertEqual(stopped_example_gen_task.state, pstate.NodeState.STOPPED)
    self.assertEqual(stopped_example_gen_task.backfill_token, '')
    self._mock_service_job_manager.ensure_node_services.assert_has_calls([
        mock.call(
            mock.ANY,
            self._example_gen.node_info.id,
            backfill_token,
        ),
    ])
    self._mock_service_job_manager.stop_node_services.assert_called_once_with(
        mock.ANY, self._example_gen.node_info.id
    )


if __name__ == '__main__':
  tf.test.main()
