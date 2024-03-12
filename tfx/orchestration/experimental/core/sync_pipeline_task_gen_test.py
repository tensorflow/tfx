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

import itertools
import os
from typing import Literal
import uuid

from absl.testing import parameterized
from absl.testing.absltest import mock
import tensorflow as tf
from tfx.dsl.compiler import constants as compiler_constants
from tfx.orchestration import data_types_utils
from tfx.orchestration.experimental.core import constants
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import pipeline_ops
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import sync_pipeline_task_gen as sptg
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.experimental.core.testing import test_sync_pipeline
from tfx.orchestration import mlmd_connection_manager as mlmd_cm
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib

from ml_metadata.proto import metadata_store_pb2


class SyncPipelineTaskGeneratorTest(test_utils.TfxTest, parameterized.TestCase):

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
    pipeline = self._make_pipeline(self._pipeline_root, str(uuid.uuid4()))
    self._pipeline = pipeline

    # Extracts components.
    self._example_gen = test_utils.get_node(pipeline, 'my_example_gen')
    self._stats_gen = test_utils.get_node(pipeline, 'my_statistics_gen')
    self._schema_gen = test_utils.get_node(pipeline, 'my_schema_gen')
    self._transform = test_utils.get_node(pipeline, 'my_transform')
    self._example_validator = test_utils.get_node(pipeline,
                                                  'my_example_validator')
    self._trainer = test_utils.get_node(pipeline, 'my_trainer')
    self._evaluator = test_utils.get_node(pipeline, 'my_evaluator')
    self._chore_a = test_utils.get_node(pipeline, 'chore_a')
    self._chore_b = test_utils.get_node(pipeline, 'chore_b')

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
      return service_jobs.ServiceStatus(
          code=service_jobs.ServiceStatusCode.SUCCESS
      )

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _default_ensure_node_services)

  def _make_pipeline(
      self,
      pipeline_root,
      pipeline_run_id,
      pipeline_type: Literal['standard', 'chore', 'lifetime'] = 'standard',
  ):
    if pipeline_type == 'standard':
      pipeline = test_sync_pipeline.create_pipeline()
    elif pipeline_type == 'chore':
      pipeline = test_sync_pipeline.create_chore_pipeline()
    elif pipeline_type == 'lifetime':
      pipeline = test_sync_pipeline.create_resource_lifetime_pipeline()
    else:
      raise ValueError(
          f'Unsupported pipeline type: {pipeline_type}. Supported types:'
          ' "standard", "chore", and "lifetime".'
      )

    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline, {
            compiler_constants.PIPELINE_ROOT_PARAMETER_NAME: pipeline_root,
            compiler_constants.PIPELINE_RUN_ID_PARAMETER_NAME: pipeline_run_id,
        })
    return pipeline

  def _start_processing(self, use_task_queue, exec_node_task):
    if use_task_queue:
      dequeued_task = self._task_queue.dequeue()
      self.assertEqual(exec_node_task.task_id, dequeued_task.task_id)

  def _finish_processing(self, use_task_queue, dequeued_task):
    if use_task_queue:
      self._task_queue.task_done(dequeued_task)

  def _finish_node_execution(self,
                             use_task_queue,
                             exec_node_task,
                             artifact_custom_properties=None):
    """Simulates successful execution of a node."""
    self._start_processing(use_task_queue, exec_node_task)
    test_utils.fake_execute_node(
        self._mlmd_connection,
        exec_node_task,
        artifact_custom_properties=artifact_custom_properties)
    self._finish_processing(use_task_queue, exec_node_task)

  def _generate(self,
                use_task_queue,
                ignore_update_node_state_tasks=False,
                fail_fast=False):
    return test_utils.run_generator(
        self._mlmd_cm,
        sptg.SyncPipelineTaskGenerator,
        self._pipeline,
        self._task_queue,
        use_task_queue,
        self._mock_service_job_manager,
        ignore_update_node_state_tasks=ignore_update_node_state_tasks,
        fail_fast=fail_fast)

  def _run_next(self,
                use_task_queue,
                expect_nodes,
                finish_nodes=None,
                artifact_custom_properties=None,
                fail_fast=False):
    """Runs a complete cycle of task generation and simulating their completion.

    Args:
      use_task_queue: Whether to use task queue.
      expect_nodes: List of nodes whose task generation is expected.
      finish_nodes: List of nodes whose completion should be simulated. If
        `None` (default), all of `expect_nodes` will be finished.
      artifact_custom_properties: A dict of custom properties to attach to the
        output artifacts.
      fail_fast: If `True`, pipeline is aborted immediately if any node fails.
    """
    tasks = self._generate(use_task_queue, True, fail_fast=fail_fast)
    for task in tasks:
      self.assertIsInstance(task, task_lib.ExecNodeTask)
    expected_node_ids = [n.node_info.id for n in expect_nodes]
    task_node_ids = [task.node_uid.node_id for task in tasks]
    self.assertCountEqual(expected_node_ids, task_node_ids)
    finish_node_ids = set([n.node_info.id for n in finish_nodes]
                          if finish_nodes is not None else expected_node_ids)
    for task in tasks:
      if task.node_uid.node_id in finish_node_ids:
        self._finish_node_execution(
            use_task_queue,
            task,
            artifact_custom_properties=artifact_custom_properties)

  def _generate_and_test(self,
                         use_task_queue,
                         num_initial_executions,
                         num_tasks_generated,
                         num_new_executions,
                         num_active_executions,
                         pipeline=None,
                         expected_exec_nodes=None,
                         ignore_update_node_state_tasks=False,
                         fail_fast=False):
    """Generates tasks and tests the effects."""
    return test_utils.run_generator_and_test(
        self,
        self._mlmd_cm,
        sptg.SyncPipelineTaskGenerator,
        pipeline or self._pipeline,
        self._task_queue,
        use_task_queue,
        self._mock_service_job_manager,
        num_initial_executions=num_initial_executions,
        num_tasks_generated=num_tasks_generated,
        num_new_executions=num_new_executions,
        num_active_executions=num_active_executions,
        expected_exec_nodes=expected_exec_nodes,
        ignore_update_node_state_tasks=ignore_update_node_state_tasks,
        fail_fast=fail_fast)

  @parameterized.parameters(False, True)
  @mock.patch.object(task_gen_utils, 'update_external_artifact_type')
  def test_tasks_generated_when_upstream_done(
      self, use_task_queue, mock_update_external_artifact_type):
    """Tests that tasks are generated when upstream is done.

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

    # Generate once. Stats-gen task should be generated.
    [stats_gen_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=1,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._stats_gen],
        ignore_update_node_state_tasks=True)

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
        expected_exec_nodes=[self._schema_gen],
        ignore_update_node_state_tasks=True)

    # Finish schema-gen execution.
    self._finish_node_execution(use_task_queue, schema_gen_task)

    # Transform and ExampleValidator should both execute next.
    [example_validator_task, transform_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=3,
        num_tasks_generated=2,
        num_new_executions=2,
        num_active_executions=2,
        expected_exec_nodes=[self._example_validator, self._transform],
        ignore_update_node_state_tasks=True)

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
        expected_exec_nodes=[] if use_task_queue else [self._transform],
        ignore_update_node_state_tasks=True)
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
        expected_exec_nodes=[self._trainer],
        ignore_update_node_state_tasks=True)

    # Finish trainer execution.
    self._finish_node_execution(use_task_queue, trainer_task)

    # Test task-only dependencies: chore_a and chore_b nodes have no input or
    # output specs but should still be executed in the DAG order.
    [chore_a_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=6,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._chore_a],
        ignore_update_node_state_tasks=True)
    self._finish_node_execution(use_task_queue, chore_a_task)
    [chore_b_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=7,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._chore_b],
        ignore_update_node_state_tasks=True)
    self._finish_node_execution(use_task_queue, chore_b_task)

    # No more components to execute, FinalizePipelineTask should be generated.
    [finalize_task] = self._generate_and_test(
        use_task_queue,
        num_initial_executions=8,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0,
        ignore_update_node_state_tasks=True)
    self.assertIsInstance(finalize_task, task_lib.FinalizePipelineTask)
    self.assertEqual(status_lib.Code.OK, finalize_task.status.code)
    if use_task_queue:
      self.assertTrue(self._task_queue.is_empty())

    mock_update_external_artifact_type.assert_called()

  @parameterized.parameters(itertools.product((False, True), repeat=2))
  def test_pipeline_succeeds_when_terminal_nodes_succeed(
      self, use_task_queue, fail_fast):
    """Tests that pipeline is finalized only after terminal nodes are successful.

    Args:
      use_task_queue: If task queue is enabled, new tasks are only generated if
        a task with the same task_id does not already exist in the queue.
        `use_task_queue=False` is useful to test the case of task generation
        when task queue is empty (for eg: due to orchestrator restart).
      fail_fast: If `True`, pipeline is aborted immediately if any node fails.
    """
    # Start executing the pipeline:
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)

    self._run_next(use_task_queue, expect_nodes=[self._stats_gen])
    self._run_next(use_task_queue, expect_nodes=[self._schema_gen])

    # Both example-validator and transform are ready to execute.
    [example_validator_task, transform_task] = self._generate(
        use_task_queue, True, fail_fast=fail_fast)
    self.assertEqual(self._example_validator.node_info.id,
                     example_validator_task.node_uid.node_id)
    self.assertEqual(self._transform.node_info.id,
                     transform_task.node_uid.node_id)
    # Start processing (but do not finish) example-validator.
    self._start_processing(use_task_queue, example_validator_task)
    # But finish transform which is in the same layer.
    self._finish_node_execution(use_task_queue, transform_task)

    # Readability note: below, example-validator task should continue to be
    # generated when not using task queue because the execution is active.

    # Trainer and downstream nodes can execute as transform is finished.
    self._run_next(
        use_task_queue,
        expect_nodes=[self._trainer]
        if use_task_queue else [self._example_validator, self._trainer],
        finish_nodes=[self._trainer],
        fail_fast=fail_fast)
    self._run_next(
        use_task_queue,
        expect_nodes=[self._chore_a]
        if use_task_queue else [self._example_validator, self._chore_a],
        finish_nodes=[self._chore_a],
        fail_fast=fail_fast)
    self._run_next(
        use_task_queue,
        expect_nodes=[self._chore_b]
        if use_task_queue else [self._example_validator, self._chore_b],
        finish_nodes=[self._chore_b],
        fail_fast=fail_fast)
    self._run_next(
        use_task_queue,
        expect_nodes=[] if use_task_queue else [self._example_validator],
        finish_nodes=[],
        fail_fast=fail_fast)

    # FinalizePipelineTask is generated only after example-validator finishes.
    test_utils.fake_execute_node(self._mlmd_connection, example_validator_task)
    self._finish_processing(use_task_queue, example_validator_task)
    [finalize_task] = self._generate(use_task_queue, True, fail_fast=fail_fast)
    self.assertIsInstance(finalize_task, task_lib.FinalizePipelineTask)
    self.assertEqual(status_lib.Code.OK, finalize_task.status.code)

  def test_terminal_nodes_with_partial_run(self):
    """Tests that nodes with only skipped downstream nodes are terminal nodes."""
    # Check the expected skipped and terminal nodes.
    self._example_gen.execution_options.skip.SetInParent()
    self._chore_a.execution_options.skip.SetInParent()
    self._chore_b.execution_options.skip.SetInParent()
    self._evaluator.execution_options.skip.SetInParent()

    with self._mlmd_connection as m:
      pipeline_state = test_utils.get_or_create_pipeline_state(
          m, self._pipeline
      )
      with pipeline_state:
        node_states_dict = pipeline_state.get_node_states_dict()
    expected_skipped_node_ids = {
        'my_example_gen', 'chore_a', 'chore_b', 'my_evaluator'
    }
    self.assertEqual(
        expected_skipped_node_ids, sptg._skipped_node_ids(node_states_dict)
    )

    test_utils.fake_cached_example_gen_run(self._mlmd_connection,
                                           self._example_gen)
    self._run_next(False, expect_nodes=[self._stats_gen])
    self._run_next(False, expect_nodes=[self._schema_gen])
    self._run_next(
        False, expect_nodes=[self._example_validator, self._transform])
    self._run_next(False, expect_nodes=[self._trainer])
    # All runnable nodes executed, finalization task should be produced.
    [finalize_task] = self._generate(False, True)
    self.assertIsInstance(finalize_task, task_lib.FinalizePipelineTask)

  def test_terminal_nodes_with_partial_run_and_programatically_skipped(self):
    """Tests that nodes with only skipped downstream nodes are terminal nodes.

    Since we mark SKIPPED nodes as "succesful" we should make sure that the
    parent nodes of SKIPPED (or SKIPPED_PARTIAL_RUN) nodes are considered as
    terminal nodes so the pipeline will not finish prematurely.

    There was a bug (b/282034382) were we only treated SKIPPED_PARTIAL_RUN nodes
    as "skipped" so for nodes that were SKIPPED programtically would still be
    treated as terminal nodes, causing some pipelines to pre-maturely finish.
    """
    # Check the expected skipped and terminal nodes.
    self._example_gen.execution_options.skip.SetInParent()
    self._chore_a.execution_options.skip.SetInParent()
    self._chore_b.execution_options.skip.SetInParent()
    self._evaluator.execution_options.skip.SetInParent()

    # Mark trainer as programatically skipped, not as part of the partial run.
    with self._mlmd_connection as m:
      pipeline_state = test_utils.get_or_create_pipeline_state(
          m, self._pipeline
      )
      with pipeline_state:
        with pipeline_state.node_state_update_context(
            task_lib.NodeUid.from_node(self._pipeline, self._trainer)
        ) as node_state:
          assert node_state.is_programmatically_skippable()
          node_state.update(
              pstate.NodeState.SKIPPED,
              status_lib.Status(
                  code=status_lib.Code.OK,
                  message='Node skipped by client request.',
              ),
          )
        node_states_dict = pipeline_state.get_node_states_dict()

    expected_skipped_node_ids = {
        'my_example_gen',
        'chore_a',
        'chore_b',
        'my_evaluator',
        'my_trainer',
    }
    self.assertEqual(
        expected_skipped_node_ids, sptg._skipped_node_ids(node_states_dict)
    )

    # Start executing the pipeline:
    test_utils.fake_cached_example_gen_run(
        self._mlmd_connection, self._example_gen
    )
    self._run_next(False, expect_nodes=[self._stats_gen])
    self._run_next(False, expect_nodes=[self._schema_gen])

    # Trigger PAUSE on transform so it doesn't get run next.
    with self._mlmd_connection as m:
      pipeline_state = test_utils.get_or_create_pipeline_state(
          m, self._pipeline
      )
      with pipeline_state:
        with pipeline_state.node_state_update_context(
            task_lib.NodeUid.from_node(self._pipeline, self._transform)
        ) as node_state:
          assert node_state.is_stoppable()
          node_state.update(
              pstate.NodeState.STOPPING,
              status_lib.Status(
                  code=status_lib.Code.CANCELLED,
                  message='Cancellation requested by client.',
              ),
          )

    # Let example_validator "finish running".
    self._run_next(False, expect_nodes=[self._example_validator])

    # All tasks that can be run have been run, assume nothing happens since
    # transform is paused.
    tasks = self._generate(False, True)
    self.assertEmpty(tasks)

    # Pause the pipeline
    with self._mlmd_connection as m:
      pipeline_state = test_utils.get_or_create_pipeline_state(
          m, self._pipeline
      )
      with pipeline_state:
        pipeline_state.initiate_stop(
            status_lib.Status(
                code=status_lib.Code.CANCELLED,
                message='Cancellation requested by client.',
            ),
        )
    # All tasks that can be run have been run, assume nothing happens since
    # transform is paused.
    tasks = self._generate(False, True)
    self.assertEmpty(tasks)

    # Unpause just pipeline and transform and make sure pipeline will not
    # finalize.
    with self._mlmd_connection as m:
      pipeline_state = test_utils.get_or_create_pipeline_state(
          m, self._pipeline
      )
      with pipeline_state:
        pipeline_state.initiate_resume()

    tasks = self._generate(False, True)
    self.assertEmpty(tasks)

    # Unpause transform and make sure pipeline can continue as expected.
    with self._mlmd_connection as m:
      pipeline_state = test_utils.get_or_create_pipeline_state(
          m, self._pipeline
      )
      with pipeline_state:
        with pipeline_state.node_state_update_context(
            task_lib.NodeUid.from_node(self._pipeline, self._transform)
        ) as node_state:
          node_state.update(
              pstate.NodeState.STARTED,
              status_lib.Status(
                  code=status_lib.Code.OK,
              ),
          )

    self._run_next(False, expect_nodes=[self._transform])
    # All runnable nodes executed, finalization task should be produced.
    [finalize_task] = self._generate(False, True)
    self.assertIsInstance(finalize_task, task_lib.FinalizePipelineTask)

  def test_service_job_running(self):
    """Tests task generation when example-gen service job is still running."""

    def _ensure_node_services(unused_pipeline_state, node_id):
      self.assertEqual('my_example_gen', node_id)
      return service_jobs.ServiceStatus(
          code=service_jobs.ServiceStatusCode.RUNNING
      )

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _ensure_node_services)
    [task] = self._generate_and_test(
        True,
        num_initial_executions=0,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0)
    self.assertIsInstance(task, task_lib.UpdateNodeStateTask)
    self.assertEqual('my_example_gen', task.node_uid.node_id)
    self.assertEqual(pstate.NodeState.RUNNING, task.state)

  def test_service_job_success(self):
    """Tests task generation when example-gen service job succeeds."""
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)

    [eg_update_node_state_task, sg_update_node_state_task,
     sg_exec_node_task] = self._generate_and_test(
         True,
         num_initial_executions=1,
         num_tasks_generated=3,
         num_new_executions=1,
         num_active_executions=1,
         expected_exec_nodes=[self._stats_gen])
    self.assertIsInstance(eg_update_node_state_task,
                          task_lib.UpdateNodeStateTask)
    self.assertEqual('my_example_gen',
                     eg_update_node_state_task.node_uid.node_id)
    self.assertEqual(pstate.NodeState.COMPLETE, eg_update_node_state_task.state)
    self.assertIsInstance(sg_update_node_state_task,
                          task_lib.UpdateNodeStateTask)
    self.assertEqual('my_statistics_gen',
                     sg_update_node_state_task.node_uid.node_id)
    self.assertEqual(pstate.NodeState.RUNNING, sg_update_node_state_task.state)
    self.assertIsInstance(sg_exec_node_task, task_lib.ExecNodeTask)

  @parameterized.parameters(False, True)
  def test_service_job_failed(self, fail_fast):
    """Tests task generation when example-gen service job fails."""

    def _ensure_node_services(unused_pipeline_state, node_id):
      self.assertEqual('my_example_gen', node_id)
      return service_jobs.ServiceStatus(
          code=service_jobs.ServiceStatusCode.FAILED,
          msg='foobar error',
      )

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _ensure_node_services)
    [update_node_state_task, finalize_task] = self._generate_and_test(
        True,
        num_initial_executions=0,
        num_tasks_generated=2,
        num_new_executions=0,
        num_active_executions=0,
        fail_fast=fail_fast)
    self.assertIsInstance(update_node_state_task, task_lib.UpdateNodeStateTask)
    self.assertEqual('my_example_gen', update_node_state_task.node_uid.node_id)
    self.assertEqual(pstate.NodeState.FAILED, update_node_state_task.state)
    self.assertEqual(
        status_lib.Code.UNKNOWN, update_node_state_task.status.code
    )
    self.assertEqual(
        'service job failed; error message: foobar error',
        update_node_state_task.status.message,
    )
    self.assertIsInstance(finalize_task, task_lib.FinalizePipelineTask)
    self.assertEqual(status_lib.Code.UNKNOWN, finalize_task.status.code)

  def test_node_success(self):
    """Tests task generation when a node execution succeeds."""
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)

    [stats_gen_task] = self._generate_and_test(
        False,
        num_initial_executions=1,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        ignore_update_node_state_tasks=True)

    # Finish stats-gen execution.
    self._finish_node_execution(False, stats_gen_task)

    [
        stats_gen_update_node_state_task, schema_gen_update_node_state_task,
        schema_gen_exec_node_task
    ] = self._generate_and_test(
        False,
        num_initial_executions=2,
        num_tasks_generated=3,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._schema_gen])
    self.assertIsInstance(stats_gen_update_node_state_task,
                          task_lib.UpdateNodeStateTask)
    self.assertEqual('my_statistics_gen',
                     stats_gen_update_node_state_task.node_uid.node_id)
    self.assertEqual(pstate.NodeState.COMPLETE,
                     stats_gen_update_node_state_task.state)
    self.assertIsInstance(schema_gen_update_node_state_task,
                          task_lib.UpdateNodeStateTask)
    self.assertEqual('my_schema_gen',
                     schema_gen_update_node_state_task.node_uid.node_id)
    self.assertEqual(pstate.NodeState.RUNNING,
                     schema_gen_update_node_state_task.state)
    self.assertIsInstance(schema_gen_exec_node_task, task_lib.ExecNodeTask)

  @parameterized.parameters(False, True)
  def test_node_failed(self, fail_fast):
    """Tests task generation when a node registers a failed execution."""
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)

    [stats_gen_task] = self._generate_and_test(
        False,
        num_initial_executions=1,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        ignore_update_node_state_tasks=True,
        fail_fast=fail_fast)
    self.assertEqual(
        task_lib.NodeUid.from_node(self._pipeline, self._stats_gen),
        stats_gen_task.node_uid)
    with self._mlmd_connection as m:
      with mlmd_state.mlmd_execution_atomic_op(
          m, stats_gen_task.execution_id) as stats_gen_exec:
        # Fail stats-gen execution.
        stats_gen_exec.last_known_state = metadata_store_pb2.Execution.FAILED
        data_types_utils.set_metadata_value(
            stats_gen_exec.custom_properties[
                constants.EXECUTION_ERROR_CODE_KEY
            ],
            status_lib.Code.UNAVAILABLE,
        )
        data_types_utils.set_metadata_value(
            stats_gen_exec.custom_properties[constants.EXECUTION_ERROR_MSG_KEY],
            'foobar error',
        )

    # Test generation of FinalizePipelineTask.
    [update_node_state_task, finalize_task] = self._generate_and_test(
        True,
        num_initial_executions=2,
        num_tasks_generated=2,
        num_new_executions=0,
        num_active_executions=0,
        fail_fast=fail_fast)
    self.assertIsInstance(update_node_state_task, task_lib.UpdateNodeStateTask)
    self.assertEqual('my_statistics_gen',
                     update_node_state_task.node_uid.node_id)
    self.assertEqual(pstate.NodeState.FAILED, update_node_state_task.state)
    self.assertEqual(
        status_lib.Code.UNAVAILABLE, update_node_state_task.status.code
    )
    self.assertRegexMatch(update_node_state_task.status.message,
                          ['foobar error'])
    self.assertIsInstance(finalize_task, task_lib.FinalizePipelineTask)
    self.assertEqual(status_lib.Code.UNAVAILABLE, finalize_task.status.code)
    self.assertRegexMatch(finalize_task.status.message, ['foobar error'])

  @parameterized.parameters(False, True)
  def test_task_generation_when_node_stopped(self, stop_stats_gen):
    """Tests stopped nodes are ignored when generating tasks."""
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)

    num_initial_executions = 1
    if stop_stats_gen:
      num_tasks_generated = 0
      num_new_executions = 0
      num_active_executions = 0
      with self._mlmd_connection as m:
        pipeline_state = test_utils.get_or_create_pipeline_state(
            m, self._pipeline)
        with pipeline_state:
          with pipeline_state.node_state_update_context(
              task_lib.NodeUid.from_node(self._pipeline,
                                         self._stats_gen)) as node_state:
            node_state.update(pstate.NodeState.STOPPING,
                              status_lib.Status(code=status_lib.Code.CANCELLED))
    else:
      num_tasks_generated = 1
      num_new_executions = 1
      num_active_executions = 1
    tasks = self._generate_and_test(
        True,
        num_initial_executions=num_initial_executions,
        num_tasks_generated=num_tasks_generated,
        num_new_executions=num_new_executions,
        num_active_executions=num_active_executions,
        ignore_update_node_state_tasks=True)
    self.assertLen(tasks, num_tasks_generated)

  def test_restart_node_cancelled_due_to_stopping(self):
    """Tests that a node previously cancelled due to stopping can be restarted."""
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)

    [stats_gen_task] = self._generate_and_test(
        False,
        num_initial_executions=1,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        ignore_update_node_state_tasks=True)
    node_uid = task_lib.NodeUid.from_node(self._pipeline, self._stats_gen)
    self.assertEqual(node_uid, stats_gen_task.node_uid)

    # Simulate stopping the node while it is under execution, which leads to
    # the node execution being cancelled.
    with self._mlmd_connection as m:
      with mlmd_state.mlmd_execution_atomic_op(
          m, stats_gen_task.execution_id) as stats_gen_exec:
        stats_gen_exec.last_known_state = metadata_store_pb2.Execution.CANCELED
        data_types_utils.set_metadata_value(
            stats_gen_exec.custom_properties[constants.EXECUTION_ERROR_MSG_KEY],
            'manually stopped')

    # Change state of node to STARTED.
    with self._mlmd_connection as m:
      pipeline_state = test_utils.get_or_create_pipeline_state(
          m, self._pipeline)
      with pipeline_state:
        with pipeline_state.node_state_update_context(node_uid) as node_state:
          node_state.update(pstate.NodeState.STARTED)

    # New execution should be created for any previously canceled node when the
    # node state is STARTED.
    [update_node_state_task, stats_gen_task] = self._generate_and_test(
        False,
        num_initial_executions=2,
        num_tasks_generated=2,
        num_new_executions=1,
        num_active_executions=1)
    self.assertIsInstance(update_node_state_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(node_uid, update_node_state_task.node_uid)
    self.assertEqual(pstate.NodeState.RUNNING, update_node_state_task.state)
    self.assertEqual(node_uid, stats_gen_task.node_uid)

  def test_restart_node_cancelled_due_to_stopping_with_foreach(self):
    """Tests that a node in ForEach previously cancelled can be restarted."""
    pipeline = test_sync_pipeline.create_pipeline_with_foreach()
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline,
        {
            compiler_constants.PIPELINE_ROOT_PARAMETER_NAME: (
                self._pipeline_root
            ),
            compiler_constants.PIPELINE_RUN_ID_PARAMETER_NAME: str(
                uuid.uuid4()
            ),
        },
    )
    example_gen = test_utils.get_node(pipeline, 'my_example_gen')
    stats_gen = test_utils.get_node(pipeline, 'my_statistics_gen_in_foreach')

    # Simulates that ExampleGen has processed two spans.
    test_utils.fake_example_gen_run(self._mlmd_connection, example_gen, 1, 1)
    test_utils.fake_example_gen_run(self._mlmd_connection, example_gen, 2, 1)

    # StatsGen should have two executions.
    [stats_gen_task] = self._generate_and_test(
        False,
        num_initial_executions=2,
        num_tasks_generated=1,
        num_new_executions=2,
        num_active_executions=2,
        ignore_update_node_state_tasks=True,
        pipeline=pipeline,
    )
    stats_gen_node_uid = task_lib.NodeUid.from_node(pipeline, stats_gen)
    self.assertEqual(stats_gen_node_uid, stats_gen_task.node_uid)

    with self._mlmd_connection as m:
      # Simulates that the first execution of StatsGen is completed.
      with mlmd_state.mlmd_execution_atomic_op(
          m, stats_gen_task.execution_id
      ) as e:
        e.last_known_state = metadata_store_pb2.Execution.COMPLETE

      stats_gen_execution_type = [
          t for t in m.store.get_execution_types() if 'statistics_gen' in t.name
      ][0]
      executions = m.store.get_executions_by_type(stats_gen_execution_type.name)
      self.assertLen(executions, 2)

      # Simulates that all other uncompleted executions of StatsGen is CANCELED.
      with mlmd_state.mlmd_execution_atomic_op(m, executions[1].id) as e:
        e.last_known_state = metadata_store_pb2.Execution.CANCELED

      # Makes sure that at this point there are 2 executioins for StatsGen
      # One of them is completed, while the other is canceled.
      executions = m.store.get_executions_by_type(stats_gen_execution_type.name)
      self.assertLen(executions, 2)
      self.assertEqual(
          executions[0].last_known_state, metadata_store_pb2.Execution.COMPLETE
      )
      self.assertEqual(
          executions[1].last_known_state, metadata_store_pb2.Execution.CANCELED
      )

    # Changes node state of StatsGen to STARTED.
    with self._mlmd_connection as m:
      pipeline_state = test_utils.get_or_create_pipeline_state(m, pipeline)
      with pipeline_state:
        with pipeline_state.node_state_update_context(
            stats_gen_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.STARTED)

    # 1 new executions should be created for stats_gen.
    [stats_gen_task] = self._generate_and_test(
        False,
        num_initial_executions=4,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        ignore_update_node_state_tasks=True,
        pipeline=pipeline,
    )
    self.assertEqual(stats_gen_node_uid, stats_gen_task.node_uid)
    self.assertIsInstance(stats_gen_task, task_lib.ExecNodeTask)

  def test_restart_node_cancelled_due_to_fail_with_foreach(self):
    """Tests that a node in ForEach previously failed can be restarted."""
    pipeline = test_sync_pipeline.create_pipeline_with_foreach()
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline,
        {
            compiler_constants.PIPELINE_ROOT_PARAMETER_NAME: (
                self._pipeline_root
            ),
            compiler_constants.PIPELINE_RUN_ID_PARAMETER_NAME: str(
                uuid.uuid4()
            ),
        },
    )
    example_gen = test_utils.get_node(pipeline, 'my_example_gen')
    stats_gen = test_utils.get_node(pipeline, 'my_statistics_gen_in_foreach')

    # Simulates that ExampleGen has processed two spans.
    test_utils.fake_example_gen_run(self._mlmd_connection, example_gen, 1, 1)
    test_utils.fake_example_gen_run(self._mlmd_connection, example_gen, 2, 1)

    # StatsGen should have two executions.
    [stats_gen_task] = self._generate_and_test(
        False,
        num_initial_executions=2,
        num_tasks_generated=1,
        num_new_executions=2,
        num_active_executions=2,
        ignore_update_node_state_tasks=True,
        pipeline=pipeline,
    )
    stats_gen_node_uid = task_lib.NodeUid.from_node(pipeline, stats_gen)
    self.assertEqual(stats_gen_node_uid, stats_gen_task.node_uid)

    with self._mlmd_connection as m:
      # Simulates that the first execution of StatsGen is FAILED.
      with mlmd_state.mlmd_execution_atomic_op(
          m, stats_gen_task.execution_id
      ) as e:
        e.last_known_state = metadata_store_pb2.Execution.FAILED

      stats_gen_execution_type = [
          t for t in m.store.get_execution_types() if 'statistics_gen' in t.name
      ][0]
      executions = m.store.get_executions_by_type(stats_gen_execution_type.name)
      self.assertLen(executions, 2)

      # Simulates that all other uncompleted executions of StatsGen is CANCELED.
      with mlmd_state.mlmd_execution_atomic_op(m, executions[1].id) as e:
        e.last_known_state = metadata_store_pb2.Execution.CANCELED

      # Makes sure that at this point there are 2 executioins for StatsGen
      # One of them is failed, while the other is canceled.
      executions = m.store.get_executions_by_type(stats_gen_execution_type.name)
      self.assertLen(executions, 2)
      self.assertEqual(
          executions[0].last_known_state, metadata_store_pb2.Execution.FAILED
      )
      self.assertEqual(
          executions[1].last_known_state, metadata_store_pb2.Execution.CANCELED
      )

    # Changes node state of StatsGen to STARTED.
    with self._mlmd_connection as m:
      pipeline_state = test_utils.get_or_create_pipeline_state(m, pipeline)
      with pipeline_state:
        with pipeline_state.node_state_update_context(
            stats_gen_node_uid
        ) as node_state:
          node_state.update(pstate.NodeState.STARTED)

    # 1 new task should be created for stats_gen.
    [stats_gen_task] = self._generate_and_test(
        False,
        num_initial_executions=4,
        num_tasks_generated=1,
        num_new_executions=2,
        num_active_executions=2,
        ignore_update_node_state_tasks=True,
        pipeline=pipeline,
    )
    self.assertEqual(stats_gen_node_uid, stats_gen_task.node_uid)
    self.assertIsInstance(stats_gen_task, task_lib.ExecNodeTask)

    # Now there are 4 executions for stats_gen.
    # The first 2 of them are old from last failure of the node.
    # The last 2 of them are newly created executions when the node is restarted
    executions = m.store.get_executions_by_type(stats_gen_execution_type.name)
    self.assertLen(executions, 4)
    self.assertEqual(
        executions[0].last_known_state, metadata_store_pb2.Execution.FAILED
    )
    self.assertEqual(
        executions[1].last_known_state, metadata_store_pb2.Execution.CANCELED
    )
    self.assertEqual(
        executions[2].last_known_state, metadata_store_pb2.Execution.RUNNING
    )
    self.assertEqual(
        executions[3].last_known_state, metadata_store_pb2.Execution.NEW
    )

  @parameterized.parameters(False, True)
  def test_conditional_execution(self, evaluate):
    """Tests conditionals in the pipeline.

    Args:
      evaluate: Whether to run the conditional evaluator.
    """

    # Start executing the pipeline:

    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)

    self._run_next(False, expect_nodes=[self._stats_gen])
    self._run_next(False, expect_nodes=[self._schema_gen])
    self._run_next(
        False, expect_nodes=[self._example_validator, self._transform])

    # Evaluator is run conditionally based on whether the Model artifact
    # produced by the trainer has a custom property evaluate=1.
    self._run_next(
        False,
        expect_nodes=[self._trainer],
        artifact_custom_properties={'evaluate': 1} if evaluate else None)

    tasks = self._generate(False)
    [evaluator_update_node_state_task] = [
        t for t in tasks if isinstance(t, task_lib.UpdateNodeStateTask) and
        t.node_uid.node_id == 'my_evaluator'
    ]
    self.assertEqual(
        pstate.NodeState.RUNNING if evaluate else pstate.NodeState.SKIPPED,
        evaluator_update_node_state_task.state)

    exec_node_tasks = [t for t in tasks if isinstance(t, task_lib.ExecNodeTask)]
    if evaluate:
      [chore_a_exec_node_task, evaluator_exec_node_task] = exec_node_tasks
      self.assertEqual('chore_a', chore_a_exec_node_task.node_uid.node_id)
      self.assertEqual('my_evaluator',
                       evaluator_exec_node_task.node_uid.node_id)
      self._finish_node_execution(False, chore_a_exec_node_task)
      self._finish_node_execution(False, evaluator_exec_node_task)
    else:
      [chore_a_exec_node_task] = exec_node_tasks
      self.assertEqual('chore_a', chore_a_exec_node_task.node_uid.node_id)
      self._finish_node_execution(False, chore_a_exec_node_task)

    self._run_next(False, expect_nodes=[self._chore_b])

    # All nodes executed, finalization task should be produced.
    [finalize_task] = self._generate(False, True)
    self.assertIsInstance(finalize_task, task_lib.FinalizePipelineTask)

  @parameterized.parameters(False, True)
  def test_pipeline_failure_strategies(self, fail_fast):
    """Tests pipeline failure strategies."""
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)

    self._run_next(False, expect_nodes=[self._stats_gen], fail_fast=fail_fast)
    self._run_next(False, expect_nodes=[self._schema_gen], fail_fast=fail_fast)

    # Both example-validator and transform are ready to execute.
    [example_validator_task, transform_task] = self._generate(
        False, True, fail_fast=fail_fast)
    self.assertEqual(self._example_validator.node_info.id,
                     example_validator_task.node_uid.node_id)
    self.assertEqual(self._transform.node_info.id,
                     transform_task.node_uid.node_id)

    # Simulate Transform success.
    self._finish_node_execution(False, transform_task)

    # But fail example-validator.
    with self._mlmd_connection as m:
      with mlmd_state.mlmd_execution_atomic_op(
          m, example_validator_task.execution_id) as ev_exec:
        # Fail stats-gen execution.
        ev_exec.last_known_state = metadata_store_pb2.Execution.FAILED
        data_types_utils.set_metadata_value(
            ev_exec.custom_properties[constants.EXECUTION_ERROR_CODE_KEY],
            status_lib.Code.PERMISSION_DENIED,
        )
        data_types_utils.set_metadata_value(
            ev_exec.custom_properties[constants.EXECUTION_ERROR_MSG_KEY],
            'example-validator error',
        )

    if fail_fast:
      # Pipeline run should immediately fail because example-validator failed.
      [finalize_task] = self._generate(False, True, fail_fast=fail_fast)
      self.assertIsInstance(finalize_task, task_lib.FinalizePipelineTask)
      self.assertEqual(
          status_lib.Code.PERMISSION_DENIED, finalize_task.status.code
      )
    else:
      # Trainer and downstream nodes can execute as transform has finished.
      # example-validator failure does not impact them as it is not upstream.
      # Pipeline run will still fail but when no more progress can be made.
      self._run_next(False, expect_nodes=[self._trainer], fail_fast=fail_fast)
      self._run_next(False, expect_nodes=[self._chore_a], fail_fast=fail_fast)
      self._run_next(False, expect_nodes=[self._chore_b], fail_fast=fail_fast)
      [finalize_task] = self._generate(False, True, fail_fast=fail_fast)
      self.assertIsInstance(finalize_task, task_lib.FinalizePipelineTask)
      self.assertEqual(
          status_lib.Code.PERMISSION_DENIED, finalize_task.status.code
      )

  @parameterized.parameters(
      (
          'chore_a',
          pipeline_pb2.NodeExecutionOptions(node_success_optional=True),
      ),
      (
          'chore_b',
          pipeline_pb2.NodeExecutionOptions(
              strategy=pipeline_pb2.NodeExecutionOptions.ALL_UPSTREAM_NODES_COMPLETED
          ),
      ),
      (
          'chore_b',
          pipeline_pb2.NodeExecutionOptions(
              strategy=pipeline_pb2.NodeExecutionOptions.LAZILY_ALL_UPSTREAM_NODES_COMPLETED
          ),
      ),
  )
  def test_node_triggering_strategies(self, node_id, node_execution_options):
    """Tests node triggering strategies."""
    if node_id == 'chore_a':
      # Set chore_a's node_success_optional bit to True.
      self._chore_a.execution_options.CopyFrom(node_execution_options)
    elif node_id == 'chore_b':
      # Set chore_b's node triggering strategy to all upstream node succeeded.
      self._chore_b.execution_options.CopyFrom(node_execution_options)
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)
    self._run_next(False, expect_nodes=[self._stats_gen])
    self._run_next(False, expect_nodes=[self._schema_gen])
    self._run_next(
        False, expect_nodes=[self._example_validator, self._transform])
    self._run_next(False, expect_nodes=[self._trainer])
    [chore_a_task] = self._generate_and_test(
        False,
        num_initial_executions=6,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        ignore_update_node_state_tasks=True,
        fail_fast=False)
    with self._mlmd_connection as m:
      with mlmd_state.mlmd_execution_atomic_op(
          m, chore_a_task.execution_id) as chore_a_exec:
        # Fail chore a execution.
        chore_a_exec.last_known_state = metadata_store_pb2.Execution.FAILED
        data_types_utils.set_metadata_value(
            chore_a_exec.custom_properties[constants.EXECUTION_ERROR_MSG_KEY],
            'foobar error')
        data_types_utils.set_metadata_value(
            chore_a_exec.custom_properties[constants.EXECUTION_ERROR_CODE_KEY],
            status_lib.Code.RESOURCE_EXHAUSTED,
        )

    # Despite upstream node failure, chore b proceeds because:
    # 1) It's failure strategy is ALL_UPSTREAM_NODES_COMPLETED, or
    # 2) chore a's `success_optional` bit is set to True.
    self._run_next(False, expect_nodes=[self._chore_b])
    # All runnable nodes executed, finalization task should be produced.
    [finalize_task] = self._generate(False, True)
    self.assertIsInstance(finalize_task, task_lib.FinalizePipelineTask)

    # Pipeline should only be ok if the failed node is optional.
    if node_execution_options.node_success_optional:
      self.assertEqual(status_lib.Code.OK, finalize_task.status.code)
    else:
      self.assertEqual(
          status_lib.Code.RESOURCE_EXHAUSTED, finalize_task.status.code
      )

  def test_component_retry(self):
    """Tests component retry."""
    test_utils.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1,
                                    1)
    self._stats_gen.execution_options.max_execution_retries = 2
    [exec_node_task] = self._generate(False, True, fail_fast=True)
    self.assertEqual(self._stats_gen.node_info.id,
                     exec_node_task.node_uid.node_id)

    # Simulate fail and rerun StatsGen twice.
    for _ in range(self._stats_gen.execution_options.max_execution_retries):
      # Simulate StatsGen failure.
      with self._mlmd_connection as m:
        with mlmd_state.mlmd_execution_atomic_op(
            m, exec_node_task.execution_id) as ev_exec:
          ev_exec.last_known_state = metadata_store_pb2.Execution.FAILED

      # It should generate a ExecNodeTask due to retry.
      [update_node_task, exec_node_task] = self._generate(
          False, False, fail_fast=True)
      self.assertIsInstance(exec_node_task, task_lib.ExecNodeTask)
      self.assertIsInstance(update_node_task, task_lib.UpdateNodeStateTask)
      self.assertEqual(update_node_task.state, pstate.NodeState.RUNNING)

    # Fail StatsGen the third time.
    with self._mlmd_connection as m:
      with mlmd_state.mlmd_execution_atomic_op(
          m, exec_node_task.execution_id) as ev_exec:
        ev_exec.last_known_state = metadata_store_pb2.Execution.FAILED

    # Fail the pipeline since StatsGen can not retry anymore.
    [finalize_task] = self._generate(False, True, fail_fast=True)
    self.assertIsInstance(finalize_task, task_lib.FinalizePipelineTask)
    self.assertEqual(status_lib.Code.UNKNOWN, finalize_task.status.code)

  def test_component_retry_when_node_is_started(self):
    """Tests component retry when node is STARTED."""
    test_utils.fake_example_gen_run(
        self._mlmd_connection, self._example_gen, 1, 1
    )
    node_uid = task_lib.NodeUid.from_node(self._pipeline, self._stats_gen)

    self._stats_gen.execution_options.max_execution_retries = 2
    [exec_node_task] = self._generate(False, True, fail_fast=True)
    self.assertEqual(
        self._stats_gen.node_info.id, exec_node_task.node_uid.node_id
    )

    # Simulate fail and rerun StatsGen twice.
    for _ in range(self._stats_gen.execution_options.max_execution_retries):
      # Simulate StatsGen failure.
      with self._mlmd_connection as m:
        with mlmd_state.mlmd_execution_atomic_op(
            m, exec_node_task.execution_id
        ) as ev_exec:
          ev_exec.last_known_state = metadata_store_pb2.Execution.FAILED

      # It should generate a ExecNodeTask due to retry.
      [update_node_task, exec_node_task] = self._generate(
          False, False, fail_fast=True
      )
      self.assertIsInstance(exec_node_task, task_lib.ExecNodeTask)
      self.assertEqual(
          self._stats_gen.node_info.id, exec_node_task.node_uid.node_id
      )
      self.assertIsInstance(update_node_task, task_lib.UpdateNodeStateTask)
      self.assertEqual(update_node_task.state, pstate.NodeState.RUNNING)

    # Fail StatsGen the third time.
    with self._mlmd_connection as m:
      with mlmd_state.mlmd_execution_atomic_op(
          m, exec_node_task.execution_id
      ) as ev_exec:
        ev_exec.last_known_state = metadata_store_pb2.Execution.FAILED

    # Change state of node to STARTED.
    with self._mlmd_connection as m:
      pipeline_state = test_utils.get_or_create_pipeline_state(
          m, self._pipeline
      )
      with pipeline_state:
        with pipeline_state.node_state_update_context(node_uid) as node_state:
          node_state.update(pstate.NodeState.STARTED)

    # It should generate a ExecNodeTask due to state being STARTED.
    [update_node_task, exec_node_task] = self._generate(
        False, False, fail_fast=True
    )
    self.assertIsInstance(exec_node_task, task_lib.ExecNodeTask)
    self.assertEqual(
        self._stats_gen.node_info.id, exec_node_task.node_uid.node_id
    )
    self.assertIsInstance(update_node_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(update_node_task.state, pstate.NodeState.RUNNING)

  def _setup_for_chore_pipeline(self):
    pipeline = self._make_pipeline(
        self._pipeline_root, str(uuid.uuid4()), pipeline_type='chore'
    )
    self._pipeline = pipeline
    self.eg_1 = test_utils.get_node(pipeline, 'my_example_gen_1')
    self.eg_2 = test_utils.get_node(pipeline, 'my_example_gen_2')
    self.chore_a = test_utils.get_node(pipeline, 'chore_a')
    self.chore_b = test_utils.get_node(pipeline, 'chore_b')
    self.chore_c = test_utils.get_node(pipeline, 'chore_c')
    self.chore_d = test_utils.get_node(pipeline, 'chore_d')
    self.chore_e = test_utils.get_node(pipeline, 'chore_e')
    self.chore_f = test_utils.get_node(pipeline, 'chore_f')
    self.chore_g = test_utils.get_node(pipeline, 'chore_g')

  def test_lazy_execution(self):
    self._setup_for_chore_pipeline()

    # chore_a and chore_b can execute way earlier but should wait for chore_f
    self.chore_a.execution_options.strategy = (
        pipeline_pb2.NodeExecutionOptions.LAZILY_ALL_UPSTREAM_NODES_SUCCEEDED
    )
    self.chore_b.execution_options.strategy = (
        pipeline_pb2.NodeExecutionOptions.LAZILY_ALL_UPSTREAM_NODES_SUCCEEDED
    )

    # chore_d and chore_e are on the same level so they should execute at the
    # same time Also use LAZILY_ALL_UPSTREAM_NODES_COMPLETED to check both
    # strategies can work in the happy path.
    self.chore_d.execution_options.strategy = (
        pipeline_pb2.NodeExecutionOptions.LAZILY_ALL_UPSTREAM_NODES_COMPLETED
    )
    self.chore_e.execution_options.strategy = (
        pipeline_pb2.NodeExecutionOptions.LAZILY_ALL_UPSTREAM_NODES_COMPLETED
    )

    # chore_g is terminal and should execute normally.
    self.chore_g.execution_options.strategy = (
        pipeline_pb2.NodeExecutionOptions.ALL_UPSTREAM_NODES_COMPLETED
    )

    test_utils.fake_example_gen_run(self._mlmd_connection, self.eg_1, 1, 1)
    test_utils.fake_example_gen_run(self._mlmd_connection, self.eg_2, 1, 1)

    self._run_next(False, expect_nodes=[self.chore_d, self.chore_e])
    self._run_next(False, expect_nodes=[self.chore_f, self.chore_g])

    # Need to wait a cycle for chore_f to get marked as succesful.
    # TODO(kmonte): Figure out how to avoid this.
    self._run_next(False, expect_nodes=[])
    self._run_next(False, expect_nodes=[self.chore_a])
    self._run_next(False, expect_nodes=[self.chore_b])
    self._run_next(False, expect_nodes=[self.chore_c])

  def test_lazy_nodes_are_unrunnable_if_downstream_are_unrunnable(self):
    self._setup_for_chore_pipeline()
    # chore_a and chore_b can execute way earlier but should wait for chore_f
    self.chore_a.execution_options.strategy = (
        pipeline_pb2.NodeExecutionOptions.LAZILY_ALL_UPSTREAM_NODES_SUCCEEDED
    )
    self.chore_b.execution_options.strategy = (
        pipeline_pb2.NodeExecutionOptions.LAZILY_ALL_UPSTREAM_NODES_SUCCEEDED
    )
    test_utils.fake_example_gen_run(self._mlmd_connection, self.eg_1, 1, 1)
    test_utils.fake_example_gen_run(self._mlmd_connection, self.eg_2, 1, 1)
    self._run_next(False, expect_nodes=[self.chore_d, self.chore_e])

    [chore_f_task, chore_g_task] = self._generate_and_test(
        False,
        num_initial_executions=4,
        num_tasks_generated=2,
        num_new_executions=2,
        num_active_executions=2,
        ignore_update_node_state_tasks=True,
    )
    self.assertEqual(
        task_lib.NodeUid.from_node(self._pipeline, self.chore_g),
        chore_g_task.node_uid,
    )
    self.assertEqual(
        task_lib.NodeUid.from_node(self._pipeline, self.chore_f),
        chore_f_task.node_uid,
    )
    # G can succeed.
    with self._mlmd_connection as m:
      with mlmd_state.mlmd_execution_atomic_op(
          m, chore_g_task.execution_id
      ) as chore_g_exec:
        chore_g_exec.last_known_state = (
            metadata_store_pb2.Execution.State.COMPLETE
        )

    # F must fail, leaving C as unrunnable.
    with self._mlmd_connection as m:
      with mlmd_state.mlmd_execution_atomic_op(
          m, chore_f_task.execution_id
      ) as chore_f_exec:
        chore_f_exec.last_known_state = metadata_store_pb2.Execution.FAILED
        data_types_utils.set_metadata_value(
            chore_f_exec.custom_properties[constants.EXECUTION_ERROR_CODE_KEY],
            status_lib.Code.UNAVAILABLE,
        )
        data_types_utils.set_metadata_value(
            chore_f_exec.custom_properties[constants.EXECUTION_ERROR_MSG_KEY],
            'foobar error',
        )

    # Pipeline should fail due to there there being no more unrunnable nodes.
    [finalize_task] = self._generate(False, True)
    self.assertEqual(status_lib.Code.UNAVAILABLE, finalize_task.status.code)
    self.assertEqual('foobar error', finalize_task.status.message)

  def test_generate_tasks_for_node(self):
    pipeline = self._make_pipeline(
        self._pipeline_root, str(uuid.uuid4()), pipeline_type='chore'
    )
    self._pipeline = pipeline
    chore_b = test_utils.get_node(pipeline, 'chore_b')

    def id_tracked_fn():
      raise ValueError('Should not be called!')

    task_gen = sptg.SyncPipelineTaskGenerator(
        mlmd_connection_manager=self._mlmd_cm,
        is_task_id_tracked_fn=id_tracked_fn,
        service_job_manager=self._mock_service_job_manager,
    )
    chore_b_uid = task_lib.NodeUid.from_node(self._pipeline, chore_b)

    with self._mlmd_connection as m:
      pipeline_state = test_utils.get_or_create_pipeline_state(
          m, self._pipeline
      )
      tasks = task_gen.get_tasks_for_node(chore_b, pipeline_state)

    self.assertLen(tasks, 2)
    [update_task, exec_task] = tasks
    self.assertIsInstance(update_task, task_lib.UpdateNodeStateTask)
    self.assertEqual(update_task.state, pstate.NodeState.RUNNING)
    self.assertEqual(update_task.node_uid, chore_b_uid)
    self.assertIsInstance(exec_task, task_lib.ExecNodeTask)
    self.assertEqual(exec_task.node_uid, chore_b_uid)

  def _setup_for_resource_lifetime_pipeline(self):
    pipeline = self._make_pipeline(
        self._pipeline_root, str(uuid.uuid4()), pipeline_type='lifetime'
    )
    self._pipeline = pipeline
    self.start_a = test_utils.get_node(pipeline, 'start_a')
    self.start_b = test_utils.get_node(pipeline, 'start_b')
    self.worker = test_utils.get_node(pipeline, 'worker')
    self.end_b = test_utils.get_node(pipeline, 'end_b')
    self.end_a = test_utils.get_node(pipeline, 'end_a')

  def test_trigger_strategy_lifetime_end_when_subgraph_cannot_progress_multiple_lifetimes_only_worker_fails(
      self,
  ):
    self._setup_for_resource_lifetime_pipeline()

    test_utils.fake_example_gen_run(self._mlmd_connection, self.start_a, 1, 1)

    self._run_next(False, expect_nodes=[self.start_b])
    [worker_task] = self._generate_and_test(
        False,
        num_initial_executions=2,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        ignore_update_node_state_tasks=True,
    )
    self.assertEqual(
        task_lib.NodeUid.from_node(self._pipeline, self.worker),
        worker_task.node_uid,
    )

    with self._mlmd_connection as m:
      with mlmd_state.mlmd_execution_atomic_op(
          m, worker_task.execution_id
      ) as worker_b_exec:
        # Fail stats-gen execution.
        worker_b_exec.last_known_state = metadata_store_pb2.Execution.FAILED
        data_types_utils.set_metadata_value(
            worker_b_exec.custom_properties[constants.EXECUTION_ERROR_CODE_KEY],
            status_lib.Code.UNAVAILABLE,
        )
        data_types_utils.set_metadata_value(
            worker_b_exec.custom_properties[constants.EXECUTION_ERROR_MSG_KEY],
            'foobar error',
        )

    self._run_next(False, expect_nodes=[self.end_b])
    self._run_next(False, expect_nodes=[self.end_a])

    # Pipeline should fail due to chore_a having failed.
    [finalize_task] = self._generate(False, True)
    self.assertEqual(status_lib.Code.UNAVAILABLE, finalize_task.status.code)
    self.assertEqual('foobar error', finalize_task.status.message)

  def test_trigger_strategy_lifetime_end_when_subgraph_cannot_progress_multiple_lifetimes_inner_start_fails(
      self,
  ):
    self._setup_for_resource_lifetime_pipeline()

    test_utils.fake_example_gen_run(self._mlmd_connection, self.start_a, 1, 1)

    [start_b_task] = self._generate_and_test(
        False,
        num_initial_executions=1,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        ignore_update_node_state_tasks=True,
    )
    self.assertEqual(
        task_lib.NodeUid.from_node(self._pipeline, self.start_b),
        start_b_task.node_uid,
    )
    # Fail start_b execution
    with self._mlmd_connection as m:
      with mlmd_state.mlmd_execution_atomic_op(
          m, start_b_task.execution_id
      ) as start_b_exec:
        # Fail stats-gen execution.
        start_b_exec.last_known_state = metadata_store_pb2.Execution.FAILED
        data_types_utils.set_metadata_value(
            start_b_exec.custom_properties[constants.EXECUTION_ERROR_CODE_KEY],
            status_lib.Code.UNAVAILABLE,
        )
        data_types_utils.set_metadata_value(
            start_b_exec.custom_properties[constants.EXECUTION_ERROR_MSG_KEY],
            'foobar error',
        )

    self._run_next(False, expect_nodes=[])
    self._run_next(False, expect_nodes=[self.end_a])
    # Pipeline should fail due to chore_a having failed.
    [finalize_task] = self._generate(False, True)
    self.assertEqual(status_lib.Code.UNAVAILABLE, finalize_task.status.code)
    self.assertEqual('foobar error', finalize_task.status.message)

  def test_trigger_strategy_lifetime_end_when_subgraph_cannot_progress_pipeline_fails_when_start_node_fails(
      self,
  ):
    # This test is so that a pipeline will fail if:
    # 1. There are no nodes using the lifetime (only start and end)
    # 2. The start node fails.
    # We only care about start -> start_b -> worker for this case, where
    # worker.lifetime_start = start_b.
    self._setup_for_resource_lifetime_pipeline()
    self.worker.execution_options.resource_lifetime.lifetime_start = (
        self.start_b.node_info.id
    )

    # clear out the rest of the nodes - we don't care about them.
    self.end_b.execution_options.Clear()
    self.end_a.execution_options.Clear()

    test_utils.fake_example_gen_run(self._mlmd_connection, self.start_a, 1, 1)

    [start_b_task] = self._generate_and_test(
        False,
        num_initial_executions=1,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        ignore_update_node_state_tasks=True,
    )
    self.assertEqual(
        task_lib.NodeUid.from_node(self._pipeline, self.start_b),
        start_b_task.node_uid,
    )
    # Fail start_b execution
    with self._mlmd_connection as m:
      with mlmd_state.mlmd_execution_atomic_op(
          m, start_b_task.execution_id
      ) as start_b_exec:
        # Fail stats-gen execution.
        start_b_exec.last_known_state = metadata_store_pb2.Execution.FAILED
        data_types_utils.set_metadata_value(
            start_b_exec.custom_properties[constants.EXECUTION_ERROR_CODE_KEY],
            status_lib.Code.UNAVAILABLE,
        )
        data_types_utils.set_metadata_value(
            start_b_exec.custom_properties[constants.EXECUTION_ERROR_MSG_KEY],
            'foobar error',
        )

    # Pipeline should fail due to start_b having failed.
    [finalize_task] = self._generate(False, True)
    self.assertEqual(status_lib.Code.UNAVAILABLE, finalize_task.status.code)
    self.assertEqual('foobar error', finalize_task.status.message)

  def test_trigger_strategy_lifetime_end_with_start_node_not_upstream_of_failure(
      self,
  ):
    self._setup_for_chore_pipeline()

    self.chore_c.execution_options.strategy = (
        pipeline_pb2.NodeExecutionOptions.LIFETIME_END_WHEN_SUBGRAPH_CANNOT_PROGRESS
    )
    self.chore_c.execution_options.resource_lifetime.lifetime_start = (
        'my_example_gen_1'
    )

    test_utils.fake_example_gen_run(self._mlmd_connection, self.eg_1, 1, 1)
    test_utils.fake_example_gen_run(self._mlmd_connection, self.eg_2, 1, 1)

    [_, chore_d_task, _] = self._generate_and_test(
        False,
        num_initial_executions=2,
        num_tasks_generated=3,
        num_new_executions=3,
        num_active_executions=3,
        ignore_update_node_state_tasks=True,
    )
    self.assertEqual(
        task_lib.NodeUid.from_node(self._pipeline, self.chore_d),
        chore_d_task.node_uid,
    )

    # Fail chore_d execution
    with self._mlmd_connection as m:
      with mlmd_state.mlmd_execution_atomic_op(
          m, chore_d_task.execution_id
      ) as chore_d_exec:
        chore_d_exec.last_known_state = metadata_store_pb2.Execution.FAILED
        data_types_utils.set_metadata_value(
            chore_d_exec.custom_properties[constants.EXECUTION_ERROR_CODE_KEY],
            status_lib.Code.UNAVAILABLE,
        )
        data_types_utils.set_metadata_value(
            chore_d_exec.custom_properties[constants.EXECUTION_ERROR_MSG_KEY],
            'foobar error',
        )

    self._run_next(False, expect_nodes=[self.chore_a, self.chore_e])
    self._run_next(False, expect_nodes=[self.chore_b])

    # chore_c should run as all of its subgraph ancestors succeeded, failed,
    # or became unrunnable.
    self._run_next(False, expect_nodes=[self.chore_c])

    # Pipeline should fail due to chore_d having failed.
    [finalize_task] = self._generate(False, True)
    self.assertEqual(status_lib.Code.UNAVAILABLE, finalize_task.status.code)
    self.assertEqual('foobar error', finalize_task.status.message)

  def test_retry_with_pre_revive_executions(self):
    self._setup_for_resource_lifetime_pipeline()

    test_utils.fake_example_gen_run(self._mlmd_connection, self.start_a, 1, 1)
    self.start_b.execution_options.node_success_optional = True

    # Generate tasks for start_b and worker, and mark both as failed.
    for idx, next_node in enumerate([self.start_b, self.worker]):
      [next_node_task] = self._generate_and_test(
          False,
          num_initial_executions=1 + idx,
          num_tasks_generated=1,
          num_new_executions=1,
          num_active_executions=1,
          ignore_update_node_state_tasks=True,
      )
      self.assertEqual(
          task_lib.NodeUid.from_node(self._pipeline, next_node),
          next_node_task.node_uid,
      )
      with self._mlmd_connection as m:
        with mlmd_state.mlmd_execution_atomic_op(
            m, next_node_task.execution_id
        ) as next_node_exec:
          next_node_exec.last_known_state = metadata_store_pb2.Execution.FAILED

    self._run_next(False, expect_nodes=[self.end_b])
    self._run_next(False, expect_nodes=[self.end_a])
    [finalize_task_1] = self._generate(False, True)
    self.assertIsInstance(finalize_task_1, task_lib.FinalizePipelineTask)

    # Mark pipeline as failed.
    with self._mlmd_connection as m:
      pipeline_state = pstate.PipelineState.load(
          m, task_lib.PipelineUid.from_pipeline(self._pipeline)
      )
      with pipeline_state:
        pipeline_state.execution.last_known_state = (
            metadata_store_pb2.Execution.FAILED
        )
      pipeline_id = pipeline_state.pipeline_uid.pipeline_id
      pipeline_run_id = pipeline_state.pipeline_run_id

    # Pipeline revive should start the failed nodes: start_b and worker.
    with pipeline_ops.revive_pipeline_run(
        m, pipeline_id, pipeline_run_id
    ) as revive_pipeline_state:
      for node in [self.start_b, self.worker]:
        node_uid = task_lib.NodeUid.from_node(self._pipeline, node)
        self.assertEqual(
            revive_pipeline_state.get_node_state(node_uid).state,
            pstate.NodeState.STARTED,
        )

    # Because the pipeline has been revived, the previous failed executions
    # should not prevent re-execution of start_b and worker.
    self._run_next(False, expect_nodes=[self.start_b])
    self._run_next(False, expect_nodes=[self.worker])
    [finalize_task_2] = self._generate(False, True)
    self.assertIsInstance(finalize_task_2, task_lib.FinalizePipelineTask)


if __name__ == '__main__':
  tf.test.main()
