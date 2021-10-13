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
import uuid

from absl.testing import parameterized
from absl.testing.absltest import mock
import tensorflow as tf
from tfx.dsl.compiler import constants as compiler_constants
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import constants
from tfx.orchestration.experimental.core import mlmd_state
from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import service_jobs
from tfx.orchestration.experimental.core import sync_pipeline_task_gen as sptg
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_gen_utils
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.experimental.core.testing import test_sync_pipeline
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.orchestration.portable.mlmd import execution_lib
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
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self._mlmd_connection = metadata.Metadata(
        connection_config=connection_config)

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
      return service_jobs.ServiceStatus.SUCCESS

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _default_ensure_node_services)

  def _make_pipeline(self, pipeline_root, pipeline_run_id):
    pipeline = test_sync_pipeline.create_pipeline()
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
        self._mlmd_connection,
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
      self.assertTrue(task_lib.is_exec_node_task(task))
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
        expected_exec_nodes=expected_exec_nodes,
        ignore_update_node_state_tasks=ignore_update_node_state_tasks,
        fail_fast=fail_fast)

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
    self.assertTrue(task_lib.is_finalize_pipeline_task(finalize_task))
    self.assertEqual(status_lib.Code.OK, finalize_task.status.code)
    if use_task_queue:
      self.assertTrue(self._task_queue.is_empty())

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
    # Check the expected terminal nodes.
    layers = sptg._topsorted_layers(self._pipeline)
    self.assertEqual(
        {
            self._example_validator.node_info.id,
            self._chore_b.node_info.id,
            # evaluator execution will be skipped as it is run conditionally and
            # the condition always evaluates to False in the current test.
            self._evaluator.node_info.id,
        },
        sptg._terminal_node_ids(layers))

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
    self.assertTrue(task_lib.is_finalize_pipeline_task(finalize_task))
    self.assertEqual(status_lib.Code.OK, finalize_task.status.code)

  def test_service_job_running(self):
    """Tests task generation when example-gen service job is still running."""

    def _ensure_node_services(unused_pipeline_state, node_id):
      self.assertEqual('my_example_gen', node_id)
      return service_jobs.ServiceStatus.RUNNING

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _ensure_node_services)
    [task] = self._generate_and_test(
        True,
        num_initial_executions=0,
        num_tasks_generated=1,
        num_new_executions=0,
        num_active_executions=0)
    self.assertTrue(task_lib.is_update_node_state_task(task))
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
    self.assertTrue(
        task_lib.is_update_node_state_task(eg_update_node_state_task))
    self.assertEqual('my_example_gen',
                     eg_update_node_state_task.node_uid.node_id)
    self.assertEqual(pstate.NodeState.COMPLETE, eg_update_node_state_task.state)
    self.assertTrue(
        task_lib.is_update_node_state_task(sg_update_node_state_task))
    self.assertEqual('my_statistics_gen',
                     sg_update_node_state_task.node_uid.node_id)
    self.assertEqual(pstate.NodeState.RUNNING, sg_update_node_state_task.state)
    self.assertTrue(task_lib.is_exec_node_task(sg_exec_node_task))

  @parameterized.parameters(False, True)
  def test_service_job_failed(self, fail_fast):
    """Tests task generation when example-gen service job fails."""

    def _ensure_node_services(unused_pipeline_state, node_id):
      self.assertEqual('my_example_gen', node_id)
      return service_jobs.ServiceStatus.FAILED

    self._mock_service_job_manager.ensure_node_services.side_effect = (
        _ensure_node_services)
    [update_node_state_task, finalize_task] = self._generate_and_test(
        True,
        num_initial_executions=0,
        num_tasks_generated=2,
        num_new_executions=0,
        num_active_executions=0,
        fail_fast=fail_fast)
    self.assertTrue(task_lib.is_update_node_state_task(update_node_state_task))
    self.assertEqual('my_example_gen', update_node_state_task.node_uid.node_id)
    self.assertEqual(pstate.NodeState.FAILED, update_node_state_task.state)
    self.assertTrue(task_lib.is_finalize_pipeline_task(finalize_task))
    self.assertEqual(status_lib.Code.ABORTED, finalize_task.status.code)
    self.assertRegexMatch(finalize_task.status.message, ['service job failed'])

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
    self.assertTrue(
        task_lib.is_update_node_state_task(stats_gen_update_node_state_task))
    self.assertEqual('my_statistics_gen',
                     stats_gen_update_node_state_task.node_uid.node_id)
    self.assertEqual(pstate.NodeState.COMPLETE,
                     stats_gen_update_node_state_task.state)
    self.assertTrue(
        task_lib.is_update_node_state_task(schema_gen_update_node_state_task))
    self.assertEqual('my_schema_gen',
                     schema_gen_update_node_state_task.node_uid.node_id)
    self.assertEqual(pstate.NodeState.RUNNING,
                     schema_gen_update_node_state_task.state)
    self.assertTrue(task_lib.is_exec_node_task(schema_gen_exec_node_task))

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
        task_lib.NodeUid.from_pipeline_node(self._pipeline, self._stats_gen),
        stats_gen_task.node_uid)
    with self._mlmd_connection as m:
      with mlmd_state.mlmd_execution_atomic_op(
          m, stats_gen_task.execution_id) as stats_gen_exec:
        # Fail stats-gen execution.
        stats_gen_exec.last_known_state = metadata_store_pb2.Execution.FAILED
        data_types_utils.set_metadata_value(
            stats_gen_exec.custom_properties[constants.EXECUTION_ERROR_MSG_KEY],
            'foobar error')

    # Test generation of FinalizePipelineTask.
    [update_node_state_task, finalize_task] = self._generate_and_test(
        True,
        num_initial_executions=2,
        num_tasks_generated=2,
        num_new_executions=0,
        num_active_executions=0,
        fail_fast=fail_fast)
    self.assertTrue(task_lib.is_update_node_state_task(update_node_state_task))
    self.assertEqual('my_statistics_gen',
                     update_node_state_task.node_uid.node_id)
    self.assertEqual(pstate.NodeState.FAILED, update_node_state_task.state)
    self.assertRegexMatch(update_node_state_task.status.message,
                          ['foobar error'])
    self.assertTrue(task_lib.is_finalize_pipeline_task(finalize_task))
    self.assertEqual(status_lib.Code.ABORTED, finalize_task.status.code)
    self.assertRegexMatch(finalize_task.status.message, ['foobar error'])

  def test_cached_execution(self):
    """Tests that cached execution is used if one is available."""

    # Fake ExampleGen run.
    example_gen_exec = test_utils.fake_example_gen_run(self._mlmd_connection,
                                                       self._example_gen, 1, 1)

    # Invoking generator should produce an ExecNodeTask for StatsGen.
    [stats_gen_task] = self._generate_and_test(
        False,
        num_initial_executions=1,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        ignore_update_node_state_tasks=True)
    self.assertEqual('my_statistics_gen', stats_gen_task.node_uid.node_id)

    # Finish StatsGen execution.
    test_utils.fake_execute_node(self._mlmd_connection, stats_gen_task)

    # Finish pipeline execution.
    with self._mlmd_connection as m:
      pipeline_state = test_utils.get_or_create_pipeline_state(
          m, self._pipeline)
      with pipeline_state:
        pipeline_state.set_pipeline_execution_state(
            metadata_store_pb2.Execution.COMPLETE)

    # Prepare another pipeline with a new pipeline_run_id.
    pipeline_run_id = str(uuid.uuid4())
    new_pipeline = self._make_pipeline(self._pipeline_root, pipeline_run_id)

    with self._mlmd_connection as m:
      contexts = m.store.get_contexts_by_execution(example_gen_exec.id)
      # We use node context as cache context for ease of testing.
      cache_context = [
          c for c in contexts if c.name == 'my_pipeline.my_example_gen'
      ][0]
    # Fake example_gen cached execution.
    test_utils.fake_cached_execution(
        self._mlmd_connection, cache_context,
        test_utils.get_node(new_pipeline, 'my_example_gen'))

    stats_gen = test_utils.get_node(new_pipeline, 'my_statistics_gen')

    # Invoking generator for the new pipeline should result in:
    # 1. An UpdateNodeStateTask each for ExampleGen, StatsGen and SchemaGen
    #    signaling execution completion.
    # 2. StatsGen execution succeeds with state "CACHED" but no ExecNodeTask
    #    generated.
    # 2. An ExecNodeTask is generated for SchemaGen (component downstream of
    #    StatsGen) with an active execution in MLMD.
    tasks = self._generate_and_test(
        False,
        pipeline=new_pipeline,
        num_initial_executions=3,
        num_tasks_generated=4,
        num_new_executions=2,
        num_active_executions=1)
    self.assertLen(tasks, 4)
    update_node_state_tasks = [
        t for t in tasks if task_lib.is_update_node_state_task(t)
    ]
    self.assertLen(update_node_state_tasks, 3)
    update_node_state_task_by_node_id = {
        t.node_uid.node_id: t for t in update_node_state_tasks
    }
    self.assertCountEqual(
        ['my_example_gen', 'my_statistics_gen', 'my_schema_gen'],
        list(update_node_state_task_by_node_id.keys()))
    self.assertEqual(pstate.NodeState.COMPLETE,
                     update_node_state_task_by_node_id['my_example_gen'].state)
    self.assertEqual(
        pstate.NodeState.COMPLETE,
        update_node_state_task_by_node_id['my_statistics_gen'].state)
    self.assertEqual(pstate.NodeState.RUNNING,
                     update_node_state_task_by_node_id['my_schema_gen'].state)

    [schema_gen_task] = [t for t in tasks if task_lib.is_exec_node_task(t)]
    self.assertEqual('my_schema_gen', schema_gen_task.node_uid.node_id)

    # Check that StatsGen execution is successful in state "CACHED".
    with self._mlmd_connection as m:
      executions = task_gen_utils.get_executions(m, stats_gen)
      self.assertLen(executions, 1)
      execution = executions[0]
      self.assertTrue(execution_lib.is_execution_successful(execution))
      self.assertEqual(metadata_store_pb2.Execution.CACHED,
                       execution.last_known_state)

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
              task_lib.NodeUid.from_pipeline_node(
                  self._pipeline, self._stats_gen)) as node_state:
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
    node_uid = task_lib.NodeUid.from_pipeline_node(self._pipeline,
                                                   self._stats_gen)
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

    # Change state of node to STARTING.
    with self._mlmd_connection as m:
      pipeline_state = test_utils.get_or_create_pipeline_state(
          m, self._pipeline)
      with pipeline_state:
        with pipeline_state.node_state_update_context(node_uid) as node_state:
          node_state.update(pstate.NodeState.STARTING)

    # New execution should be created for any previously canceled node when the
    # node state is STARTING.
    [update_node_state_task, stats_gen_task] = self._generate_and_test(
        False,
        num_initial_executions=2,
        num_tasks_generated=2,
        num_new_executions=1,
        num_active_executions=1)
    self.assertTrue(task_lib.is_update_node_state_task(update_node_state_task))
    self.assertEqual(node_uid, update_node_state_task.node_uid)
    self.assertEqual(pstate.NodeState.RUNNING, update_node_state_task.state)
    self.assertEqual(node_uid, stats_gen_task.node_uid)

  @parameterized.parameters(False, True)
  def test_conditional_execution(self, evaluate):
    """Tests conditionals in the pipeline.

    Args:
      evaluate: Whether to run the conditional evaluator.
    """
    # Check the expected terminal nodes.
    layers = sptg._topsorted_layers(self._pipeline)
    self.assertEqual(
        {
            self._example_validator.node_info.id,
            self._chore_b.node_info.id,
            self._evaluator.node_info.id,
        }, sptg._terminal_node_ids(layers))

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
        t for t in tasks if task_lib.is_update_node_state_task(t) and
        t.node_uid.node_id == 'my_evaluator'
    ]
    self.assertEqual(
        pstate.NodeState.RUNNING if evaluate else pstate.NodeState.SKIPPED,
        evaluator_update_node_state_task.state)

    exec_node_tasks = [t for t in tasks if task_lib.is_exec_node_task(t)]
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
    self.assertTrue(task_lib.is_finalize_pipeline_task(finalize_task))

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
            ev_exec.custom_properties[constants.EXECUTION_ERROR_MSG_KEY],
            'example-validator error')

    if fail_fast:
      # Pipeline run should immediately fail because example-validator failed.
      [finalize_task] = self._generate(False, True, fail_fast=fail_fast)
      self.assertTrue(task_lib.is_finalize_pipeline_task(finalize_task))
      self.assertEqual(status_lib.Code.ABORTED, finalize_task.status.code)
    else:
      # Trainer and downstream nodes can execute as transform has finished.
      # example-validator failure does not impact them as it is not upstream.
      # Pipeline run will still fail but when no more progress can be made.
      self._run_next(False, expect_nodes=[self._trainer], fail_fast=fail_fast)
      self._run_next(False, expect_nodes=[self._chore_a], fail_fast=fail_fast)
      self._run_next(False, expect_nodes=[self._chore_b], fail_fast=fail_fast)
      [finalize_task] = self._generate(False, True, fail_fast=fail_fast)
      self.assertTrue(task_lib.is_finalize_pipeline_task(finalize_task))
      self.assertEqual(status_lib.Code.ABORTED, finalize_task.status.code)


if __name__ == '__main__':
  tf.test.main()
