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

import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import async_pipeline_task_gen as asptg
from tfx.orchestration.experimental.core import test_utils as otu
from tfx.orchestration.portable import test_utils as tu
from tfx.proto.orchestration import pipeline_pb2

from ml_metadata.proto import metadata_store_pb2


class AsyncPipelineTaskGeneratorTest(tu.TfxTest):

  def setUp(self):
    super(AsyncPipelineTaskGeneratorTest, self).setUp()
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
    self._pipeline = pipeline
    self._pipeline_info = pipeline.pipeline_info
    self._pipeline_runtime_spec = pipeline.runtime_spec
    self._pipeline_runtime_spec.pipeline_root.field_value.string_value = (
        pipeline_root)

    # Extracts components.
    self._example_gen = pipeline.nodes[0].pipeline_node
    self._transform = pipeline.nodes[1].pipeline_node
    self._trainer = pipeline.nodes[2].pipeline_node

  def _verify_node_execution_task(self, node, execution, task):
    self.assertEqual(node.node_info.id, task.exec_task.node_id)
    self.assertEqual(self._pipeline.pipeline_info.id,
                     task.exec_task.pipeline_id)
    self.assertEmpty(task.exec_task.pipeline_run_id)
    self.assertEqual(execution.id, task.exec_task.execution_id)

  def _generate_and_test(self, num_initial_executions, num_tasks_generated,
                         num_new_executions, num_active_executions):
    """Generates tasks and tests the effects."""
    with self._mlmd_connection as m:
      executions = m.store.get_executions()
    self.assertLen(
        executions, num_initial_executions,
        'Expected {} execution(s) in MLMD.'.format(num_initial_executions))
    task_gen = asptg.AsyncPipelineTaskGenerator(self._mlmd_connection,
                                                self._pipeline)
    tasks = task_gen.generate()
    self.assertLen(
        tasks, num_tasks_generated,
        'Expected {} task(s) to be generated.'.format(num_tasks_generated))
    with self._mlmd_connection as m:
      executions = m.store.get_executions()
    num_total_executions = num_initial_executions + num_new_executions
    self.assertLen(
        executions, num_total_executions,
        'Expected {} execution(s) in MLMD.'.format(num_total_executions))
    active_executions = [
        e for e in executions
        if e.last_known_state == metadata_store_pb2.Execution.RUNNING
    ]
    self.assertLen(
        active_executions, num_active_executions,
        'Expected {} active execution(s) in MLMD.'.format(
            num_active_executions))
    return tasks, active_executions

  def test_no_tasks_generated_when_new(self):
    task_gen = asptg.AsyncPipelineTaskGenerator(self._mlmd_connection,
                                                self._pipeline)
    tasks = task_gen.generate()
    self.assertEmpty(tasks, 'Expected no task generation when no inputs.')
    with self._mlmd_connection as m:
      self.assertEmpty(
          m.store.get_executions(),
          'There must not be any registered executions since no tasks were '
          'geerated.')

  def test_task_generation(self):
    # Simulate that ExampleGen has already completed successfully.
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1, 1)

    # Before generation, there's 1 execution in MLMD.
    with self._mlmd_connection as m:
      executions = m.store.get_executions()
    self.assertLen(executions, 1)

    # Generate once.
    with self.subTest(generate=1):
      tasks, active_executions = self._generate_and_test(
          num_initial_executions=1,
          num_tasks_generated=1,
          num_new_executions=1,
          num_active_executions=1)
      self._verify_node_execution_task(self._transform, active_executions[0],
                                       tasks[0])

    # No new effects if generate called again.
    with self.subTest(generate=2):
      tasks, active_executions = self._generate_and_test(
          num_initial_executions=2,
          num_tasks_generated=1,
          num_new_executions=0,
          num_active_executions=1)
      self._verify_node_execution_task(self._transform, active_executions[0],
                                       tasks[0])

    # Mark transform execution complete.
    otu.fake_transform_output(self._mlmd_connection, self._transform,
                              active_executions[0])

    # Trainer execution task should be generated next.
    with self.subTest(generate=3):
      tasks, active_executions = self._generate_and_test(
          num_initial_executions=2,
          num_tasks_generated=1,
          num_new_executions=1,
          num_active_executions=1)
      self._verify_node_execution_task(self._trainer, active_executions[0],
                                       tasks[0])

    # Mark the trainer execution complete.
    otu.fake_trainer_output(self._mlmd_connection, self._trainer,
                            active_executions[0])

    # No more tasks should be generated as there are no new inputs.
    with self.subTest(generate=4):
      self._generate_and_test(
          num_initial_executions=3,
          num_tasks_generated=0,
          num_new_executions=0,
          num_active_executions=0)

    # Fake another ExampleGen run.
    otu.fake_example_gen_run(self._mlmd_connection, self._example_gen, 1, 1)

    # Both transform and trainer tasks should be generated as they both find
    # new inputs.
    with self.subTest(generate=4):
      tasks, active_executions = self._generate_and_test(
          num_initial_executions=4,
          num_tasks_generated=2,
          num_new_executions=2,
          num_active_executions=2)
      self._verify_node_execution_task(self._transform, active_executions[0],
                                       tasks[0])
      self._verify_node_execution_task(self._trainer, active_executions[1],
                                       tasks[1])

    # Re-generation will produce the same tasks again.
    with self.subTest(generate=5):
      tasks, active_executions = self._generate_and_test(
          num_initial_executions=6,
          num_tasks_generated=2,
          num_new_executions=0,
          num_active_executions=2)
      self._verify_node_execution_task(self._transform, active_executions[0],
                                       tasks[0])
      self._verify_node_execution_task(self._trainer, active_executions[1],
                                       tasks[1])

    # Mark transform execution complete.
    otu.fake_transform_output(self._mlmd_connection, self._transform,
                              active_executions[0])

    # Mark the trainer execution complete.
    otu.fake_trainer_output(self._mlmd_connection, self._trainer,
                            active_executions[1])

    # Trainer should be triggered again due to transform producing new output.
    with self.subTest(generate=6):
      tasks, active_executions = self._generate_and_test(
          num_initial_executions=6,
          num_tasks_generated=1,
          num_new_executions=1,
          num_active_executions=1)
      self._verify_node_execution_task(self._trainer, active_executions[0],
                                       tasks[0])

    # Finally, no new tasks once trainer completes.
    otu.fake_trainer_output(self._mlmd_connection, self._trainer,
                            active_executions[0])
    with self.subTest(generate=7):
      self._generate_and_test(
          num_initial_executions=7,
          num_tasks_generated=0,
          num_new_executions=0,
          num_active_executions=0)


if __name__ == '__main__':
  tf.test.main()
