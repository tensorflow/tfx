# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.experimental.core.task_schedulers.resolver_task_scheduler."""

import os
import uuid

import tensorflow as tf
from tfx import types
from tfx.dsl.compiler import constants
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import post_execution_utils
from tfx.orchestration.experimental.core import sync_pipeline_task_gen as sptg
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import task_scheduler
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.experimental.core.task_schedulers import resolver_task_scheduler
from tfx.orchestration.experimental.core.testing import test_pipeline_with_resolver
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.orchestration.portable.mlmd import context_lib
from tfx.utils import status as status_lib


class ResolverTaskSchedulerTest(test_utils.TfxTest):

  def setUp(self):
    super().setUp()

    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())

    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self._mlmd_connection = metadata.Metadata(
        connection_config=connection_config)

    pipeline = self._make_pipeline(pipeline_root, str(uuid.uuid4()))
    self._pipeline = pipeline
    self._trainer = self._pipeline.nodes[0].pipeline_node
    self._resolver_node = self._pipeline.nodes[1].pipeline_node
    self._consumer_node = self._pipeline.nodes[2].pipeline_node

  def _make_pipeline(self, pipeline_root, pipeline_run_id):
    pipeline = test_pipeline_with_resolver.create_pipeline()
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline, {
            constants.PIPELINE_ROOT_PARAMETER_NAME: pipeline_root,
            constants.PIPELINE_RUN_ID_PARAMETER_NAME: pipeline_run_id,
        })
    return pipeline

  def test_resolver_task_scheduler(self):
    with self._mlmd_connection as m:
      # Publishes two models which will be consumed by downstream resolver.
      output_model_1 = types.Artifact(
          self._trainer.outputs.outputs['model'].artifact_spec.type)
      output_model_1.uri = 'my_model_uri_1'

      output_model_2 = types.Artifact(
          self._trainer.outputs.outputs['model'].artifact_spec.type)
      output_model_2.uri = 'my_model_uri_2'

      contexts = context_lib.prepare_contexts(m, self._trainer.contexts)
      execution = execution_publish_utils.register_execution(
          m, self._trainer.node_info.type, contexts)
      execution_publish_utils.publish_succeeded_execution(
          m, execution.id, contexts, {
              'model': [output_model_1, output_model_2],
          })

    task_queue = tq.TaskQueue()

    # Verify that resolver task is generated.
    [resolver_task] = test_utils.run_generator_and_test(
        test_case=self,
        mlmd_connection=self._mlmd_connection,
        generator_class=sptg.SyncPipelineTaskGenerator,
        pipeline=self._pipeline,
        task_queue=task_queue,
        use_task_queue=False,
        service_job_manager=None,
        num_initial_executions=1,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._resolver_node],
        ignore_update_node_state_tasks=True)

    with self._mlmd_connection as m:
      # Run resolver task scheduler and publish results.
      ts_result = resolver_task_scheduler.ResolverTaskScheduler(
          mlmd_handle=m, pipeline=self._pipeline,
          task=resolver_task).schedule()
      self.assertEqual(status_lib.Code.OK, ts_result.status.code)
      self.assertIsInstance(ts_result.output, task_scheduler.ResolverNodeOutput)
      self.assertCountEqual(['resolved_model'],
                            ts_result.output.resolved_input_artifacts.keys())
      models = ts_result.output.resolved_input_artifacts['resolved_model']
      self.assertLen(models, 1)
      self.assertEqual('my_model_uri_2', models[0].mlmd_artifact.uri)
      post_execution_utils.publish_execution_results_for_task(
          m, resolver_task, ts_result)

    # Verify resolver node output is input to the downstream consumer node.
    [consumer_task] = test_utils.run_generator_and_test(
        test_case=self,
        mlmd_connection=self._mlmd_connection,
        generator_class=sptg.SyncPipelineTaskGenerator,
        pipeline=self._pipeline,
        task_queue=task_queue,
        use_task_queue=False,
        service_job_manager=None,
        num_initial_executions=2,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._consumer_node],
        ignore_update_node_state_tasks=True)
    self.assertCountEqual(['resolved_model'],
                          consumer_task.input_artifacts.keys())
    input_models = consumer_task.input_artifacts['resolved_model']
    self.assertLen(input_models, 1)
    self.assertEqual('my_model_uri_2', input_models[0].mlmd_artifact.uri)


if __name__ == '__main__':
  tf.test.main()
