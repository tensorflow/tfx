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
"""Tests for tfx.orchestration.experimental.core.task_schedulers.importer_task_scheduler."""

import os
from unittest import mock
import uuid

import tensorflow as tf
from tfx.dsl.compiler import constants
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import post_execution_utils
from tfx.orchestration.experimental.core import sync_pipeline_task_gen as sptg
from tfx.orchestration.experimental.core import task_queue as tq
from tfx.orchestration.experimental.core import task_scheduler
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.experimental.core.task_schedulers import importer_task_scheduler
from tfx.orchestration.experimental.core.testing import test_pipeline_with_importer
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.utils import status as status_lib


class ImporterTaskSchedulerTest(test_utils.TfxTest):

  def setUp(self):
    super().setUp()

    self.addCleanup(mock.patch.stopall)
    # Set a constant version for artifact version tag.
    mock.patch('tfx.version.__version__', '0.123.4.dev').start()

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
    self._importer_node = self._pipeline.nodes[0].pipeline_node

    self._task_queue = tq.TaskQueue()
    [importer_task] = test_utils.run_generator_and_test(
        test_case=self,
        mlmd_connection=self._mlmd_connection,
        generator_class=sptg.SyncPipelineTaskGenerator,
        pipeline=self._pipeline,
        task_queue=self._task_queue,
        use_task_queue=True,
        service_job_manager=None,
        num_initial_executions=0,
        num_tasks_generated=1,
        num_new_executions=1,
        num_active_executions=1,
        expected_exec_nodes=[self._importer_node],
        ignore_update_node_state_tasks=True)
    self._importer_task = importer_task

  def _make_pipeline(self, pipeline_root, pipeline_run_id):
    pipeline = test_pipeline_with_importer.create_pipeline()
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline, {
            constants.PIPELINE_ROOT_PARAMETER_NAME: pipeline_root,
            constants.PIPELINE_RUN_ID_PARAMETER_NAME: pipeline_run_id,
        })
    return pipeline

  def test_importer_task_scheduler(self):
    with self._mlmd_connection as m:
      ts_result = importer_task_scheduler.ImporterTaskScheduler(
          mlmd_handle=m, pipeline=self._pipeline,
          task=self._importer_task).schedule()
      self.assertEqual(status_lib.Code.OK, ts_result.status.code)
      self.assertIsInstance(ts_result.output, task_scheduler.ImporterNodeOutput)
      post_execution_utils.publish_execution_results_for_task(
          m, self._importer_task, ts_result)
      [artifact] = m.store.get_artifacts_by_type('Schema')
      self.assertProtoPartiallyEquals(
          """
          uri: "my_url"
          custom_properties {
            key: "int_custom_property"
            value {
              int_value: 123
            }
          }
          custom_properties {
            key: "is_external"
            value {
              int_value: 1
            }
          }
          custom_properties {
            key: "str_custom_property"
            value {
              string_value: "abc"
            }
          }
          custom_properties {
            key: "tfx_version"
            value {
              string_value: "0.123.4.dev"
            }
          }
          state: LIVE""",
          artifact,
          ignored_fields=[
              'id', 'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch'
          ])

      [execution
      ] = m.store.get_executions_by_id([self._importer_task.execution_id])
      del execution.custom_properties['__execution_timestamp__']
      self.assertProtoPartiallyEquals(
          """
          last_known_state: COMPLETE
          custom_properties {
            key: "__execution_set_size__"
            value {
              int_value: 1
            }
          }
          custom_properties {
            key: "__external_execution_index__"
            value {
              int_value: 0
            }
          }
          custom_properties {
            key: "artifact_uri"
            value {
              string_value: "my_url"
            }
          }
          custom_properties {
            key: "output_key"
            value {
              string_value: "result"
            }
          }
          custom_properties {
            key: "reimport"
            value {
              int_value: 1
            }
          }
          """,
          execution,
          ignored_fields=[
              'id', 'type_id', 'create_time_since_epoch',
              'last_update_time_since_epoch', 'name'
          ])


if __name__ == '__main__':
  tf.test.main()
