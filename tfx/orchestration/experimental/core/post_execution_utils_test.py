# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.experimental.core.post_execution_utils."""
import os

from absl.testing import parameterized
from absl.testing.absltest import mock
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import component_generated_alert_pb2
from tfx.orchestration.experimental.core import constants
from tfx.orchestration.experimental.core import event_observer
from tfx.orchestration.experimental.core import post_execution_utils
from tfx.orchestration.experimental.core import task as task_lib
from tfx.orchestration.experimental.core import task_scheduler as ts
from tfx.orchestration.experimental.core import test_utils
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable import execution_publish_utils
from tfx.proto.orchestration import execution_invocation_pb2
from tfx.proto.orchestration import execution_result_pb2
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts
from tfx.utils import status as status_lib
from tfx.utils import test_case_utils as tu

from ml_metadata import proto


class PostExecutionUtilsTest(tu.TfxTest, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.stateful_working_dir = self.create_tempdir().full_path
    metadata_path = os.path.join(self.tmp_dir, 'metadata', 'metadata.db')
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self.mlmd_handle = metadata.Metadata(connection_config=connection_config)
    self.mlmd_handle.__enter__()

    self.execution_type = proto.ExecutionType(name='my_ex_type')

    self.example_artifact = standard_artifacts.Examples()
    example_artifact_uri = os.path.join(self.tmp_dir, 'ExampleOutput')
    fileio.makedirs(example_artifact_uri)
    self.example_artifact.uri = example_artifact_uri

  def tearDown(self):
    self.mlmd_handle.__exit__(None, None, None)
    super().tearDown()

  def _prepare_execution_info(self):
    execution_publish_utils.register_execution(
        self.mlmd_handle,
        self.execution_type,
        contexts=[],
        exec_properties={'foo_arg': 'haha'})
    [execution] = self.mlmd_handle.store.get_executions()
    self.assertEqual(execution.last_known_state, proto.Execution.RUNNING)

    execution_invocation = execution_invocation_pb2.ExecutionInvocation(
        execution_properties=data_types_utils.build_metadata_value_dict(
            {'foo_arg': 'haha'}
        ),
        output_dict=data_types_utils.build_artifact_struct_dict(
            {'example': [self.example_artifact]}
        ),
        execution_id=execution.id,
        stateful_working_dir=self.stateful_working_dir,
    )
    return data_types.ExecutionInfo.from_proto(execution_invocation)

  @parameterized.named_parameters(
      dict(
          testcase_name='canceled-execution',
          code=status_lib.Code.CANCELLED,
          expected_execution_state=proto.Execution.CANCELED),
      dict(
          testcase_name='failed-execution',
          code=status_lib.Code.INVALID_ARGUMENT,
          expected_execution_state=proto.Execution.FAILED))
  def test_publish_execution_results_failed_execution(self, code,
                                                      expected_execution_state):
    execution_info = self._prepare_execution_info()

    executor_output = execution_result_pb2.ExecutorOutput()
    executor_output.execution_result.code = code
    executor_output.execution_result.result_message = 'failed execution'

    post_execution_utils.publish_execution_results(
        self.mlmd_handle, executor_output, execution_info, contexts=[])

    [execution] = self.mlmd_handle.store.get_executions()

    self.assertEqual(execution.last_known_state, expected_execution_state)
    self.assertTrue(fileio.exists(self.stateful_working_dir))

  @mock.patch.object(execution_publish_utils, 'publish_succeeded_execution')
  def test_publish_execution_results_succeeded_execution(self, mock_publish):
    execution_info = self._prepare_execution_info()

    executor_output = execution_result_pb2.ExecutorOutput()
    executor_output.execution_result.code = 0

    mock_publish.return_value = [None, None]

    post_execution_utils.publish_execution_results(
        self.mlmd_handle, executor_output, execution_info, contexts=[])

    [execution] = self.mlmd_handle.store.get_executions()
    mock_publish.assert_called_once_with(
        self.mlmd_handle,
        execution_id=execution.id,
        contexts=[],
        output_artifacts=execution_info.output_dict,
        executor_output=executor_output)
    self.assertFalse(fileio.exists(self.stateful_working_dir))

  @mock.patch.object(event_observer, 'notify')
  def test_publish_execution_results_for_task_with_alerts(self, mock_notify):
    _ = self._prepare_execution_info()

    executor_output = execution_result_pb2.ExecutorOutput()
    executor_output.execution_result.code = 0

    component_generated_alerts = (
        component_generated_alert_pb2.ComponentGeneratedAlertList()
    )
    component_generated_alerts.component_generated_alert_list.append(
        component_generated_alert_pb2.ComponentGeneratedAlertInfo(
            alert_name='test_alert',
            alert_body='test_alert_body',
        )
    )
    executor_output.execution_properties[
        constants.COMPONENT_GENERATED_ALERTS_KEY
    ].proto_value.Pack(component_generated_alerts)

    [execution] = self.mlmd_handle.store.get_executions()

    # Create test pipeline.
    deployment_config = pipeline_pb2.IntermediateDeploymentConfig()
    executor_spec = pipeline_pb2.ExecutorSpec.PythonClassExecutorSpec(
        class_path='trainer.TrainerExecutor')
    deployment_config.executor_specs['AlertGenerator'].Pack(
        executor_spec
    )
    pipeline = pipeline_pb2.Pipeline()
    pipeline.nodes.add().pipeline_node.node_info.id = 'AlertGenerator'
    pipeline.pipeline_info.id = 'test-pipeline'
    pipeline.deployment_config.Pack(deployment_config)

    node_uid = task_lib.NodeUid(
        pipeline_uid=task_lib.PipelineUid(
            pipeline_id=pipeline.pipeline_info.id
        ),
        node_id='AlertGenerator',
    )
    task = test_utils.create_exec_node_task(
        node_uid=node_uid,
        execution=execution,
        pipeline=pipeline,
    )
    result = ts.TaskSchedulerResult(
        status=status_lib.Status(
            code=status_lib.Code.OK,
            message='test TaskScheduler result'
        ),
        output=ts.ExecutorNodeOutput(executor_output=executor_output)
    )
    post_execution_utils.publish_execution_results_for_task(
        self.mlmd_handle, task, result
    )
    mock_notify.assert_called_once()


if __name__ == '__main__':
  tf.test.main()
