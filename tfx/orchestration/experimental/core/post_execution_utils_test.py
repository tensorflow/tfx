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

from absl.testing.absltest import mock
import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.orchestration import data_types_utils
from tfx.orchestration import metadata
from tfx.orchestration.experimental.core import post_execution_utils
from tfx.orchestration.portable import data_types
from tfx.orchestration.portable import execution_publish_utils
from tfx.proto.orchestration import execution_invocation_pb2
from tfx.proto.orchestration import execution_result_pb2
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils as tu

from ml_metadata import proto


class PostExecutionUtilsTest(tu.TfxTest):

  def setUp(self):
    super().setUp()
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
            {'foo_arg': 'haha'}),
        output_dict=data_types_utils.build_artifact_struct_dict(
            {'example': [self.example_artifact]}),
        execution_id=execution.id)
    return data_types.ExecutionInfo.from_proto(execution_invocation)

  def test_publish_execution_results_failed_execution(self):
    execution_info = self._prepare_execution_info()

    executor_output = execution_result_pb2.ExecutorOutput()
    # Code as defined in google.rpc.Code - INVALID_ARGUMENT
    executor_output.execution_result.code = 3
    executor_output.execution_result.result_message = 'failed execution'

    post_execution_utils.publish_execution_results(
        self.mlmd_handle, executor_output, execution_info, contexts=[])

    self.assertFalse(fileio.exists(self.example_artifact.uri))
    [execution] = self.mlmd_handle.store.get_executions()

    self.assertEqual(execution.last_known_state, proto.Execution.FAILED)

  @mock.patch.object(execution_publish_utils, 'publish_succeeded_execution')
  def test_publish_execution_results_succeeded_execution(self, mock_publish):
    execution_info = self._prepare_execution_info()

    executor_output = execution_result_pb2.ExecutorOutput()
    executor_output.execution_result.code = 0

    post_execution_utils.publish_execution_results(
        self.mlmd_handle, executor_output, execution_info, contexts=[])

    self.assertTrue(fileio.exists(self.example_artifact.uri))
    [execution] = self.mlmd_handle.store.get_executions()
    mock_publish.assert_called_once_with(
        self.mlmd_handle,
        execution_id=execution.id,
        contexts=[],
        output_artifacts=execution_info.output_dict,
        executor_output=executor_output)


if __name__ == '__main__':
  tf.test.main()
