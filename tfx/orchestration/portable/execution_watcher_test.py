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
"""Tests for tfx.orchestration.portable.execution_watcher."""

import os

import grpc
import portpicker
import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import execution_watcher
from tfx.proto.orchestration import execution_watcher_pb2
from tfx.utils import test_case_utils

from ml_metadata.proto import metadata_store_pb2


class ExecutionWatcherTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    # Set up MLMD connection.
    pipeline_root = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self.id())
    metadata_path = os.path.join(pipeline_root, 'metadata', 'metadata.db')
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    self._mlmd_connection = metadata.Metadata(
        connection_config=connection_config)
    with self._mlmd_connection as m:
      self._execution = execution_publish_utils.register_execution(
          metadata_handler=m,
          execution_type=metadata_store_pb2.ExecutionType(
              name='test_execution_type'),
          contexts=[],
          input_artifacts=[])
    # Set up gRPC stub.
    port = portpicker.pick_unused_port()
    self.sidecar = execution_watcher.ExecutionWatcher(
        port,
        mlmd_connection=self._mlmd_connection,
        execution=self._execution,
        creds=grpc.local_server_credentials())
    self.sidecar.start()
    self.stub = execution_watcher.generate_service_stub(
        self.sidecar.address, grpc.local_channel_credentials())

  def tearDown(self):
    super().tearDown()
    self.sidecar.stop()

  def testExecutionWatcher_LocalWithEmptyUpdates(self):
    req = execution_watcher_pb2.UpdateExecutionInfoRequest()
    with self.assertRaisesRegex(grpc.RpcError,
                                'not tracked') as exception_context:
      self.stub.UpdateExecutionInfo(req)
    self.assertIs(grpc.StatusCode.NOT_FOUND, exception_context.exception.code())

  def testExecutionWatcher_Local(self):
    req = execution_watcher_pb2.UpdateExecutionInfoRequest()
    value = metadata_store_pb2.Value()
    value.string_value = 'string_value'
    req.execution_id = self._execution.id
    req.updates['test_key'].CopyFrom(value)
    res = self.stub.UpdateExecutionInfo(req)
    self.assertEqual(execution_watcher_pb2.UpdateExecutionInfoResponse(), res)
    with self._mlmd_connection as m:
      executions = m.store.get_executions_by_id([self._execution.id])
    self.assertEqual(len(executions), 1)
    self.assertProtoPartiallyEquals(
        """
      id: 1
      last_known_state: RUNNING
      custom_properties {
        key: "test_key"
        value {
          string_value: "string_value"
        }
      }
      """,
        executions[0],
        ignored_fields=[
            'type_id', 'create_time_since_epoch', 'last_update_time_since_epoch'
        ])


if __name__ == '__main__':
  tf.test.main()
