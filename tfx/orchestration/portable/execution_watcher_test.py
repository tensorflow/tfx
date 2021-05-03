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

import grpc
import portpicker
import tensorflow as tf
from tfx.orchestration.portable import execution_watcher
from tfx.proto.orchestration import execution_watcher_pb2
from tfx.proto.orchestration import execution_watcher_pb2_grpc
from tfx.utils import test_case_utils


class ExecutionWatcherTest(test_case_utils.TfxTest):

  def testExecutionWatcher_LocalWithEmptyRequest(self):
    port = portpicker.pick_unused_port()
    sidecar = execution_watcher.ExecutionWatcher(
        port, creds=grpc.local_server_credentials())
    sidecar.start()
    creds = grpc.local_channel_credentials()
    channel = grpc.secure_channel(sidecar.local_address, creds)
    stub = execution_watcher_pb2_grpc.ExecutionWatcherServiceStub(channel)
    req = execution_watcher_pb2.UpdateExecutionInfoRequest()
    res = stub.UpdateExecutionInfo(req)
    sidecar.stop()
    self.assertEqual(execution_watcher_pb2.UpdateExecutionInfoResponse(), res)


if __name__ == '__main__':
  tf.test.main()
