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
"""Tests for tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service.kubernetes_orchestrator_service."""

import grpc
from grpc.framework.foundation import logging_pool
import portpicker
import tensorflow as tf
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service import kubernetes_orchestrator_service
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service.proto import service_pb2
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service.proto import service_pb2_grpc


class KubernetesOrchestratorServiceTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    port = portpicker.pick_unused_port()

    server_pool = logging_pool.pool(max_workers=25)
    cls._server = grpc.server(server_pool)
    cls._server.add_secure_port(f'[::]:{port}'.format(port),
                                grpc.local_server_credentials())
    servicer = kubernetes_orchestrator_service.KubernetesOrchestratorServicer()
    service_pb2_grpc.add_KubernetesOrchestratorServicer_to_server(
        servicer, cls._server)
    cls._server.start()
    cls._channel = grpc.secure_channel(f'localhost:{port}',
                                       grpc.local_channel_credentials())
    cls._stub = service_pb2_grpc.KubernetesOrchestratorStub(cls._channel)

  @classmethod
  def tearDownClass(cls):
    cls._channel.close()
    cls._server.stop(None)
    super().tearDownClass()

  def test_echo(self):
    msg = 'This is a test message.'
    request = service_pb2.EchoRequest(msg=msg)
    response = self._stub.Echo(request)

    self.assertEqual(response.msg, msg)


if __name__ == '__main__':
  tf.test.main()
