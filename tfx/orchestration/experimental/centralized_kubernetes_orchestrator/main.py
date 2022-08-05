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
"""Server for orchestration.

Registers the OrchestratorServicer, starts the server and listens for any events
through network connections.
"""

from concurrent import futures

from absl import app
from absl import flags
import grpc
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service import kubernetes_orchestrator_service
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service.proto import service_pb2_grpc

# Flags to use in the command line to specifiy the port and the number of
# threads. Commands can be changed later.
FLAGS = flags.FLAGS
flags.DEFINE_integer('port', 10000, 'port to listen on')


def _start_grpc_server() -> grpc.Server:
  """Starts GRPC server."""
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  servicer = kubernetes_orchestrator_service.KubernetesOrchestratorServicer()
  service_pb2_grpc.add_KubernetesOrchestratorServicer_to_server(
      servicer, server)
  server_creds = grpc.local_server_credentials()
  server.add_secure_port(f'[::]:{FLAGS.port}', server_creds)
  server.start()
  server.wait_for_termination()
  return server


def main(unused_argv):
  # unused_server will be used later.
  unused_server = _start_grpc_server()


if __name__ == '__main__':
  app.run(main)
