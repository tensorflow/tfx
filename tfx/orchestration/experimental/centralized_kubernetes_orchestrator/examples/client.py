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
"""Client for orchestrator.

A simple client to communicate with the orchestrator server.
"""

from absl import app
from absl import flags
import grpc
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service.proto import service_pb2
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service.proto import service_pb2_grpc

# Flags to use in the command line to specifiy the port and the msg.
# Commands can be changed later.
FLAGS = flags.FLAGS
flags.DEFINE_string('server', 'dns:///[::1]:10000', 'server address')
flags.DEFINE_string('msg', 'Hello World', 'default message')


def _echo_message(stub, request):
  """Echoes user's message."""
  try:
    response = stub.Echo(request)
    print(response)
    return 0
  except grpc.RpcError as rpc_error:
    print(rpc_error)
    return -1


def main(unused_argv):
  channel_creds = grpc.local_channel_credentials()
  with grpc.secure_channel(FLAGS.server, channel_creds) as channel:
    grpc.channel_ready_future(channel).result()
    stub = service_pb2_grpc.KubernetesOrchestratorStub(channel)
    request = service_pb2.EchoRequest(msg=FLAGS.msg)
    return _echo_message(stub, request)


if __name__ == '__main__':
  app.run(main)
