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
"""Centralized Kubernetes Orchestrator Service.

Implementation of a servicer that will be used for Centralized Kubernetes
Orchestrator.
"""

from tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service.proto import service_pb2
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service.proto import service_pb2_grpc


class KubernetesOrchestratorServicer(
    service_pb2_grpc.KubernetesOrchestratorServicer):
  """A service interface for pipeline orchestration."""

  def __init__(self):
    pass

  def Echo(self, request, servicer_context):
    """Echoes the input user message to test the server.

    Args:
      request: A service_pb2.Echo object containing the message user wants to
        echo.
      servicer_context: A grpc.ServicerContext for use during service of the
        RPC.

    Returns:
      A service_pb2.Echo object containing the message to echo.
    """
    return service_pb2.EchoResponse(msg=request.msg)
