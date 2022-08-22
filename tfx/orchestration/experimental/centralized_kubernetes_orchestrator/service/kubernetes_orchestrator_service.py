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

from typing import Dict

import grpc
from tfx.orchestration import metadata
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service.proto import service_pb2
from tfx.orchestration.experimental.centralized_kubernetes_orchestrator.service.proto import service_pb2_grpc
from tfx.orchestration.experimental.core import pipeline_ops
from tfx.utils import status as status_lib

_CANONICAL_TO_GRPC_CODES: Dict[int, grpc.StatusCode] = {
    status_lib.Code.OK: grpc.StatusCode.OK,
    status_lib.Code.CANCELLED: grpc.StatusCode.CANCELLED,
    status_lib.Code.UNKNOWN: grpc.StatusCode.UNKNOWN,
    status_lib.Code.INVALID_ARGUMENT: grpc.StatusCode.INVALID_ARGUMENT,
    status_lib.Code.DEADLINE_EXCEEDED: grpc.StatusCode.DEADLINE_EXCEEDED,
    status_lib.Code.NOT_FOUND: grpc.StatusCode.NOT_FOUND,
    status_lib.Code.ALREADY_EXISTS: grpc.StatusCode.ALREADY_EXISTS,
    status_lib.Code.PERMISSION_DENIED: grpc.StatusCode.PERMISSION_DENIED,
    status_lib.Code.RESOURCE_EXHAUSTED: grpc.StatusCode.RESOURCE_EXHAUSTED,
    status_lib.Code.FAILED_PRECONDITION: grpc.StatusCode.FAILED_PRECONDITION,
    status_lib.Code.ABORTED: grpc.StatusCode.ABORTED,
    status_lib.Code.OUT_OF_RANGE: grpc.StatusCode.OUT_OF_RANGE,
    status_lib.Code.UNIMPLEMENTED: grpc.StatusCode.UNIMPLEMENTED,
    status_lib.Code.INTERNAL: grpc.StatusCode.INTERNAL,
    status_lib.Code.UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
    status_lib.Code.DATA_LOSS: grpc.StatusCode.DATA_LOSS,
    status_lib.Code.UNAUTHENTICATED: grpc.StatusCode.UNAUTHENTICATED,
}


class KubernetesOrchestratorServicer(
    service_pb2_grpc.KubernetesOrchestratorServicer):
  """A service interface for pipeline orchestration."""

  def __init__(self, mlmd_handle: metadata.Metadata):
    self._mlmd_handle = mlmd_handle

  def Echo(self, request: service_pb2.EchoRequest,
           servicer_context: grpc.ServicerContext):
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

  def StartPipeline(
      self, request: service_pb2.StartPipelineRequest,
      context: grpc.ServicerContext) -> service_pb2.StartPipelineResponse:
    try:
      pipeline_ops.initiate_pipeline_start(self._mlmd_handle, request.pipeline)
    except status_lib.StatusNotOkError as e:
      context.set_code(_CANONICAL_TO_GRPC_CODES[e.code])
      context.set_details(e.message)
    return service_pb2.StartPipelineResponse()
