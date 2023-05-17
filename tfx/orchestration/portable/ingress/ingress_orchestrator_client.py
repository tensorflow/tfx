# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Orchestrator client primiarly created to suit Ingress use cases."""

import collections
from collections.abc import Mapping, Sequence

import grpc
from tfx import types
from tfx.orchestration.experimental.core.service.proto import service_pb2
from tfx.orchestration.experimental.core.service.proto import service_pb2_grpc
from tfx.proto.orchestration import execution_result_pb2
from tfx.types import artifact_utils

from tfx.orchestration.portable.ingress import utils


class IngressOrchestratorClient:
  """Orchestrator RPC client primiarly used by Ingress components to call Orchestrator.

  If only orchestrator_address is provided, a new Orchestrator channel will be
  created to set up RPC stub.
  If only channel is provided, the channel will be used to set up RPC stub.
  If both orchestrator_address and channel are provided, channel will be used.
  In general, orchestrator address should be used in production code and
  channel should be used for easier unit test set up.

  Usage:
    client = IngressOrchestratorClient(orchestrator_address=...)
    client.get_node_live_output_artifacts_by_output_key(...)
    client.publish_external_output_artifacts(...)
  """

  def __init__(
      self,
      orchestrator_address: str = "",
      channel: grpc.Channel | None = None,
  ):
    if channel:
      self._stub = service_pb2_grpc.OrchestratorStub(channel)
      return

    if orchestrator_address:
      creds = utils.get_credentials(orchestrator_address)
      channel = grpc.secure_channel(orchestrator_address, creds)
      self._stub = service_pb2_grpc.OrchestratorStub(channel)
      return

    raise RuntimeError(
        "Both channel and orchestrator_address are empty. At least one "
        "of them needs to be provided with an non-empty value."
    )

  def get_node_live_output_artifacts_by_output_key(
      self,
      pipeline_id: str,
      pipeline_run_id: str,
      node_id: str,
      execution_limit: int = 0,
  ) -> Mapping[str, Sequence[Sequence[types.Artifact]]]:
    """Gets LIVE output artifacts grouped by output key for the node by calling Orchestrator RPC.

      1. If execution_limit = 0, live output artifacts from all executions will
        be returned.
      2. If execution_limit > 0 and the node has fewer executions than
        execution_limit, LIVE output artifacts from all executions will be
        returned.
      3. If execution_limit > 0 and the node has more or equal executions than
        execution_limit, only LIVE output artifacts from the execution_limit
        latest executions will be returned.

    Args:
      pipeline_id: The pipeline ID.
      pipeline_run_id: The pipeline run ID that the node belongs to. Should be
        empty in ASYNC mode and non-empty in SYNC mode.
      node_id: The node ID.
      execution_limit: Maximum number of latest successful executions from which
        LIVE output artifacts will be returned, must be non-negative.

    Returns:
      A mapping from output key to LIVE output artifacts for the given node.
      Output artifacts are returned in a two-layer nested list. The outer layer
      represents output artfacts from different executions ordered in the
      descending order of execution creatime time (latest to oldest). The
      inner layer represents output artifacts from one execution.
    """
    request = service_pb2.GetNodeLiveOutputArtifactsByOutputKeyRequest(
        pipeline_id=pipeline_id,
        pipeline_run_id=pipeline_run_id,
        node_id=node_id,
        execution_limit=execution_limit,
    )
    response = self._stub.GetNodeLiveOutputArtifactsByOutputKey(request)
    result = collections.defaultdict(list)
    for (
        output_key,
        execution_artifacts_list,
    ) in response.execution_artifacts_list_by_output_key.items():
      for execution_artifacts in execution_artifacts_list.execution_artifacts:
        if not execution_artifacts.artifacts:
          result[output_key].append([])
        else:
          result[output_key].append(
              artifact_utils.deserialize_artifacts(
                  execution_artifacts.artifact_type,
                  execution_artifacts.artifacts,
              )
          )
    return result

  def publish_external_output_artifacts(
      self,
      pipeline_id: str,
      pipeline_run_id: str,
      node_id: str,
      executor_output: execution_result_pb2.ExecutorOutput,
  ) -> None:
    """Publishes output artifacts based on executor output by calling Orchestrator RPC.

    If executor output contains OK execution result, output artifacts will be
    published with a new COMPLETE execution. Published artifacts will be marked
    as LIVE external artifacts (go/tflex-external-artifacts).

    If executor output contains non-OK execution result, a FAILED execution will
    be published with no artifacts.

    Args:
      pipeline_id: The pipeline ID.
      pipeline_run_id: The pipeline run ID that the node belongs to. Should be
        empty in ASYNC mode and non-empty in SYNC mode.
      node_id: The node ID.
      executor_output: Executor output for the execution.
    """
    request = service_pb2.PublishExternalOutputArtifactsRequest(
        pipeline_id=pipeline_id,
        pipeline_run_id=pipeline_run_id,
        node_id=node_id,
        executor_output=executor_output,
    )
    self._stub.PublishExternalOutputArtifacts(request)
