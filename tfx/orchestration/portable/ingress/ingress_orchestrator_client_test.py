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
"""Unit Tests for tfx.orchestration.portable.ingress.ingress_orchestrator_client."""

from unittest import mock

from absl.testing import parameterized
import grpc
import tensorflow as tf
from tfx.orchestration.experimental.core.service.proto import service_pb2
from tfx.orchestration.portable.ingress import ingress_orchestrator_client
from tfx.proto.orchestration import execution_result_pb2
from tfx.utils import test_case_utils

from ml_metadata.proto import metadata_store_pb2

ExecutionArtifactsList = (
    service_pb2.GetNodeLiveOutputArtifactsByOutputKeyResponse.ExecutionArtifactsList
)
ExecutionArtifacts = (
    service_pb2.GetNodeLiveOutputArtifactsByOutputKeyResponse.ExecutionArtifactsList.ExecutionArtifacts
)


class IngressOrchestratorClientTest(
    test_case_utils.TfxTest, parameterized.TestCase
):

  def setUp(self):
    super().setUp()

    self.channel = mock.create_autospec(grpc.Channel, instance=True)
    self.orchestrator_client = (
        ingress_orchestrator_client.IngressOrchestratorClient(
            channel=self.channel
        )
    )
    self.stub = self.orchestrator_client._stub
    self.pipeline_id = 'ingress-pipeline'
    self.pipeline_run_id = 'run-123456'
    self.node_id = 'myingress'

  @parameterized.named_parameters(('Async', 'run-123456'), ('Sync', ''))
  def testGetNodeLiveOutputArtifactsByOutputKey(self, pipeline_run_id: str):
    request = service_pb2.GetNodeLiveOutputArtifactsByOutputKeyRequest(
        pipeline_id=self.pipeline_id,
        pipeline_run_id=pipeline_run_id,
        node_id=self.node_id,
        execution_limit=3,
    )
    mlmd_artifact_1 = metadata_store_pb2.Artifact(id=101)
    mlmd_artifact_2 = metadata_store_pb2.Artifact(id=102)
    mlmd_artifact_3 = metadata_store_pb2.Artifact(id=103)
    mlmd_artifact_4 = metadata_store_pb2.Artifact(id=104)
    mlmd_artifact_5 = metadata_store_pb2.Artifact(id=105)
    mlmd_artifact_6 = metadata_store_pb2.Artifact(id=106)
    integer_artifact_type = metadata_store_pb2.ArtifactType(name='Integer')
    examples_source_artifact_type = metadata_store_pb2.ArtifactType(
        name='ExamplesSource'
    )
    examples_source_execution_artifacts_list = ExecutionArtifactsList(
        execution_artifacts=[
            ExecutionArtifacts(
                artifact_type=examples_source_artifact_type,
                artifacts=[mlmd_artifact_1, mlmd_artifact_2],
            ),
            ExecutionArtifacts(
                artifact_type=examples_source_artifact_type,
                artifacts=[mlmd_artifact_3],
            ),
            ExecutionArtifacts(
                artifact_type=examples_source_artifact_type,
                artifacts=[],
            ),
        ]
    )
    span_execution_artifacts_list = ExecutionArtifactsList(
        execution_artifacts=[
            ExecutionArtifacts(
                artifact_type=integer_artifact_type,
                artifacts=[mlmd_artifact_4, mlmd_artifact_5],
            ),
            ExecutionArtifacts(
                artifact_type=integer_artifact_type,
                artifacts=[mlmd_artifact_6],
            ),
            ExecutionArtifacts(
                artifact_type=integer_artifact_type,
                artifacts=[],
            ),
        ]
    )
    execution_artifacts_list_by_output_key = {
        'examples_source': examples_source_execution_artifacts_list,
        'span': span_execution_artifacts_list,
    }
    self.stub.GetNodeLiveOutputArtifactsByOutputKey.return_value = service_pb2.GetNodeLiveOutputArtifactsByOutputKeyResponse(
        execution_artifacts_list_by_output_key=execution_artifacts_list_by_output_key
    )

    actual_result = (
        self.orchestrator_client.get_node_live_output_artifacts_by_output_key(
            pipeline_id=self.pipeline_id,
            pipeline_run_id=pipeline_run_id,
            node_id=self.node_id,
            execution_limit=3,
        )
    )

    self.stub.GetNodeLiveOutputArtifactsByOutputKey.assert_called_once_with(
        request
    )
    self.assertLen(actual_result, 2)
    actual_examples_source_artifacts = actual_result['examples_source']
    self.assertLen(actual_examples_source_artifacts, 3)
    self.assertEqual(actual_examples_source_artifacts[0][0].id, 101)
    self.assertEqual(actual_examples_source_artifacts[0][1].id, 102)
    self.assertEqual(actual_examples_source_artifacts[1][0].id, 103)
    self.assertEmpty(actual_examples_source_artifacts[2])
    actual_span_artifacts = actual_result['span']
    self.assertLen(actual_span_artifacts, 3)
    self.assertEqual(actual_span_artifacts[0][0].id, 104)
    self.assertEqual(actual_span_artifacts[0][1].id, 105)
    self.assertEqual(actual_span_artifacts[1][0].id, 106)
    self.assertEmpty(actual_span_artifacts[2])

  @parameterized.named_parameters(('Async', 'run-123456'), ('Sync', ''))
  def testGetNodeLiveOutputArtifactsByOutputKeyEmpty(
      self, pipeline_run_id: str
  ):
    request = service_pb2.GetNodeLiveOutputArtifactsByOutputKeyRequest(
        pipeline_id=self.pipeline_id,
        pipeline_run_id=pipeline_run_id,
        node_id=self.node_id,
        execution_limit=3,
    )
    self.stub.GetNodeLiveOutputArtifactsByOutputKey.return_value = (
        service_pb2.GetNodeLiveOutputArtifactsByOutputKeyResponse(
            execution_artifacts_list_by_output_key={}
        )
    )
    actual_result = (
        self.orchestrator_client.get_node_live_output_artifacts_by_output_key(
            pipeline_id=self.pipeline_id,
            pipeline_run_id=pipeline_run_id,
            node_id=self.node_id,
            execution_limit=3,
        )
    )
    self.stub.GetNodeLiveOutputArtifactsByOutputKey.assert_called_once_with(
        request
    )
    self.assertEmpty(actual_result)

  @parameterized.named_parameters(('Async', 'run-123456'), ('Sync', ''))
  def testGetNodeLiveOutputArtifactsByOutputKeyFailure(
      self, pipeline_run_id: str
  ):
    self.stub.GetNodeLiveOutputArtifactsByOutputKey.side_effect = (
        grpc.RpcError()
    )
    with self.assertRaises(grpc.RpcError):
      self.orchestrator_client.get_node_live_output_artifacts_by_output_key(
          pipeline_id=self.pipeline_id,
          pipeline_run_id=pipeline_run_id,
          node_id=self.node_id,
          execution_limit=3,
      )

  @parameterized.named_parameters(('Async', 'run-123456'), ('Sync', ''))
  def testPublishExternalOutputArtifacts(self, pipeline_run_id: str):
    executor_output = execution_result_pb2.ExecutorOutput(
        execution_result=execution_result_pb2.ExecutionResult(code=0)
    )
    request = service_pb2.PublishExternalOutputArtifactsRequest(
        pipeline_id=self.pipeline_id,
        pipeline_run_id=pipeline_run_id,
        node_id=self.node_id,
        executor_output=executor_output,
    )
    self.stub.PublishExternalOutputArtifacts.return_value = (
        service_pb2.PublishExternalOutputArtifactsResponse()
    )
    actual_result = self.orchestrator_client.publish_external_output_artifacts(
        pipeline_id=self.pipeline_id,
        pipeline_run_id=pipeline_run_id,
        node_id=self.node_id,
        executor_output=executor_output,
    )
    self.stub.PublishExternalOutputArtifacts.assert_called_once_with(request)
    self.assertIsNone(actual_result)

  @parameterized.named_parameters(('Async', 'run-123456'), ('Sync', ''))
  def testPublishExternalOutputArtifactsFailure(self, pipeline_run_id: str):
    self.stub.PublishExternalOutputArtifacts.side_effect = grpc.RpcError()
    executor_output = execution_result_pb2.ExecutorOutput(
        execution_result=execution_result_pb2.ExecutionResult(code=0)
    )
    with self.assertRaises(grpc.RpcError):
      self.orchestrator_client.publish_external_output_artifacts(
          pipeline_id=self.pipeline_id,
          pipeline_run_id=pipeline_run_id,
          node_id=self.node_id,
          executor_output=executor_output,
      )


if __name__ == '__main__':
  tf.test.main()
