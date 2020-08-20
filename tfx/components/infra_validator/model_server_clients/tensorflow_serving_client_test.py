# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Tests for tfx.components.infra_validator.model_server_clients.tensorflow_serving_client."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import grpc
import mock
import tensorflow as tf
from typing import Any, Dict, Text

from google.protobuf import json_format
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis import regression_pb2
from tfx.components.infra_validator import error_types
from tfx.components.infra_validator import types
from tfx.components.infra_validator.model_server_clients import tensorflow_serving_client


def _make_response(
    payload: Dict[Text, Any]) -> get_model_status_pb2.GetModelStatusResponse:
  result = get_model_status_pb2.GetModelStatusResponse()
  json_format.ParseDict(payload, result)
  return result


class TensorflowServingClientTest(tf.test.TestCase):

  def setUp(self):
    super(TensorflowServingClientTest, self).setUp()
    self.model_stub_patcher = mock.patch('tensorflow_serving.apis.model_service_pb2_grpc.ModelServiceStub')  # pylint: disable=line-too-long
    self.model_stub_cls = self.model_stub_patcher.start()
    self.model_stub = self.model_stub_cls.return_value
    self.prediction_stub_patcher = mock.patch('tensorflow_serving.apis.prediction_service_pb2_grpc.PredictionServiceStub')  # pylint: disable=line-too-long
    self.prediction_stub_cls = self.prediction_stub_patcher.start()
    self.prediction_stub = self.prediction_stub_cls.return_value

  def tearDown(self):
    super(TensorflowServingClientTest, self).tearDown()
    self.model_stub_patcher.stop()
    self.prediction_stub_patcher.stop()

  def testGetModelState_ReturnsReady_IfAllAvailable(self):
    # Prepare stub and client.
    self.model_stub.GetModelStatus.return_value = _make_response({
        'model_version_status': [
            {'state': 'AVAILABLE'},
            {'state': 'AVAILABLE'},
            {'state': 'AVAILABLE'}
        ]
    })
    client = tensorflow_serving_client.TensorFlowServingClient(
        'localhost:1234', 'a_model_name')

    # Call.
    result = client._GetServingStatus()

    # Check result.
    self.assertEqual(result, types.ModelServingStatus.READY)

  def testGetModelState_ReturnsNotReady_IfAnyStateNotAvailable(self):
    # Prepare stub and client.
    self.model_stub.GetModelStatus.return_value = _make_response({
        'model_version_status': [
            {'state': 'AVAILABLE'},
            {'state': 'AVAILABLE'},
            {'state': 'LOADING'}
        ]
    })
    client = tensorflow_serving_client.TensorFlowServingClient(
        'localhost:1234', 'a_model_name')

    # Call.
    result = client._GetServingStatus()

    # Check result.
    self.assertEqual(result, types.ModelServingStatus.NOT_READY)

  def testGetModelState_ReturnsUnavailable_IfAnyStateEnded(self):
    # Prepare stub and client.
    self.model_stub.GetModelStatus.return_value = _make_response({
        'model_version_status': [
            {'state': 'AVAILABLE'},
            {'state': 'AVAILABLE'},
            {'state': 'END'}
        ]
    })
    client = tensorflow_serving_client.TensorFlowServingClient(
        'localhost:1234', 'a_model_name')

    # Call.
    result = client._GetServingStatus()

    # Check result.
    self.assertEqual(result, types.ModelServingStatus.UNAVAILABLE)

  def testGetModelState_ReturnsNotReady_IfEmptyState(self):
    # Prepare stub and client.
    self.model_stub.GetModelStatus.return_value = _make_response({
        'model_version_status': []  # Empty
    })
    client = tensorflow_serving_client.TensorFlowServingClient(
        'localhost:1234', 'a_model_name')

    # Calls
    result = client._GetServingStatus()

    # Check result.
    self.assertEqual(result, types.ModelServingStatus.NOT_READY)

  def testGetModelState_ReturnsNotReady_IfServerUnavailable(self):
    # Prepare stub and client.
    self.model_stub.GetModelStatus.side_effect = grpc.RpcError
    client = tensorflow_serving_client.TensorFlowServingClient(
        'localhost:1234', 'a_model_name')

    # Call.
    result = client._GetServingStatus()

    # Check result.
    self.assertEqual(result, types.ModelServingStatus.NOT_READY)

  def testIssueRequests_NoErrorIfSucceeded(self):
    # Prepare requests and client.
    r1 = classification_pb2.ClassificationRequest()
    r2 = classification_pb2.ClassificationRequest()
    r3 = regression_pb2.RegressionRequest()
    client = tensorflow_serving_client.TensorFlowServingClient(
        'localhost:1234', 'a_model_name')

    # Call.
    client.SendRequests([r1, r2, r3])

    # Check calls
    self.prediction_stub.Classify.assert_called_with(r1)
    self.prediction_stub.Classify.assert_called_with(r2)
    self.prediction_stub.Regress.assert_called_with(r3)

  def testIssueRequests_RaiseValueErrorOnUnrecognizedRequestType(self):
    # Prepare requests and client.
    not_a_request = 'i am a request'
    client = tensorflow_serving_client.TensorFlowServingClient(
        'localhost:1234', 'a_model_name')

    # Call
    with self.assertRaises(error_types.ValidationFailed):
      client.SendRequests([not_a_request])

  def testIssueRequests_RaiseRpcErrorIfRpcFailed(self):
    # Prepare client and a side effect.
    request = classification_pb2.ClassificationRequest()
    client = tensorflow_serving_client.TensorFlowServingClient(
        'localhost:1234', 'a_model_name')
    self.prediction_stub.Classify.side_effect = grpc.RpcError

    # Call.
    with self.assertRaises(error_types.ValidationFailed):
      client.SendRequests([request])


if __name__ == '__main__':
  tf.test.main()
