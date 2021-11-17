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
"""Module for TensorFlowServingClient."""

from absl import logging
import grpc
from tfx.components.infra_validator import types
from tfx.components.infra_validator.model_server_clients import base_client

from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import get_model_status_pb2
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import model_service_pb2_grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import regression_pb2

State = get_model_status_pb2.ModelVersionStatus.State


class TensorFlowServingClient(base_client.BaseModelServerClient):
  """A model server client for TensorFlow Serving.

  It uses gRPC client to talk to TensorFlow Serving server.
  """

  def __init__(self, endpoint: str, model_name: str):
    # Note that the channel instance is automatically closed (unsubscribed) on
    # deletion, so we don't have to manually close this on __del__.
    self._channel = grpc.insecure_channel(endpoint)
    self._model_name = model_name
    self._model_service = model_service_pb2_grpc.ModelServiceStub(self._channel)
    self._prediction_service = prediction_service_pb2_grpc.PredictionServiceStub(self._channel)  # pylint: disable=line-too-long

  def _GetModelStatus(self) -> get_model_status_pb2.GetModelStatusResponse:
    """Call GetModelStatus() from model service.

    https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/model_service.proto

    Returns:
      GetModelStatusResponse from GetModelStatus().
    """
    request = get_model_status_pb2.GetModelStatusRequest(
        model_spec=model_pb2.ModelSpec(name=self._model_name))
    return self._model_service.GetModelStatus(request)

  def _GetServingStatus(self) -> types.ModelServingStatus:
    """Check whether the model is available for query or not.

    In TensorFlow Serving, model is READY if and only if the state from
    _GetModelStatus() is AVAILABLE. If returned state is END, it will never
    become READY therefore returns UNAVAILABLE. Otherwise it will return
    NOT_READY.

    Returns:
      A ModelState.
    """
    try:
      resp = self._GetModelStatus()
    except grpc.RpcError as e:
      logging.info('Model status is not available yet:\n%s', e)
      return types.ModelServingStatus.NOT_READY

    # When no versions available. (empty list)
    if not resp.model_version_status:
      return types.ModelServingStatus.NOT_READY

    # Wait until all serving model versions are in AVAILABLE state.
    # In TensorFlow Serving, model state lifecycle is
    #     START -> LOADING -> AVAILABLE -> UNLOADING -> END
    # if loaded successfully or
    #     START -> LOADING -> END
    # if loaded unsuccessfully. The model is available iff state is AVAILABLE.
    # The model is unavailable for goods iff state is END.
    # https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/get_model_status.proto
    if all(mvs.state == State.AVAILABLE
           for mvs in resp.model_version_status):
      return types.ModelServingStatus.READY
    if any(mvs.state == State.END
           for mvs in resp.model_version_status):
      return types.ModelServingStatus.UNAVAILABLE
    return types.ModelServingStatus.NOT_READY

  def _SendRequest(self, request: types.Request) -> None:
    if isinstance(request, classification_pb2.ClassificationRequest):
      self._prediction_service.Classify(request)
    elif isinstance(request, regression_pb2.RegressionRequest):
      self._prediction_service.Regress(request)
    elif isinstance(request, predict_pb2.PredictRequest):
      self._prediction_service.Predict(request)
    else:
      raise NotImplementedError('Unsupported request type {}'.format(
          type(request).__name__))
