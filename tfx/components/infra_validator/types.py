# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Common types that are used across infra_validator implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
from typing import Union

from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import regression_pb2

TensorFlowServingRequest = Union[
    classification_pb2.ClassificationRequest,
    regression_pb2.RegressionRequest,
    predict_pb2.PredictRequest,
]

Request = Union[TensorFlowServingRequest]


class ModelServingStatus(enum.Enum):
  """Serving status of the model in the model server."""
  # Model is not ready yet but will be ready soon.
  NOT_READY = 1
  # Model is ready.
  READY = 2
  # Failed to load a model and will not be recovered. Indicates infra validation
  # failure.
  UNAVAILABLE = 3
