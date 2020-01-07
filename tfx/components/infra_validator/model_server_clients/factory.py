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
"""Factory for making model server clients."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
from typing import Callable, Text

from tfx.components.infra_validator.model_server_clients import base_client
from tfx.components.infra_validator.model_server_clients import tensorflow_serving_client
from tfx.proto import infra_validator_pb2
from tfx.types import standard_artifacts
from tfx.utils import path_utils

# ClientFactory gets endpoint as an argument and returns BaseModelServerClient.
ClientFactory = Callable[[Text], base_client.BaseModelServerClient]

TENSORFLOW_SERVING = 'tensorflow_serving'


def make_client_factory(
    model: standard_artifacts.Model,
    serving_spec: infra_validator_pb2.ServingSpec) -> ClientFactory:
  """Creates ClientFactory from Model artifact and ServingSpec configuration.

  Note that for each `serving_binary` in ServingSpec there is a corresponding
  ModelServerClient class. (1on1 mapping)

  Args:
    model: A `Model` artifact.
    serving_spec: A `ServingSpec` configuration.

  Returns:
    A ModelServerClient factory function that takes Text endpoint as an argument
    and returns a ModelServerClient.
  """
  serving_binary = serving_spec.WhichOneof('serving_binary')
  if not serving_binary:
    raise ValueError('serving_binary must be set.')

  if serving_binary == TENSORFLOW_SERVING:
    model_name = os.path.basename(
        os.path.dirname(path_utils.serving_model_path(model.uri)))
    return functools.partial(
        tensorflow_serving_client.TensorFlowServingClient,
        model_name=model_name)
  else:
    raise NotImplementedError('{} is not supported'.format(serving_binary))
