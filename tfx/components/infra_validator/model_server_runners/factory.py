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
"""Factory for making model server runners."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Iterable

from tfx.components.infra_validator import binary_kinds
from tfx.components.infra_validator.model_server_runners import base_runner
from tfx.components.infra_validator.model_server_runners import local_docker_runner
from tfx.proto import infra_validator_pb2
from tfx.types import standard_artifacts


def create_model_server_runners(
    model: standard_artifacts.Model,
    serving_spec: infra_validator_pb2.ServingSpec
) -> Iterable[base_runner.BaseModelServerRunner]:
  """Create model server runners based on given model and serving spec.

  In ServingSpec you can specify multiple versions for validation on single
  image. In such case it returns multiple model server runners for each
  (image, version) pair.

  Args:
    model: A model artifact whose uri contains the path to the servable model.
    serving_spec: A ServingSpec configuration.

  Returns:
    An iterable of `BaseModelServerRunner`.
  """
  result = []
  platform_kind = serving_spec.WhichOneof('serving_platform')
  if platform_kind == 'local_docker':
    for binary_kind in binary_kinds.parse_binary_kinds(serving_spec):
      result.append(local_docker_runner.LocalDockerRunner(
          model=model,
          binary_kind=binary_kind,
          serving_spec=serving_spec
      ))
  else:
    raise NotImplementedError('{} platform is not yet supported'
                              .format(platform_kind))
  return result
