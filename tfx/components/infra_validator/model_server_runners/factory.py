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

from tfx.components.infra_validator.model_server_clients import factory
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
  platform_kind = serving_spec.WhichOneof('serving_platform')
  if platform_kind == 'local_docker':
    return _create_local_docker_runners(model, serving_spec)
  else:
    raise NotImplementedError('{} platform is not yet supported'
                              .format(platform_kind))


def _create_local_docker_runners(
    model: standard_artifacts.Model,
    serving_spec: infra_validator_pb2.ServingSpec,
) -> Iterable[base_runner.BaseModelServerRunner]:
  client_factory = factory.make_client_factory(serving_spec)
  for image_uri in _build_docker_uris(serving_spec):
    yield local_docker_runner.LocalDockerModelServerRunner(
        model=model,
        image_uri=image_uri,
        serving_spec=serving_spec,
        client_factory=client_factory)


def _build_docker_uris(serving_spec):
  binary_kind = serving_spec.WhichOneof('serving_binary')
  if binary_kind == 'tensorflow_serving':
    for tag in serving_spec.tensorflow_serving.tags:
      yield 'tensorflow/serving:{}'.format(tag)
    for digest in serving_spec.tensorflow_serving.digests:
      yield 'tensorflow/serving@{}'.format(digest)
  else:
    raise NotImplementedError('{} binary is not yet supported'
                              .format(binary_kind))

