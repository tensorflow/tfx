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
"""Tests for tfx.components.infra_validator.model_server_runners.factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import mock
import tensorflow as tf
from typing import Any, Dict, Text

from google.protobuf import json_format
from tfx.components.infra_validator.model_server_runners import factory
from tfx.components.infra_validator.model_server_runners import local_docker_runner
from tfx.proto import infra_validator_pb2
from tfx.types import standard_artifacts

LocalDockerModelServerRunner = local_docker_runner.LocalDockerRunner


def _create_serving_spec(serving_spec_dict: Dict[Text, Any]):
  serving_spec = infra_validator_pb2.ServingSpec()
  json_format.ParseDict(serving_spec_dict, serving_spec)
  return serving_spec


class FactoryTest(tf.test.TestCase):

  def setUp(self):
    super(FactoryTest, self).setUp()

    base_dir = os.path.join(
        os.path.dirname(  # components
            os.path.dirname(  # infra_validator
                os.path.dirname(__file__))),  # model_server_runners
        'testdata'
    )
    self.model = standard_artifacts.Model()
    self.model.uri = os.path.join(base_dir, 'trainer', 'current')

  @mock.patch('docker.DockerClient')
  def testCreateModelServerRunners_LocalDockerRunner(self, mock_docker):
    spec = _create_serving_spec({
        'tensorflow_serving': {
            'tags': ['1.15.0']
        },
        'local_docker': {},
        'model_name': 'chicago-taxi',
    })

    # Run factory.
    runners = list(factory.create_model_server_runners(self.model, spec))

    # Check result.
    self.assertEqual(1, len(runners))
    self.assertIsInstance(runners[0], LocalDockerModelServerRunner)

  @mock.patch('docker.DockerClient')
  def testCreateModelServerRunners_MultipleRunners(self, mock_docker):
    spec = _create_serving_spec({
        'tensorflow_serving': {
            'tags': ['1.14.0', '1.15.0']
        },
        'local_docker': {},
        'model_name': 'chicago-taxi',
    })

    # Run factory.
    runners = list(factory.create_model_server_runners(self.model, spec))

    # Check result.
    self.assertEqual(2, len(runners))
    self.assertIsInstance(runners[0], LocalDockerModelServerRunner)
    self.assertIsInstance(runners[1], LocalDockerModelServerRunner)

  def testCreateModelServerRunners_FailsIfNoPlatformSpecified(self):
    spec = _create_serving_spec({
        'tensorflow_serving': {
            'tags': ['1.15.0']
        },
        'model_name': 'chicago-taxi',
    })

    with self.assertRaises(NotImplementedError):
      factory.create_model_server_runners(self.model, spec)


if __name__ == '__main__':
  tf.test.main()
