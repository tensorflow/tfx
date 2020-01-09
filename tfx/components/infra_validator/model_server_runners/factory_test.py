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
"""Tests for tfx.components.infra_validator.model_server_runners.factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import mock
import tensorflow as tf

from google.protobuf import json_format
from tfx.components import testdata
from tfx.components.infra_validator.model_server_runners import factory
from tfx.components.infra_validator.model_server_runners import local_docker_runner
from tfx.proto import infra_validator_pb2
from tfx.types import standard_artifacts

LocalDockerModelServerRunner = local_docker_runner.LocalDockerModelServerRunner


class FactoryTest(tf.test.TestCase):

  def setUp(self):
    super(FactoryTest, self).setUp()

    base_dir = os.path.dirname(testdata.__file__)
    self.model = standard_artifacts.Model()
    self.model.uri = os.path.join(base_dir, 'trainer', 'current')

  @staticmethod
  def _CreateServingSpec(payload):
    return json_format.ParseDict(payload, infra_validator_pb2.ServingSpec())

  @mock.patch('docker.DockerClient')
  def testCreateModelServerRunners_LocalDockerRunner(self, mock_docker):
    spec = self._CreateServingSpec({
        'tensorflow_serving': {
            'tags': ['1.15.0']
        },
        'local_docker': {}
    })

    # Run factory.
    runners = list(factory.create_model_server_runners(self.model, spec))

    # Check result.
    self.assertEqual(1, len(runners))
    self.assertIsInstance(runners[0], LocalDockerModelServerRunner)

  @mock.patch('docker.DockerClient')
  def testCreateModelServerRunners_MultipleRunners(self, mock_docker):
    spec = self._CreateServingSpec({
        'tensorflow_serving': {
            'tags': ['1.14.0', '1.15.0']
        },
        'local_docker': {}
    })

    # Run factory.
    runners = list(factory.create_model_server_runners(self.model, spec))

    # Check result.
    self.assertEqual(2, len(runners))
    self.assertIsInstance(runners[0], LocalDockerModelServerRunner)
    self.assertIsInstance(runners[1], LocalDockerModelServerRunner)

  def testCreateModelServerRunners_FailsIfNoPlatformSpecified(self):
    spec = self._CreateServingSpec({
        'tensorflow_serving': {
            'tags': ['1.15.0']
        }
    })

    with self.assertRaises(NotImplementedError):
      factory.create_model_server_runners(self.model, spec)


if __name__ == '__main__':
  tf.test.main()
