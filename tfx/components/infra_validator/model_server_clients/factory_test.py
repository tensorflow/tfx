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
"""Tests for tfx.components.infra_validator.model_server_clients.factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import mock
import tensorflow as tf

from google.protobuf import json_format
from tfx.components.infra_validator.model_server_clients import factory
from tfx.components.infra_validator.model_server_clients import tensorflow_serving_client
from tfx.proto import infra_validator_pb2
from tfx.types import standard_artifacts

ServingSpec = infra_validator_pb2.ServingSpec
TensorFlowServingClient = tensorflow_serving_client.TensorFlowServingClient


class FactoryTest(tf.test.TestCase):

  def setUp(self):
    super(FactoryTest, self).setUp()

    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'testdata')
    self.model = mock.Mock(standard_artifacts.Model)
    self.model.uri = os.path.join(base_dir, 'trainer', 'current')
    self.model_name = 'chicago-taxi'

  @staticmethod
  def _CreateServingSpec(payload):
    return json_format.ParseDict(payload, ServingSpec())

  @mock.patch('tfx.components.infra_validator.model_server_clients.tensorflow_serving_client.TensorFlowServingClient')  # pylint: disable=line-too-long
  def testGetTensorFlowServingClientFactory(self, mock_client_cls):
    # Prepare serving binary with tensorflow_serving.
    serving_binary = self._CreateServingSpec({
        'tensorflow_serving': {}
    })

    # Make client.
    client_factory = factory.make_client_factory(self.model, serving_binary)
    client_factory('localhost:1234')

    # Check client.
    mock_client_cls.assert_called_with('localhost:1234',
                                       model_name=self.model_name)

  def testEmptyServingSpecFails(self):
    serving_binary = self._CreateServingSpec({})

    # Make client.
    with self.assertRaises(ValueError) as err:
      factory.make_client_factory(self.model, serving_binary)

    # Check exception.
    self.assertEqual(str(err.exception), 'serving_binary must be set.')


if __name__ == '__main__':
  tf.test.main()
