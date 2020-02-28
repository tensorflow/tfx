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
"""Tests for tfx.components.infra_validator.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import mock
import tensorflow as tf
from typing import Text

from tfx.components.infra_validator import executor
from tfx.components.infra_validator.model_server_runners import local_docker_runner
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()

    # Setup Mocks

    runner_patcher = mock.patch.object(local_docker_runner, 'LocalDockerRunner')
    self.model_server = runner_patcher.start().return_value
    self.addCleanup(runner_patcher.stop)

    build_request_patcher = mock.patch(
        'tfx.components.infra_validator.request_builder'
        '.build_requests')
    self.build_requests_mock = build_request_patcher.start()
    self.addCleanup(build_request_patcher.stop)

    self.model_server.client = mock.MagicMock()

    # Setup directories

    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'testdata')
    base_output_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                                     self.get_temp_dir())
    output_data_dir = os.path.join(base_output_dir, self._testMethodName)

    # Setup input_dict.

    model = standard_artifacts.Model()
    model.uri = os.path.join(source_data_dir, 'trainer', 'current')
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir,
                                'transform',
                                'transformed_examples',
                                'eval')
    examples.split_names = artifact_utils.encode_split_names(['eval'])

    self.input_dict = {
        'model': [model],
        'examples': [examples],
    }

    # Setup output_dict.

    self.blessing = standard_artifacts.InfraBlessing()
    self.blessing.uri = os.path.join(output_data_dir, 'blessing')
    self.output_dict = {
        'blessing': [self.blessing]
    }

    # Setup Context

    temp_dir = os.path.join(output_data_dir, '.temp')
    self.context = executor.Executor.Context(tmp_dir=temp_dir, unique_id='1')

    # Setup exec_properties

    self.exec_properties = {
        'serving_spec': json.dumps({
            'tensorflow_serving': {
                'tags': ['1.15.0']
            },
            'local_docker': {},
            'model_name': 'chicago-taxi',
        }),
        'validation_spec': json.dumps({
            'max_loading_time_seconds': 10
        }),
        'request_spec': json.dumps({
            'tensorflow_serving': {
                'rpc_kind': 'CLASSIFY'
            },
            'max_examples': 10
        })
    }

  def testDo_LoadOnly(self):
    # Prepare inputs and mocks.
    input_dict = self.input_dict.copy()
    input_dict.pop('examples')
    exec_properties = self.exec_properties.copy()
    exec_properties.pop('request_spec')
    self.model_server.WaitUntilModelAvailable.return_value = True

    # Run executor.
    infra_validator = executor.Executor(self.context)
    infra_validator.Do(input_dict, self.output_dict, exec_properties)

    # Check output artifact.
    self.assertFileExists(os.path.join(self.blessing.uri, 'INFRA_BLESSED'))
    self.assertEqual(1, self.blessing.get_int_custom_property('blessed'))

    # Check cleanup done.
    self.model_server.Stop.assert_called()

  def testDo_NotBlessedIfModelUnavailable(self):
    # Prepare inputs and mocks.
    input_dict = self.input_dict.copy()
    input_dict.pop('examples')
    exec_properties = self.exec_properties.copy()
    exec_properties.pop('request_spec')
    self.model_server.WaitUntilModelAvailable.return_value = False

    # Run executor.
    infra_validator = executor.Executor(self.context)
    infra_validator.Do(input_dict, self.output_dict, exec_properties)

    # Check output artifact.
    self.assertFileExists(os.path.join(self.blessing.uri, 'INFRA_NOT_BLESSED'))
    self.assertEqual(0, self.blessing.get_int_custom_property('blessed'))

    # Check cleanup done.
    self.model_server.Stop.assert_called()

  def testDo_LoadAndRequest(self):
    # Prepare inputs and mocks.
    requests = [mock.Mock()]
    self.build_requests_mock.return_value = requests
    self.model_server.WaitUntilModelAvailable.return_value = True
    self.model_server.client.IssueRequests.return_value = True

    # Run executor.
    infra_validator = executor.Executor(self.context)
    infra_validator.Do(self.input_dict, self.output_dict, self.exec_properties)

    # Check output artifact.
    self.model_server.client.IssueRequests.assert_called_with(requests)
    self.assertFileExists(os.path.join(self.blessing.uri, 'INFRA_BLESSED'))
    self.assertEqual(1, self.blessing.get_int_custom_property('blessed'))

    # Check cleanup done.
    self.model_server.Stop.assert_called()

  def assertFileExists(self, path: Text):
    self.assertTrue(tf.io.gfile.exists(path))


if __name__ == '__main__':
  tf.test.main()
