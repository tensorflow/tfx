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

from tfx.components.infra_validator import executor
from tfx.components.infra_validator.model_server_runners import local_docker_runner
from tfx.types import standard_artifacts

LocalDockerModelServerRunner = local_docker_runner.LocalDockerModelServerRunner


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()

    self._source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'testdata')
    base_output_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR',
                                     self.get_temp_dir())
    output_data_dir = os.path.join(base_output_dir, self._testMethodName)

    # Setup input_dict.
    model = standard_artifacts.Model()
    model.uri = os.path.join(self._source_data_dir, 'trainer', 'current')
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(self._source_data_dir,
                                'transform',
                                'transformed_examples',
                                'eval')
    self._input_dict = {
        'model': [model],
        'examples': [examples],
    }

    # Setup output_dict.
    self._blessing = standard_artifacts.InfraBlessing()
    self._blessing.uri = os.path.join(output_data_dir, 'blessing')
    self._output_dict = {
        'blessing': [self._blessing]
    }

    self._temp_dir = os.path.join(output_data_dir, '.temp')
    self._context = executor.Executor.Context(tmp_dir=self._temp_dir,
                                              unique_id='1')

  @mock.patch.object(LocalDockerModelServerRunner, 'WaitUntilModelAvailable')
  @mock.patch.object(LocalDockerModelServerRunner, 'Stop')
  @mock.patch.object(LocalDockerModelServerRunner, 'Start')
  def testDoWithoutTestExamples(self, mock_start, mock_stop,
                                mock_wait_until_model_available):
    # Prepare inputs and mocks.
    input_dict = self._input_dict.copy()
    input_dict.pop('examples')
    exec_properties = {
        'serving_spec': json.dumps({
            'tensorflow_serving': {
                'tags': ['1.15.0']
            },
            'local_docker': {}
        }),
        'validation_spec': json.dumps({
            'max_loading_time_seconds': 10
        })
    }
    mock_wait_until_model_available.return_value = True

    # Run executor.
    infra_validator = executor.Executor(self._context)
    infra_validator.Do(input_dict, self._output_dict, exec_properties)

    # Check output artifact.
    self.assertTrue(tf.io.gfile.exists(
        os.path.join(self._blessing.uri, 'BLESSED')))
    self.assertEqual(1, self._blessing.get_int_custom_property('blessed'))


if __name__ == '__main__':
  tf.test.main()
