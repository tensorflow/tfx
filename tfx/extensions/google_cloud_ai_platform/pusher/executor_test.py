# Lint as: python2, python3
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
"""Tests for tfx.extensions.google_cloud_ai_platform.pusher.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Standard Imports
import mock
import tensorflow as tf

from tfx.extensions.google_cloud_ai_platform.pusher import executor
from tfx.extensions.google_cloud_ai_platform.pusher.executor import Executor
from tfx.types import standard_artifacts


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self._source_data_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        'components', 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    tf.io.gfile.makedirs(self._output_data_dir)
    self._model_export = standard_artifacts.Model()
    self._model_export.uri = os.path.join(self._source_data_dir,
                                          'trainer/current/')
    self._model_blessing = standard_artifacts.ModelBlessing()
    self._input_dict = {
        'model_export': [self._model_export],
        'model_blessing': [self._model_blessing],
    }

    self._model_push = standard_artifacts.PushedModel()
    self._model_push.uri = os.path.join(self._output_data_dir, 'model_push')
    tf.io.gfile.makedirs(self._model_push.uri)
    self._output_dict = {
        'model_push': [self._model_push],
    }
    self._exec_properties = {
        'custom_config': {
            'ai_platform_serving_args': {
                'model_name': 'model_name',
                'project_id': 'project_id'
            },
        },
        'push_destination': None,
    }
    self._executor = Executor()

  @mock.patch.object(executor, 'runner', autospec=True)
  def testDoBlessed(self, mock_runner):
    self._model_blessing.uri = os.path.join(self._source_data_dir,
                                            'model_validator/blessed/')
    self._model_blessing.set_int_custom_property('blessed', 1)
    self._executor.Do(self._input_dict, self._output_dict,
                      self._exec_properties)
    mock_runner.deploy_model_for_cmle_serving.assert_called_once_with(
        self._model_push.artifact.custom_properties['pushed_model']
        .string_value, mock.ANY, mock.ANY)
    self.assertEqual(
        1, self._model_push.artifact.custom_properties['pushed'].int_value)

  @mock.patch.object(executor, 'runner', autospec=True)
  def testDoNotBlessed(self, mock_runner):
    self._model_blessing.uri = os.path.join(self._source_data_dir,
                                            'model_validator/not_blessed/')
    self._model_blessing.set_int_custom_property('blessed', 0)
    self._executor.Do(self._input_dict, self._output_dict,
                      self._exec_properties)
    self.assertEqual(
        0, self._model_push.artifact.custom_properties['pushed'].int_value)
    mock_runner.deploy_model_for_cmle_serving.assert_not_called()


if __name__ == '__main__':
  tf.test.main()
