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

import copy
import os
from typing import Any, Dict, Text
# Standard Imports
import mock
import tensorflow as tf

from tfx.components.pusher import executor as tfx_pusher_executor
from tfx.extensions.google_cloud_ai_platform.pusher import executor
from tfx.types import standard_artifacts
from tfx.utils import json_utils


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
                                          'trainer/current')
    self._model_blessing = standard_artifacts.ModelBlessing()
    self._input_dict = {
        tfx_pusher_executor.MODEL_KEY: [self._model_export],
        tfx_pusher_executor.MODEL_BLESSING_KEY: [self._model_blessing],
    }

    self._model_push = standard_artifacts.PushedModel()
    self._model_push.uri = os.path.join(self._output_data_dir, 'model_push')
    tf.io.gfile.makedirs(self._model_push.uri)
    self._output_dict = {
        tfx_pusher_executor.PUSHED_MODEL_KEY: [self._model_push],
    }
    # Dict format of exec_properties. custom_config needs to be serialized
    # before being passed into Do function.
    self._exec_properties = {
        'custom_config': {
            executor.SERVING_ARGS_KEY: {
                'model_name': 'model_name',
                'project_id': 'project_id'
            },
        },
        'push_destination': None,
    }
    self._executor = executor.Executor()

  def _serialize_custom_config_under_test(self) -> Dict[Text, Any]:
    """Converts self._exec_properties['custom_config'] to string."""
    result = copy.deepcopy(self._exec_properties)
    result['custom_config'] = json_utils.dumps(result['custom_config'])
    return result

  def assertDirectoryEmpty(self, path):
    self.assertEqual(len(tf.io.gfile.listdir(path)), 0)

  def assertDirectoryNotEmpty(self, path):
    self.assertGreater(len(tf.io.gfile.listdir(path)), 0)

  def assertPushed(self):
    self.assertDirectoryNotEmpty(self._model_push.uri)
    self.assertEqual(1, self._model_push.get_int_custom_property('pushed'))

  def assertNotPushed(self):
    self.assertDirectoryEmpty(self._model_push.uri)
    self.assertEqual(0, self._model_push.get_int_custom_property('pushed'))

  @mock.patch.object(executor, 'runner', autospec=True)
  def testDoBlessed(self, mock_runner):
    self._model_blessing.uri = os.path.join(self._source_data_dir,
                                            'model_validator/blessed')
    self._model_blessing.set_int_custom_property('blessed', 1)
    self._executor.Do(self._input_dict, self._output_dict,
                      self._serialize_custom_config_under_test())
    executor_class_path = '%s.%s' % (self._executor.__class__.__module__,
                                     self._executor.__class__.__name__)
    mock_runner.deploy_model_for_aip_prediction.assert_called_once_with(
        self._model_push.uri,
        mock.ANY,
        mock.ANY,
        executor_class_path,
    )
    self.assertPushed()
    version = self._model_push.get_string_custom_property('pushed_version')
    self.assertEqual(
        self._model_push.get_string_custom_property('pushed_destination'),
        'projects/project_id/models/model_name/versions/{}'.format(version))

  @mock.patch.object(executor, 'runner', autospec=True)
  def testDoNotBlessed(self, mock_runner):
    self._model_blessing.uri = os.path.join(self._source_data_dir,
                                            'model_validator/not_blessed')
    self._model_blessing.set_int_custom_property('blessed', 0)
    self._executor.Do(self._input_dict, self._output_dict,
                      self._serialize_custom_config_under_test())
    self.assertNotPushed()
    mock_runner.deploy_model_for_aip_prediction.assert_not_called()


if __name__ == '__main__':
  tf.test.main()
