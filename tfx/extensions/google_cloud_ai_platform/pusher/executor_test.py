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

import copy
import os
from typing import Any, Dict
import unittest
from unittest import mock

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.extensions.google_cloud_ai_platform.pusher import executor
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import json_utils
from tfx.utils import telemetry_utils


# TODO(b/163417407): remove when we can safely depend on
def _stale_googleapiclient_version_():
  import googleapiclient  # pylint: disable=g-import-not-at-top
  try:
    return googleapiclient.__version__ < '1.8'
  except AttributeError:
    # Later version of googleapiclient has no __version__
    return False


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
    fileio.makedirs(self._output_data_dir)
    self._model_export = standard_artifacts.Model()
    self._model_export.uri = os.path.join(self._source_data_dir,
                                          'trainer/current')
    self._model_blessing = standard_artifacts.ModelBlessing()
    self._input_dict = {
        standard_component_specs.MODEL_KEY: [self._model_export],
        standard_component_specs.MODEL_BLESSING_KEY: [self._model_blessing],
    }

    self._model_push = standard_artifacts.PushedModel()
    self._model_push.uri = os.path.join(self._output_data_dir, 'model_push')
    fileio.makedirs(self._model_push.uri)
    self._output_dict = {
        standard_component_specs.PUSHED_MODEL_KEY: [self._model_push],
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

  def _serialize_custom_config_under_test(self) -> Dict[str, Any]:
    """Converts self._exec_properties['custom_config'] to string."""
    result = copy.deepcopy(self._exec_properties)
    result['custom_config'] = json_utils.dumps(result['custom_config'])
    return result

  def assertDirectoryEmpty(self, path):
    self.assertEqual(len(fileio.listdir(path)), 0)

  def assertDirectoryNotEmpty(self, path):
    self.assertGreater(len(fileio.listdir(path)), 0)

  def assertPushed(self):
    self.assertDirectoryNotEmpty(self._model_push.uri)
    self.assertEqual(1, self._model_push.get_int_custom_property('pushed'))

  def assertNotPushed(self):
    self.assertDirectoryEmpty(self._model_push.uri)
    self.assertEqual(0, self._model_push.get_int_custom_property('pushed'))

  @mock.patch(
      'tfx.extensions.google_cloud_ai_platform.pusher.executor.discovery')
  @mock.patch.object(executor, 'runner', autospec=True)
  def testDoBlessed(self, mock_runner, _):
    self._model_blessing.uri = os.path.join(self._source_data_dir,
                                            'model_validator/blessed')
    self._model_blessing.set_int_custom_property('blessed', 1)
    mock_runner.get_service_name_and_api_version.return_value = ('ml', 'v1')
    self._executor.Do(self._input_dict, self._output_dict,
                      self._serialize_custom_config_under_test())
    executor_class_path = '%s.%s' % (self._executor.__class__.__module__,
                                     self._executor.__class__.__name__)
    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_EXECUTOR: executor_class_path}):
      job_labels = telemetry_utils.make_labels_dict()
    mock_runner.deploy_model_for_aip_prediction.assert_called_once_with(
        mock.ANY,
        self._model_push.uri,
        mock.ANY,
        mock.ANY,
        job_labels,
    )
    self.assertPushed()
    version = self._model_push.get_string_custom_property('pushed_version')
    self.assertEqual(
        self._model_push.get_string_custom_property('pushed_destination'),
        'projects/project_id/models/model_name/versions/{}'.format(version))

  @mock.patch(
      'tfx.extensions.google_cloud_ai_platform.pusher.executor.discovery')
  @mock.patch.object(executor, 'runner', autospec=True)
  def testDoNotBlessed(self, mock_runner, _):
    self._model_blessing.uri = os.path.join(self._source_data_dir,
                                            'model_validator/not_blessed')
    self._model_blessing.set_int_custom_property('blessed', 0)
    mock_runner.get_service_name_and_api_version.return_value = ('ml', 'v1')
    self._executor.Do(self._input_dict, self._output_dict,
                      self._serialize_custom_config_under_test())
    self.assertNotPushed()
    mock_runner.deploy_model_for_aip_prediction.assert_not_called()

  def testRegionsAndEndpointCannotCoExist(self):
    # Dict format of exec_properties. custom_config needs to be serialized
    # before being passed into Do function.
    self._exec_properties = {
        'custom_config': {
            executor.SERVING_ARGS_KEY: {
                'model_name': 'model_name',
                'project_id': 'project_id',
                'regions': ['us-central1'],
            },
            executor.ENDPOINT_ARGS_KEY: 'https://ml-us-west1.googleapis.com',
        },
        'push_destination': None,
    }
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        '\'endpoint\' and \'ai_platform_serving_args.regions\' cannot be set simultanuously'
    ):
      self._executor.Do(self._input_dict, self._output_dict,
                        self._serialize_custom_config_under_test())

  # TODO(b/163417407): template and run this test when we can safely depend on
  # googleapiclient>=1.8.
  @unittest.skipIf(_stale_googleapiclient_version_(),
                   'googleapiclient is too stale')
  @mock.patch(
      'tfx.extensions.google_cloud_ai_platform.pusher.executor.discovery')
  @mock.patch.object(executor, 'runner', autospec=True)
  def testDoBlessedOnRegionalEndpoint(self, mock_runner, _):
    self._exec_properties = {
        'custom_config': {
            executor.SERVING_ARGS_KEY: {
                'model_name': 'model_name',
                'project_id': 'project_id'
            },
            executor.ENDPOINT_ARGS_KEY: 'https://ml-us-west1.googleapis.com',
        },
    }
    self._model_blessing.uri = os.path.join(self._source_data_dir,
                                            'model_validator/blessed')
    self._model_blessing.set_int_custom_property('blessed', 1)
    mock_runner.get_service_name_and_api_version.return_value = ('ml', 'v1')
    self._executor.Do(self._input_dict, self._output_dict,
                      self._serialize_custom_config_under_test())
    executor_class_path = '%s.%s' % (self._executor.__class__.__module__,
                                     self._executor.__class__.__name__)
    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_EXECUTOR: executor_class_path}):
      job_labels = telemetry_utils.make_labels_dict()
    mock_runner.deploy_model_for_aip_prediction.assert_called_once_with(
        mock.ANY,
        self._model_push.uri,
        mock.ANY,
        mock.ANY,
        job_labels,
    )
    self.assertPushed()
    version = self._model_push.get_string_custom_property('pushed_version')
    self.assertEqual(
        self._model_push.get_string_custom_property('pushed_destination'),
        'projects/project_id/models/model_name/versions/{}'.format(version))


if __name__ == '__main__':
  tf.test.main()
