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
"""Tests for tfx.extensions.google_cloud_ai_platform.runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import sys
from typing import Any, Dict, Text

# Standard Imports

import mock
import tensorflow as tf

from tfx.extensions.google_cloud_ai_platform import runner
from tfx.extensions.google_cloud_ai_platform.trainer import executor
from tfx.utils import json_utils
from tfx.utils import telemetry_utils
from tfx.utils import version_utils


class RunnerTest(tf.test.TestCase):

  def setUp(self):
    super(RunnerTest, self).setUp()
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._project_id = '12345'
    self._mock_api_client = mock.Mock()
    self._inputs = {}
    self._outputs = {}
    self._training_inputs = {
        'project': self._project_id,
    }
    self._job_id = 'my_jobid'
    # Dict format of exec_properties. custom_config needs to be serialized
    # before being passed into start_aip_training function.
    self._exec_properties = {
        'custom_config': {
            executor.TRAINING_ARGS_KEY: self._training_inputs,
        },
    }
    self._model_name = 'model_name'
    self._ai_platform_serving_args = {
        'model_name': self._model_name,
        'project_id': self._project_id,
    }
    self._executor_class_path = 'my.executor.Executor'
    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_EXECUTOR: self._executor_class_path}):
      self._job_labels = telemetry_utils.get_labels_dict()

  def _setUpTrainingMocks(self):
    self._mock_create = mock.Mock()
    self._mock_api_client.projects().jobs().create = self._mock_create
    self._mock_get = mock.Mock()
    self._mock_api_client.projects().jobs().get.return_value = self._mock_get
    self._mock_get.execute.return_value = {
        'state': 'SUCCEEDED',
    }

  def _serialize_custom_config_under_test(self) -> Dict[Text, Any]:
    """Converts self._exec_properties['custom_config'] to string."""
    result = copy.deepcopy(self._exec_properties)
    result['custom_config'] = json_utils.dumps(result['custom_config'])
    return result

  @mock.patch('tfx.extensions.google_cloud_ai_platform.runner.discovery')
  def testStartAIPTraining(self, mock_discovery):
    mock_discovery.build.return_value = self._mock_api_client
    self._setUpTrainingMocks()

    class_path = 'foo.bar.class'

    runner.start_aip_training(self._inputs, self._outputs,
                              self._serialize_custom_config_under_test(),
                              class_path,
                              self._training_inputs, None)

    self._mock_create.assert_called_with(
        body=mock.ANY, parent='projects/{}'.format(self._project_id))
    (_, kwargs) = self._mock_create.call_args
    body = kwargs['body']

    default_image = 'gcr.io/tfx-oss-public/tfx:{}'.format(
        version_utils.get_image_version())
    self.assertDictContainsSubset(
        {
            'masterConfig': {
                'imageUri':
                    default_image,
                'containerCommand':
                    runner._CONTAINER_COMMAND + [
                        '--executor_class_path', class_path, '--inputs', '{}',
                        '--outputs', '{}', '--exec-properties',
                        '{"custom_config": '
                        '"{\\"ai_platform_training_args\\": {\\"project\\": \\"12345\\"'
                        '}}"}'
                    ],
            },
        }, body['trainingInput'])
    self.assertStartsWith(body['jobId'], 'tfx_')
    self._mock_get.execute.assert_called_with()

  @mock.patch('tfx.extensions.google_cloud_ai_platform.runner.discovery')
  def testStartAIPTrainingWithUserContainer(self, mock_discovery):
    mock_discovery.build.return_value = self._mock_api_client
    self._setUpTrainingMocks()

    class_path = 'foo.bar.class'

    self._training_inputs['masterConfig'] = {'imageUri': 'my-custom-image'}
    self._exec_properties['custom_config'][executor.JOB_ID_KEY] = self._job_id
    runner.start_aip_training(self._inputs, self._outputs,
                              self._serialize_custom_config_under_test(),
                              class_path,
                              self._training_inputs, self._job_id)

    self._mock_create.assert_called_with(
        body=mock.ANY, parent='projects/{}'.format(self._project_id))
    (_, kwargs) = self._mock_create.call_args
    body = kwargs['body']
    self.assertDictContainsSubset(
        {
            'masterConfig': {
                'imageUri':
                    'my-custom-image',
                'containerCommand':
                    runner._CONTAINER_COMMAND + [
                        '--executor_class_path', class_path, '--inputs', '{}',
                        '--outputs', '{}', '--exec-properties',
                        '{"custom_config": '
                        '"{\\"ai_platform_training_args\\": '
                        '{\\"masterConfig\\": {\\"imageUri\\": \\"my-custom-image\\"}, '
                        '\\"project\\": \\"12345\\"}, '
                        '\\"ai_platform_training_job_id\\": \\"my_jobid\\"}"}'
                    ],
            }
        }, body['trainingInput'])
    self.assertEqual(body['jobId'], 'my_jobid')
    self._mock_get.execute.assert_called_with()

  def _setUpPredictionMocks(self):
    self._serving_path = os.path.join(self._output_data_dir, 'serving_path')
    self._model_version = 'model_version'

    self._mock_models_create = mock.Mock()
    self._mock_api_client.projects().models().create = self._mock_models_create

    self._mock_versions_create = mock.Mock()
    self._mock_versions_create.return_value.execute.return_value = {
        'name': 'versions_create_op_name'
    }

    self._mock_api_client.projects().models().versions(
    ).create = self._mock_versions_create
    self._mock_get = mock.Mock()
    self._mock_api_client.projects().operations().get = self._mock_get

    self._mock_set_default = mock.Mock()
    self._mock_api_client.projects().models().versions(
    ).setDefault = self._mock_set_default

    self._mock_set_default_execute = mock.Mock()
    self._mock_api_client.projects().models().versions().setDefault(
    ).execute = self._mock_set_default_execute

    self._mock_get.return_value.execute.return_value = {
        'done': True,
        'response': {
            'name': self._model_version,
        },
    }

  def _assertDeployModelMockCalls(self,
                                  expected_models_create_body=None,
                                  expected_versions_create_body=None,
                                  expect_set_default=True):
    if not expected_models_create_body:
      expected_models_create_body = {
          'name': self._model_name,
          'regions': [],
          'labels': self._job_labels
      }

    if not expected_versions_create_body:
      expected_versions_create_body = {
          'name':
              self._model_version,
          'deployment_uri':
              self._serving_path,
          'runtime_version':
              runner._get_tf_runtime_version(tf.__version__),
          'python_version':
              runner._get_caip_python_version(
                  runner._get_tf_runtime_version(tf.__version__)),
          'labels':
              self._job_labels
      }

    self._mock_models_create.assert_called_with(
        body=mock.ANY,
        parent='projects/{}'.format(self._project_id),
    )
    (_, models_create_kwargs) = self._mock_models_create.call_args
    self.assertDictEqual(expected_models_create_body,
                         models_create_kwargs['body'])

    self._mock_versions_create.assert_called_with(
        body=mock.ANY,
        parent='projects/{}/models/{}'.format(self._project_id,
                                              self._model_name))
    (_, versions_create_kwargs) = self._mock_versions_create.call_args

    self.assertDictEqual(expected_versions_create_body,
                         versions_create_kwargs['body'])

    if not expect_set_default:
      return

    self._mock_set_default.assert_called_with(
        name='projects/{}/models/{}/versions/{}'.format(
            self._project_id, self._model_name, self._model_version))
    self._mock_set_default_execute.assert_called_with()

  def testDeployModelForAIPPrediction(self):
    self._setUpPredictionMocks()

    runner.deploy_model_for_aip_prediction(self._mock_api_client,
                                           self._serving_path,
                                           self._model_version,
                                           self._ai_platform_serving_args,
                                           self._job_labels)

    expected_models_create_body = {
        'name': self._model_name,
        'regions': [],
        'labels': self._job_labels
    }
    self._assertDeployModelMockCalls(
        expected_models_create_body=expected_models_create_body)

  def testDeployModelForAIPPredictionError(self):
    self._setUpPredictionMocks()

    self._mock_get.return_value.execute.return_value = {
        'done': True,
        'error': {
            'code': 999,
            'message': 'it was an error.'
        },
    }

    with self.assertRaises(RuntimeError):
      runner.deploy_model_for_aip_prediction(self._mock_api_client,
                                             self._serving_path,
                                             self._model_version,
                                             self._ai_platform_serving_args,
                                             self._job_labels)

    expected_models_create_body = {
        'name': self._model_name,
        'regions': [],
        'labels': self._job_labels
    }
    self._assertDeployModelMockCalls(
        expected_models_create_body=expected_models_create_body,
        expect_set_default=False)

  def testDeployModelForAIPPredictionWithCustomRegion(self):
    self._setUpPredictionMocks()

    self._ai_platform_serving_args['regions'] = ['custom-region']
    runner.deploy_model_for_aip_prediction(self._mock_api_client,
                                           self._serving_path,
                                           self._model_version,
                                           self._ai_platform_serving_args,
                                           self._job_labels)

    expected_models_create_body = {
        'name': self._model_name,
        'regions': ['custom-region'],
        'labels': self._job_labels
    }
    self._assertDeployModelMockCalls(
        expected_models_create_body=expected_models_create_body)

  def testDeployModelForAIPPredictionWithCustomRuntime(self):
    self._setUpPredictionMocks()

    self._ai_platform_serving_args['runtime_version'] = '1.23.45'
    runner.deploy_model_for_aip_prediction(self._mock_api_client,
                                           self._serving_path,
                                           self._model_version,
                                           self._ai_platform_serving_args,
                                           self._job_labels)

    expected_versions_create_body = {
        'name': self._model_version,
        'deployment_uri': self._serving_path,
        'runtime_version': '1.23.45',
        'python_version': runner._get_caip_python_version('1.23.45'),
        'labels': self._job_labels,
    }
    self._assertDeployModelMockCalls(
        expected_versions_create_body=expected_versions_create_body)

  def testDeployModelForAIPPredictionWithCustomMachineType(self):
    self._setUpPredictionMocks()

    self._ai_platform_serving_args['machine_type'] = 'custom_machine_type'
    runner.deploy_model_for_aip_prediction(self._mock_api_client,
                                           self._serving_path,
                                           self._model_version,
                                           self._ai_platform_serving_args,
                                           self._job_labels)

    expected_versions_create_body = {
        'name':
            self._model_version,
        'deployment_uri':
            self._serving_path,
        'machine_type':
            'custom_machine_type',
        'runtime_version':
            runner._get_tf_runtime_version(tf.__version__),
        'python_version':
            runner._get_caip_python_version(
                runner._get_tf_runtime_version(tf.__version__)),
        'labels':
            self._job_labels,
    }
    self._assertDeployModelMockCalls(
        expected_versions_create_body=expected_versions_create_body)

  def testGetTensorflowRuntime(self):
    self.assertEqual('1.14', runner._get_tf_runtime_version('1.14'))
    self.assertEqual('1.15', runner._get_tf_runtime_version('1.15.0'))
    self.assertEqual('1.15', runner._get_tf_runtime_version('1.15.1'))
    self.assertEqual('1.15', runner._get_tf_runtime_version('2.0.0'))
    self.assertEqual('1.15', runner._get_tf_runtime_version('2.0.1'))
    self.assertEqual('2.1', runner._get_tf_runtime_version('2.1.0'))
    self.assertEqual('2.2', runner._get_tf_runtime_version('2.2.0'))
    self.assertEqual('2.2', runner._get_tf_runtime_version('2.3.0'))

  def testGetAiPlatformTrainingPythonVersion(self):
    if sys.version_info.major == 2:
      self.assertEqual('2.7', runner._get_caip_python_version('1.14'))
      self.assertEqual('2.7', runner._get_caip_python_version('1.15'))
    else:  # 3.x
      self.assertEqual('3.5', runner._get_caip_python_version('1.14'))
      self.assertEqual('3.7', runner._get_caip_python_version('1.15'))
      self.assertEqual('3.7', runner._get_caip_python_version('2.1'))

  def _setUpDeleteModelVersionMocks(self):
    self._model_version = 'model_version'

    self._mock_models_version_delete = mock.Mock()
    self._mock_api_client.projects().models().versions().delete = (
        self._mock_models_version_delete)
    self._mock_models_version_delete.return_value.execute.return_value = {
        'name': 'version_delete_op_name'
    }
    self._mock_get = mock.Mock()
    self._mock_api_client.projects().operations().get = self._mock_get

    self._mock_get.return_value.execute.return_value = {
        'done': True,
    }

  def _assertDeleteModelVersionMockCalls(self):
    self._mock_models_version_delete.assert_called_with(
        name='projects/{}/models/{}/versions/{}'.format(self._project_id,
                                                        self._model_name,
                                                        self._model_version),)
    (_, model_version_delete_kwargs) = (
        self._mock_models_version_delete.call_args)
    self.assertNotIn('body', model_version_delete_kwargs)

  @mock.patch('tfx.extensions.google_cloud_ai_platform.runner.discovery')
  def testDeleteModelVersionForAIPPrediction(self, mock_discovery):
    self._setUpDeleteModelVersionMocks()

    runner.delete_model_version_from_aip_if_exists(
        self._mock_api_client, self._model_version,
        self._ai_platform_serving_args)

    self._assertDeleteModelVersionMockCalls()

  def _setUpDeleteModelMocks(self):
    self._mock_models_delete = mock.Mock()
    self._mock_api_client.projects().models().delete = (
        self._mock_models_delete)
    self._mock_models_delete.return_value.execute.return_value = {
        'name': 'model_delete_op_name'
    }
    self._mock_get = mock.Mock()
    self._mock_api_client.projects().operations().get = self._mock_get
    self._mock_get.return_value.execute.return_value = {
        'done': True,
    }

  def _assertDeleteModelMockCalls(self):
    self._mock_models_delete.assert_called_with(
        name='projects/{}/models/{}'.format(self._project_id,
                                            self._model_name),)
    (_, model_delete_kwargs) = self._mock_models_delete.call_args
    self.assertNotIn('body', model_delete_kwargs)

  @mock.patch('tfx.extensions.google_cloud_ai_platform.runner.discovery')
  def testDeleteModelForAIPPrediction(self, mock_discovery):
    self._setUpDeleteModelMocks()

    runner.delete_model_from_aip_if_exists(self._mock_api_client,
                                           self._ai_platform_serving_args)

    self._assertDeleteModelMockCalls()


if __name__ == '__main__':
  tf.test.main()
