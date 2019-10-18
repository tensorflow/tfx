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

import os
# Standard Imports
import mock
import tensorflow as tf

from tfx import version
from tfx.extensions.google_cloud_ai_platform import runner


class RunnerTest(tf.test.TestCase):

  def setUp(self):
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
    self._exec_properties = {
        'custom_config': {
            'gaip_training_args': self._training_inputs
        },
    }
    self._cmle_serving_args = {
        'model_name': 'model_name',
        'project_id': self._project_id,
    }

  @mock.patch(
      'tfx.extensions.google_cloud_ai_platform.runner.discovery'
  )
  def testStartCMLETraining(self, mock_discovery):
    mock_discovery.build.return_value = self._mock_api_client
    mock_create = mock.Mock()
    self._mock_api_client.projects().jobs().create = mock_create
    mock_get = mock.Mock()
    self._mock_api_client.projects().jobs().get.return_value = mock_get
    mock_get.execute.return_value = {
        'state': 'SUCCEEDED',
    }

    class_path = 'foo.bar.class'

    runner.start_cmle_training(self._inputs, self._outputs,
                               self._exec_properties, class_path,
                               self._training_inputs)

    mock_create.assert_called_with(
        body=mock.ANY, parent='projects/{}'.format(self._project_id))
    (_, kwargs) = mock_create.call_args
    body = kwargs['body']

    default_image = 'gcr.io/tfx-oss-public/tfx:%s' % (version.__version__)
    self.assertDictContainsSubset(
        {
            'masterConfig': {
                'imageUri': default_image,
            },
            'args': [
                '--executor_class_path', class_path, '--inputs', '{}',
                '--outputs', '{}', '--exec-properties', '{"custom_config": {}}'
            ],
        }, body['trainingInput'])
    self.assertStartsWith(body['jobId'], 'tfx_')
    mock_get.execute.assert_called_with()

  @mock.patch(
      'tfx.extensions.google_cloud_ai_platform.runner.discovery'
  )
  def testStartCMLETrainingWithUserContainer(self, mock_discovery):
    self._training_inputs['masterConfig'] = {'imageUri': 'my-custom-image'}
    mock_discovery.build.return_value = self._mock_api_client
    mock_create = mock.Mock()
    self._mock_api_client.projects().jobs().create = mock_create
    mock_get = mock.Mock()
    self._mock_api_client.projects().jobs().get.return_value = mock_get
    mock_get.execute.return_value = {
        'state': 'SUCCEEDED',
    }

    class_path = 'foo.bar.class'

    runner.start_cmle_training(self._inputs, self._outputs,
                               self._exec_properties, class_path,
                               self._training_inputs)

    mock_create.assert_called_with(
        body=mock.ANY, parent='projects/{}'.format(self._project_id))
    (_, kwargs) = mock_create.call_args
    body = kwargs['body']
    self.assertDictContainsSubset(
        {
            'masterConfig': {
                'imageUri': 'my-custom-image',
            },
            'args': [
                '--executor_class_path', class_path, '--inputs', '{}',
                '--outputs', '{}', '--exec-properties', '{"custom_config": {}}'
            ],
        }, body['trainingInput'])
    self.assertStartsWith(body['jobId'], 'tfx_')
    mock_get.execute.assert_called_with()

  @mock.patch(
      'tfx.extensions.google_cloud_ai_platform.runner.discovery'
  )
  def testDeployModelForCMLEServing(self, mock_discovery):
    serving_path = os.path.join(self._output_data_dir, 'serving_path')
    model_version = 'model_version'

    mock_discovery.build.return_value = self._mock_api_client
    mock_create = mock.Mock()
    mock_create.return_value.execute.return_value = {'name': 'op_name'}
    self._mock_api_client.projects().models().versions().create = mock_create
    mock_get = mock.Mock()
    self._mock_api_client.projects().operations().get = mock_get
    mock_set_default = mock.Mock()
    self._mock_api_client.projects().models().versions(
    ).setDefault = mock_set_default
    mock_set_default_execute = mock.Mock()
    self._mock_api_client.projects().models().versions(
    ).setDefault().execute = mock_set_default_execute

    mock_get.return_value.execute.return_value = {
        'done': 'Done',
        'response': {
            'name': model_version
        },
    }

    runner.deploy_model_for_cmle_serving(serving_path, model_version,
                                         self._cmle_serving_args)

    mock_create.assert_called_with(
        body=mock.ANY,
        parent='projects/{}/models/{}'.format(self._project_id, 'model_name'))
    (_, kwargs) = mock_create.call_args
    body = kwargs['body']
    self.assertDictEqual(
        {
            'name': 'v{}'.format(model_version),
            'deployment_uri': serving_path,
            'runtime_version': runner._get_tf_runtime_version(),
            'python_version': runner._get_caip_python_version(),
        }, body)
    mock_get.assert_called_with(name='op_name')

    mock_set_default.assert_called_with(
        name='projects/{}/models/{}/versions/{}'.format(
            self._project_id, 'model_name', model_version))
    mock_set_default_execute.assert_called_with()


if __name__ == '__main__':
  tf.test.main()
