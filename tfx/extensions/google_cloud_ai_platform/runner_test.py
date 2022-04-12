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

import copy
import importlib
import os
from typing import Any, Dict
from unittest import mock

from google.auth import credentials as auth_credentials
from google.cloud import aiplatform
from google.cloud.aiplatform import initializer
from google.cloud.aiplatform.compat.types import endpoint
from google.cloud.aiplatform_v1.services.endpoint_service import (
    client as endpoint_service_client)
from google.cloud.aiplatform_v1.types.custom_job import CustomJob
from google.cloud.aiplatform_v1.types.job_state import JobState
from googleapiclient import errors
import httplib2
import tensorflow as tf
from tfx.extensions.google_cloud_ai_platform import prediction_clients
from tfx.extensions.google_cloud_ai_platform import runner
from tfx.extensions.google_cloud_ai_platform.trainer import executor
from tfx.utils import json_utils
from tfx.utils import telemetry_utils
from tfx.utils import version_utils


class RunnerTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
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
    # before being passed into start_cloud_training function.
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
      self._job_labels = telemetry_utils.make_labels_dict()

  def _setUpTrainingMocks(self):
    self._mock_create_request = mock.Mock()
    self._mock_create = mock.Mock()
    self._mock_create.return_value = self._mock_create_request
    self._mock_api_client.projects().jobs().create = self._mock_create
    self._mock_get = mock.Mock()
    self._mock_api_client.projects().jobs().get.return_value = self._mock_get
    self._mock_get.execute.return_value = {
        'state': 'SUCCEEDED',
    }

  def _setUpVertexTrainingMocks(self):
    self._mock_create = mock.Mock()
    self._mock_api_client.create_custom_job = self._mock_create
    self._mock_create.return_value = CustomJob(name='vertex_job_study_id')
    self._mock_get = mock.Mock()
    self._mock_api_client.get_custom_job = self._mock_get
    self._mock_get.return_value = CustomJob(state=JobState.JOB_STATE_SUCCEEDED)

  def _serialize_custom_config_under_test(self) -> Dict[str, Any]:
    """Converts self._exec_properties['custom_config'] to string."""
    result = copy.deepcopy(self._exec_properties)
    result['custom_config'] = json_utils.dumps(result['custom_config'])
    return result

  @mock.patch(
      'tfx.extensions.google_cloud_ai_platform.training_clients.discovery')
  def testStartCloudTraining(self, mock_discovery):
    mock_discovery.build.return_value = self._mock_api_client
    self._setUpTrainingMocks()

    class_path = 'foo.bar.class'

    runner.start_cloud_training(self._inputs, self._outputs,
                                self._serialize_custom_config_under_test(),
                                class_path, self._training_inputs, None)

    self._mock_create.assert_called_with(
        body=mock.ANY, parent='projects/{}'.format(self._project_id))
    kwargs = self._mock_create.call_args[1]
    body = kwargs['body']

    default_image = 'gcr.io/tfx-oss-public/tfx:{}'.format(
        version_utils.get_image_version())
    self.assertLessEqual({
        'masterConfig': {
            'imageUri':
                default_image,
            'containerCommand':
                runner._CONTAINER_COMMAND + [
                    '--executor_class_path', class_path, '--inputs', '{}',
                    '--outputs', '{}', '--exec-properties',
                    ('{"custom_config": '
                     '"{\\"ai_platform_training_args\\": {\\"project\\": \\"12345\\"'
                     '}}"}')
                ],
        },
    }.items(), body['training_input'].items())
    self.assertNotIn('project', body['training_input'])
    self.assertStartsWith(body['job_id'], 'tfx_')
    self._mock_get.execute.assert_called_with()
    self._mock_create_request.execute.assert_called_with()

  @mock.patch(
      'tfx.extensions.google_cloud_ai_platform.training_clients.discovery')
  def testStartCloudTrainingWithUserContainer(self, mock_discovery):
    mock_discovery.build.return_value = self._mock_api_client
    self._setUpTrainingMocks()

    class_path = 'foo.bar.class'

    self._training_inputs['masterConfig'] = {'imageUri': 'my-custom-image'}
    self._exec_properties['custom_config'][executor.JOB_ID_KEY] = self._job_id
    runner.start_cloud_training(self._inputs, self._outputs,
                                self._serialize_custom_config_under_test(),
                                class_path, self._training_inputs, self._job_id)

    self._mock_create.assert_called_with(
        body=mock.ANY, parent='projects/{}'.format(self._project_id))
    kwargs = self._mock_create.call_args[1]
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
                        ('{"custom_config": '
                         '"{\\"ai_platform_training_args\\": '
                         '{\\"masterConfig\\": {\\"imageUri\\": \\"my-custom-image\\"}, '
                         '\\"project\\": \\"12345\\"}, '
                         '\\"ai_platform_training_job_id\\": \\"my_jobid\\"}"}')
                    ],
            }
        }, body['training_input'])
    self.assertEqual(body['job_id'], 'my_jobid')
    self._mock_get.execute.assert_called_with()
    self._mock_create_request.execute.assert_called_with()

  @mock.patch('tfx.extensions.google_cloud_ai_platform.training_clients.gapic')
  def testStartCloudTraining_Vertex(self, mock_gapic):
    mock_gapic.JobServiceClient.return_value = self._mock_api_client
    self._setUpVertexTrainingMocks()

    class_path = 'foo.bar.class'
    region = 'us-central1'

    runner.start_cloud_training(self._inputs, self._outputs,
                                self._serialize_custom_config_under_test(),
                                class_path, self._training_inputs, None, {},
                                True, region)

    self._mock_create.assert_called_with(
        parent='projects/{}/locations/{}'.format(self._project_id, region),
        custom_job=mock.ANY)
    kwargs = self._mock_create.call_args[1]
    body = kwargs['custom_job']

    default_image = 'gcr.io/tfx-oss-public/tfx:{}'.format(
        version_utils.get_image_version())
    self.assertDictContainsSubset(
        {
            'worker_pool_specs': [{
                'container_spec': {
                    'image_uri':
                        default_image,
                    'command':
                        runner._CONTAINER_COMMAND + [
                            '--executor_class_path', class_path, '--inputs',
                            '{}', '--outputs', '{}', '--exec-properties',
                            ('{"custom_config": '
                             '"{\\"ai_platform_training_args\\": '
                             '{\\"project\\": \\"12345\\"'
                             '}}"}')
                        ],
                },
            },],
        }, body['job_spec'])
    self.assertStartsWith(body['display_name'], 'tfx_')
    self._mock_get.assert_called_with(name='vertex_job_study_id')

  @mock.patch('tfx.extensions.google_cloud_ai_platform.training_clients.gapic')
  def testStartCloudTrainingWithUserContainer_Vertex(self, mock_gapic):
    mock_gapic.JobServiceClient.return_value = self._mock_api_client
    self._setUpVertexTrainingMocks()

    class_path = 'foo.bar.class'

    self._training_inputs['worker_pool_specs'] = [{
        'container_spec': {
            'image_uri': 'my-custom-image'
        }
    }]
    self._exec_properties['custom_config'][executor.JOB_ID_KEY] = self._job_id
    region = 'us-central2'
    runner.start_cloud_training(self._inputs, self._outputs,
                                self._serialize_custom_config_under_test(),
                                class_path, self._training_inputs, self._job_id,
                                {}, True, region)

    self._mock_create.assert_called_with(
        parent='projects/{}/locations/{}'.format(self._project_id, region),
        custom_job=mock.ANY)
    kwargs = self._mock_create.call_args[1]
    body = kwargs['custom_job']
    self.assertLessEqual({
        'worker_pool_specs': [{
            'container_spec': {
                'image_uri':
                    'my-custom-image',
                'command':
                    runner._CONTAINER_COMMAND + [
                        '--executor_class_path', class_path, '--inputs', '{}',
                        '--outputs', '{}', '--exec-properties',
                        ('{"custom_config": '
                         '"{\\"ai_platform_training_args\\": '
                         '{\\"project\\": \\"12345\\", '
                         '\\"worker_pool_specs\\": '
                         '[{\\"container_spec\\": '
                         '{\\"image_uri\\": \\"my-custom-image\\"}}]}, '
                         '\\"ai_platform_training_job_id\\": '
                         '\\"my_jobid\\"}"}')
                    ],
            },
        },],
    }.items(), body['job_spec'].items())
    self.assertEqual(body['display_name'], 'my_jobid')
    self._mock_get.assert_called_with(name='vertex_job_study_id')

  @mock.patch('tfx.extensions.google_cloud_ai_platform.training_clients.gapic')
  def testStartCloudTrainingWithVertexCustomJob(self, mock_gapic):
    mock_gapic.JobServiceClient.return_value = self._mock_api_client
    self._setUpVertexTrainingMocks()

    class_path = 'foo.bar.class'
    expected_encryption_spec = {
        'kms_key_name': 'my_kmskey',
    }
    user_provided_labels = {
        'l1': 'v1',
        'l2': 'v2',
    }

    self._training_inputs['display_name'] = 'valid_name'
    self._training_inputs['job_spec'] = {
        'worker_pool_specs': [{
            'container_spec': {
                'image_uri': 'my-custom-image'
            }
        }]
    }
    self._training_inputs['labels'] = user_provided_labels
    self._training_inputs['encryption_spec'] = expected_encryption_spec
    self._exec_properties['custom_config'][executor.JOB_ID_KEY] = self._job_id
    region = 'us-central2'
    runner.start_cloud_training(self._inputs, self._outputs,
                                self._serialize_custom_config_under_test(),
                                class_path, self._training_inputs, self._job_id,
                                {}, True, region)

    self._mock_create.assert_called_with(
        parent='projects/{}/locations/{}'.format(self._project_id, region),
        custom_job=mock.ANY)
    kwargs = self._mock_create.call_args[1]
    body = kwargs['custom_job']
    self.assertDictContainsSubset(
        {
            'worker_pool_specs': [{
                'container_spec': {
                    'image_uri':
                        'my-custom-image',
                    'command':
                        runner._CONTAINER_COMMAND + [
                            '--executor_class_path', class_path, '--inputs',
                            '{}', '--outputs', '{}', '--exec-properties',
                            ('{"custom_config": '
                             '"{\\"ai_platform_training_args\\": '
                             '{\\"display_name\\": \\"valid_name\\", '
                             '\\"encryption_spec\\": {\\"kms_key_name\\": '
                             '\\"my_kmskey\\"}, \\"job_spec\\": '
                             '{\\"worker_pool_specs\\": '
                             '[{\\"container_spec\\": '
                             '{\\"image_uri\\": \\"my-custom-image\\"}}]}, '
                             '\\"labels\\": {\\"l1\\": \\"v1\\", '
                             '\\"l2\\": \\"v2\\"}, '
                             '\\"project\\": \\"12345\\"}, '
                             '\\"ai_platform_training_job_id\\": '
                             '\\"my_jobid\\"}"}')
                        ],
                },
            },],
        }, body['job_spec'])
    self.assertEqual(body['display_name'], 'valid_name')
    self.assertDictEqual(body['encryption_spec'], expected_encryption_spec)
    self.assertLessEqual(user_provided_labels.items(), body['labels'].items())
    self._mock_get.assert_called_with(name='vertex_job_study_id')

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

  def _setUpVertexPredictionMocks(self):
    importlib.reload(initializer)
    importlib.reload(aiplatform)

    self._serving_container_image_uri = 'gcr.io/path/to/container'
    self._serving_path = os.path.join(self._output_data_dir, 'serving_path')
    self._endpoint_name = 'endpoint-name'
    self._endpoint_region = 'us-central1'
    self._deployed_model_id = 'model_id'

    self._mock_create_client = mock.Mock()
    initializer.global_config.create_client = self._mock_create_client
    self._mock_create_client.return_value = mock.Mock(
        spec=endpoint_service_client.EndpointServiceClient)

    self._mock_get_endpoint = mock.Mock()
    endpoint_service_client.EndpointServiceClient.get_endpoint = self._mock_get_endpoint
    self._mock_get_endpoint.return_value = endpoint.Endpoint(
        display_name=self._endpoint_name,)

    aiplatform.init(
        project=self._project_id,
        location=None,
        credentials=mock.Mock(spec=auth_credentials.AnonymousCredentials()))

    self._mock_endpoint = aiplatform.Endpoint(
        endpoint_name='projects/{}/locations/us-central1/endpoints/1234'.format(
            self._project_id))

    self._mock_endpoint_create = mock.Mock()
    aiplatform.Endpoint.create = self._mock_endpoint_create
    self._mock_endpoint_create.return_value = self._mock_endpoint

    self._mock_endpoint_list = mock.Mock()
    aiplatform.Endpoint.list = self._mock_endpoint_list
    self._mock_endpoint_list.return_value = []

    self._mock_model_upload = mock.Mock()
    aiplatform.Model.upload = self._mock_model_upload

    self._mock_model_deploy = mock.Mock()
    self._mock_model_upload.return_value.deploy = self._mock_model_deploy

    self._ai_platform_serving_args_vertex = {
        'endpoint_name': self._endpoint_name,
        'project_id': self._project_id,
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
              prediction_clients._get_tf_runtime_version(tf.__version__),
          'python_version':
              '3.7',
          'labels':
              self._job_labels
      }

    self._mock_models_create.assert_called_with(
        body=mock.ANY,
        parent='projects/{}'.format(self._project_id),
    )
    models_create_kwargs = self._mock_models_create.call_args[1]
    self.assertDictEqual(expected_models_create_body,
                         models_create_kwargs['body'])

    self._mock_versions_create.assert_called_with(
        body=mock.ANY,
        parent='projects/{}/models/{}'.format(self._project_id,
                                              self._model_name))
    versions_create_kwargs = self._mock_versions_create.call_args[1]

    self.assertDictEqual(expected_versions_create_body,
                         versions_create_kwargs['body'])

    if not expect_set_default:
      return

    self._mock_set_default.assert_called_with(
        name='projects/{}/models/{}/versions/{}'.format(
            self._project_id, self._model_name, self._model_version))
    self._mock_set_default_execute.assert_called_with()

  def _assertDeployModelMockCallsVertex(self,
                                        expected_endpoint_create_body=None,
                                        expected_model_upload_body=None,
                                        expected_model_deploy_body=None):
    if not expected_endpoint_create_body:
      expected_endpoint_create_body = {
          'display_name': self._endpoint_name,
          'labels': self._job_labels,
      }

    if not expected_model_upload_body:
      expected_model_upload_body = {
          'display_name': self._model_name,
          'artifact_uri': self._serving_path,
          'serving_container_image_uri': self._serving_container_image_uri,
      }

    if not expected_model_deploy_body:
      expected_model_deploy_body = {
          'endpoint': self._mock_endpoint,
          'traffic_percentage': 100,
      }

    self._mock_endpoint_create.assert_called_with(
        **expected_endpoint_create_body)

    self._mock_model_upload.assert_called_with(**expected_model_upload_body)

    self._mock_model_deploy.assert_called_with(**expected_model_deploy_body)

    self._mock_endpoint_list.assert_called_with(
        filter='display_name="{}"'.format(self._endpoint_name))

  def testDeployModelForAIPPrediction(self):
    self._setUpPredictionMocks()

    runner.deploy_model_for_aip_prediction(
        serving_path=self._serving_path,
        model_version_name=self._model_version,
        ai_platform_serving_args=self._ai_platform_serving_args,
        labels=self._job_labels,
        api=self._mock_api_client)

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
      runner.deploy_model_for_aip_prediction(
          serving_path=self._serving_path,
          model_version_name=self._model_version,
          ai_platform_serving_args=self._ai_platform_serving_args,
          labels=self._job_labels,
          api=self._mock_api_client)

    expected_models_create_body = {
        'name': self._model_name,
        'regions': [],
        'labels': self._job_labels
    }
    self._assertDeployModelMockCalls(
        expected_models_create_body=expected_models_create_body,
        expect_set_default=False)

  def testCreateModel(self):
    self._setUpPredictionMocks()

    self.assertTrue(
        runner.create_model_for_aip_prediction_if_not_exist(
            labels=self._job_labels,
            ai_platform_serving_args=self._ai_platform_serving_args,
            api=self._mock_api_client))

  def testCreateModelCreateError(self):
    self._setUpPredictionMocks()

    self._mock_models_create.return_value.execute.side_effect = (
        errors.HttpError(httplib2.Response(info={'status': 409}), b''))

    self.assertFalse(
        runner.create_model_for_aip_prediction_if_not_exist(
            labels=self._job_labels,
            ai_platform_serving_args=self._ai_platform_serving_args,
            api=self._mock_api_client))

  def testDeployModelForAIPPredictionWithCustomRegion(self):
    self._setUpPredictionMocks()

    self._ai_platform_serving_args['regions'] = ['custom-region']
    runner.deploy_model_for_aip_prediction(
        serving_path=self._serving_path,
        model_version_name=self._model_version,
        ai_platform_serving_args=self._ai_platform_serving_args,
        labels=self._job_labels,
        api=self._mock_api_client)

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
    runner.deploy_model_for_aip_prediction(
        serving_path=self._serving_path,
        model_version_name=self._model_version,
        ai_platform_serving_args=self._ai_platform_serving_args,
        labels=self._job_labels,
        api=self._mock_api_client)

    expected_versions_create_body = {
        'name': self._model_version,
        'deployment_uri': self._serving_path,
        'runtime_version': '1.23.45',
        'python_version': '3.7',
        'labels': self._job_labels,
    }
    self._assertDeployModelMockCalls(
        expected_versions_create_body=expected_versions_create_body)

  def testDeployModelForAIPPredictionWithCustomMachineType(self):
    self._setUpPredictionMocks()

    self._ai_platform_serving_args['machine_type'] = 'custom_machine_type'
    runner.deploy_model_for_aip_prediction(
        serving_path=self._serving_path,
        model_version_name=self._model_version,
        ai_platform_serving_args=self._ai_platform_serving_args,
        labels=self._job_labels,
        api=self._mock_api_client)

    expected_versions_create_body = {
        'name':
            self._model_version,
        'deployment_uri':
            self._serving_path,
        'machine_type':
            'custom_machine_type',
        'runtime_version':
            prediction_clients._get_tf_runtime_version(tf.__version__),
        'python_version':
            '3.7',
        'labels':
            self._job_labels,
    }
    self._assertDeployModelMockCalls(
        expected_versions_create_body=expected_versions_create_body)

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
    model_version_delete_kwargs = self._mock_models_version_delete.call_args[1]
    self.assertNotIn('body', model_version_delete_kwargs)

  @mock.patch('tfx.extensions.google_cloud_ai_platform.runner.discovery')
  def testDeleteModelVersionForAIPPrediction(self, mock_discovery):
    self._setUpDeleteModelVersionMocks()

    runner.delete_model_from_aip_if_exists(
        ai_platform_serving_args=self._ai_platform_serving_args,
        api=self._mock_api_client,
        model_version_name=self._model_version)

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
    model_delete_kwargs = self._mock_models_delete.call_args[1]
    self.assertNotIn('body', model_delete_kwargs)

  @mock.patch('tfx.extensions.google_cloud_ai_platform.runner.discovery')
  def testDeleteModelForAIPPrediction(self, mock_discovery):
    self._setUpDeleteModelMocks()

    runner.delete_model_from_aip_if_exists(
        ai_platform_serving_args=self._ai_platform_serving_args,
        api=self._mock_api_client,
        delete_model_endpoint=True)

    self._assertDeleteModelMockCalls()

  def testDeployModelForVertexPrediction(self):
    self._setUpVertexPredictionMocks()
    self._mock_endpoint_list.side_effect = [[], [self._mock_endpoint]]

    runner.deploy_model_for_aip_prediction(
        serving_path=self._serving_path,
        model_version_name=self._model_name,
        ai_platform_serving_args=self._ai_platform_serving_args_vertex,
        labels=self._job_labels,
        serving_container_image_uri=self._serving_container_image_uri,
        endpoint_region=self._endpoint_region,
        enable_vertex=True)

    expected_endpoint_create_body = {
        'display_name': self._endpoint_name,
        'labels': self._job_labels,
    }
    expected_model_upload_body = {
        'display_name': self._model_name,
        'artifact_uri': self._serving_path,
        'serving_container_image_uri': self._serving_container_image_uri,
    }
    expected_model_deploy_body = {
        'endpoint': self._mock_endpoint,
        'traffic_percentage': 100,
    }

    self._assertDeployModelMockCallsVertex(
        expected_endpoint_create_body=expected_endpoint_create_body,
        expected_model_upload_body=expected_model_upload_body,
        expected_model_deploy_body=expected_model_deploy_body)

  def testDeployModelForVertexPredictionError(self):
    self._setUpVertexPredictionMocks()
    self._mock_endpoint_list.side_effect = [[], [self._mock_endpoint]]

    self._mock_model_deploy.side_effect = errors.HttpError(
        httplib2.Response(info={'status': 429}), b'')

    with self.assertRaises(RuntimeError):
      runner.deploy_model_for_aip_prediction(
          serving_path=self._serving_path,
          model_version_name=self._model_name,
          ai_platform_serving_args=self._ai_platform_serving_args_vertex,
          labels=self._job_labels,
          serving_container_image_uri=self._serving_container_image_uri,
          endpoint_region=self._endpoint_region,
          enable_vertex=True)

    expected_endpoint_create_body = {
        'display_name': self._endpoint_name,
        'labels': self._job_labels,
    }
    expected_model_upload_body = {
        'display_name': self._model_name,
        'artifact_uri': self._serving_path,
        'serving_container_image_uri': self._serving_container_image_uri,
    }
    expected_model_deploy_body = {
        'endpoint': self._mock_endpoint,
        'traffic_percentage': 100,
    }

    self._assertDeployModelMockCallsVertex(
        expected_endpoint_create_body=expected_endpoint_create_body,
        expected_model_upload_body=expected_model_upload_body,
        expected_model_deploy_body=expected_model_deploy_body)

  def testCreateVertexModel(self):
    self._setUpVertexPredictionMocks()

    self.assertTrue(
        runner.create_model_for_aip_prediction_if_not_exist(
            labels=self._job_labels,
            ai_platform_serving_args=self._ai_platform_serving_args_vertex,
            enable_vertex=True))

  def testCreateVertexEndpointCreateErrorAlreadyExist(self):
    self._setUpVertexPredictionMocks()
    self._mock_endpoint_list.return_value = [self._mock_endpoint]

    self.assertFalse(
        runner.create_model_for_aip_prediction_if_not_exist(
            labels=self._job_labels,
            ai_platform_serving_args=self._ai_platform_serving_args_vertex,
            enable_vertex=True))

  def testDeployModelForVertexPredictionWithCustomRegion(self):
    self._setUpVertexPredictionMocks()
    self._mock_endpoint_list.side_effect = [[], [self._mock_endpoint]]

    self._mock_init = mock.Mock()
    aiplatform.init = self._mock_init

    self._endpoint_region = 'custom-region'
    runner.deploy_model_for_aip_prediction(
        serving_path=self._serving_path,
        model_version_name=self._model_name,
        ai_platform_serving_args=self._ai_platform_serving_args_vertex,
        labels=self._job_labels,
        serving_container_image_uri=self._serving_container_image_uri,
        endpoint_region=self._endpoint_region,
        enable_vertex=True)

    expected_init_body = {
        'project': self._project_id,
        'location': 'custom-region',
    }
    self._mock_init.assert_called_with(**expected_init_body)

  def testDeployModelForVertexPredictionWithCustomMachineType(self):
    self._setUpVertexPredictionMocks()
    self._mock_endpoint_list.side_effect = [[], [self._mock_endpoint]]

    self._ai_platform_serving_args_vertex[
        'machine_type'] = 'custom_machine_type'
    runner.deploy_model_for_aip_prediction(
        serving_path=self._serving_path,
        model_version_name=self._model_name,
        ai_platform_serving_args=self._ai_platform_serving_args_vertex,
        labels=self._job_labels,
        serving_container_image_uri=self._serving_container_image_uri,
        endpoint_region=self._endpoint_region,
        enable_vertex=True)

    expected_model_deploy_body = {
        'endpoint': self._mock_endpoint,
        'traffic_percentage': 100,
        'machine_type': 'custom_machine_type',
    }
    self._assertDeployModelMockCallsVertex(
        expected_model_deploy_body=expected_model_deploy_body)

  def _setUpDeleteVertexModelMocks(self):
    importlib.reload(initializer)
    importlib.reload(aiplatform)

    self._endpoint_name = 'endpoint_name'
    self._deployed_model_id = 'model_id'

    self._mock_create_client = mock.Mock()
    initializer.global_config.create_client = self._mock_create_client
    self._mock_create_client.return_value = mock.Mock(
        spec=endpoint_service_client.EndpointServiceClient)

    self._mock_get_endpoint = mock.Mock()
    endpoint_service_client.EndpointServiceClient.get_endpoint = self._mock_get_endpoint
    self._mock_get_endpoint.return_value = endpoint.Endpoint(
        display_name=self._endpoint_name)

    aiplatform.init(
        project=self._project_id,
        location=None,
        credentials=mock.Mock(spec=auth_credentials.AnonymousCredentials()))

    self._mock_endpoint = aiplatform.Endpoint(
        endpoint_name='projects/{}/locations/us-central1/endpoints/1234'.format(
            self._project_id))

    self._mock_endpoint_list = mock.Mock()
    aiplatform.Endpoint.list = self._mock_endpoint_list
    self._mock_endpoint_list.return_value = [self._mock_endpoint]

    self._mock_model_delete = mock.Mock()
    self._mock_endpoint.undeploy = self._mock_model_delete

    self._mock_list_models = mock.Mock()
    self._mock_list_models.return_value = [
        endpoint.DeployedModel(
            display_name=self._model_name, id=self._deployed_model_id)
    ]
    self._mock_endpoint.list_models = self._mock_list_models

    self._ai_platform_serving_args_vertex = {
        'endpoint_name': self._endpoint_name,
        'project_id': self._project_id,
    }

  def _assertDeleteVertexModelMockCalls(self):
    self._mock_model_delete.assert_called_with(
        deployed_model_id=self._deployed_model_id, sync=True)

  def testDeleteModelForVertexPrediction(self):
    self._setUpDeleteVertexModelMocks()

    runner.delete_model_from_aip_if_exists(
        ai_platform_serving_args=self._ai_platform_serving_args_vertex,
        model_version_name=self._model_name,
        enable_vertex=True)

    self._assertDeleteVertexModelMockCalls()

  def _setUpDeleteVertexEndpointMocks(self):
    importlib.reload(initializer)
    importlib.reload(aiplatform)

    self._endpoint_name = 'endpoint_name'

    self._mock_create_client = mock.Mock()
    initializer.global_config.create_client = self._mock_create_client
    self._mock_create_client.return_value = mock.Mock(
        spec=endpoint_service_client.EndpointServiceClient)

    self._mock_get_endpoint = mock.Mock()
    endpoint_service_client.EndpointServiceClient.get_endpoint = (
        self._mock_get_endpoint)
    self._mock_get_endpoint.return_value = endpoint.Endpoint(
        display_name=self._endpoint_name,)

    aiplatform.init(
        project=self._project_id,
        location=None,
        credentials=mock.Mock(spec=auth_credentials.AnonymousCredentials()))

    self._mock_endpoint = aiplatform.Endpoint(
        endpoint_name='projects/{}/locations/us-central1/endpoints/1234'.format(
            self._project_id))

    self._mock_endpoint_list = mock.Mock()
    aiplatform.Endpoint.list = self._mock_endpoint_list
    self._mock_endpoint_list.return_value = [self._mock_endpoint]

    self._mock_endpoint_delete = mock.Mock()
    self._mock_endpoint.delete = self._mock_endpoint_delete

    self._ai_platform_serving_args_vertex = {
        'endpoint_name': self._endpoint_name,
        'project_id': self._project_id,
    }

  def _assertDeleteVertexEndpointMockCalls(self):
    self._mock_endpoint_delete.assert_called_with(force=True, sync=True)

  def testDeleteEndpointForVertexPrediction(self):
    self._setUpDeleteVertexEndpointMocks()

    runner.delete_model_from_aip_if_exists(
        ai_platform_serving_args=self._ai_platform_serving_args_vertex,
        model_version_name=self._model_name,
        delete_model_endpoint=True,
        enable_vertex=True)

    self._assertDeleteVertexEndpointMockCalls()


if __name__ == '__main__':
  tf.test.main()
