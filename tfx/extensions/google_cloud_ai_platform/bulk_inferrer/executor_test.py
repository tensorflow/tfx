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
"""Tests for bulk_inferrer executor."""

import hashlib
import os

from unittest import mock
import tensorflow as tf
from tfx.extensions.google_cloud_ai_platform import constants
from tfx.extensions.google_cloud_ai_platform.bulk_inferrer import executor
from tfx.proto import bulk_inferrer_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import json_utils
from tfx.utils import path_utils
from tfx.utils import proto_utils
from tfx.utils import telemetry_utils
from tfx_bsl.public.proto import model_spec_pb2


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._source_data_dir = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        'components', 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self.component_id = 'test_component'

    # Create input dict.
    self._examples = standard_artifacts.Examples()
    self._examples.uri = os.path.join(self._source_data_dir, 'csv_example_gen')
    self._examples.split_names = artifact_utils.encode_split_names(
        ['unlabelled'])
    self._model = standard_artifacts.Model()
    self._model.uri = os.path.join(self._source_data_dir, 'trainer/current')
    self._model_version = 'version_' + hashlib.sha256(
        self._model.uri.encode()).hexdigest()

    self._model_blessing = standard_artifacts.ModelBlessing()
    self._model_blessing.uri = os.path.join(self._source_data_dir,
                                            'model_validator/blessed')
    self._model_blessing.set_int_custom_property('blessed', 1)

    self._inference_result = standard_artifacts.InferenceResult()
    self._prediction_log_dir = os.path.join(self._output_data_dir,
                                            'prediction_logs')
    self._inference_result.uri = self._prediction_log_dir

    # Create context
    self._tmp_dir = os.path.join(self._output_data_dir, '.temp')
    self._context = executor.Executor.Context(
        tmp_dir=self._tmp_dir, unique_id='2')

  @mock.patch(
      'tfx.extensions.google_cloud_ai_platform.bulk_inferrer.executor.discovery'
  )
  @mock.patch.object(executor.Executor, '_run_model_inference')
  @mock.patch.object(executor, 'runner', autospec=True)
  def testDoWithBlessedModel(self, mock_runner, mock_run_model_inference, _):
    input_dict = {
        'examples': [self._examples],
        'model': [self._model],
        'model_blessing': [self._model_blessing],
    }
    output_dict = {
        'inference_result': [self._inference_result],
    }
    ai_platform_serving_args = {
        'model_name': 'model_name',
        'project_id': 'project_id'
    }
    # Create exe properties.
    exec_properties = {
        'data_spec':
            proto_utils.proto_to_json(bulk_inferrer_pb2.DataSpec()),
        'custom_config':
            json_utils.dumps({
                executor.SERVING_ARGS_KEY:
                    ai_platform_serving_args,
                constants.ENDPOINT_ARGS_KEY:
                    'https://us-central1-ml.googleapis.com',
            }),
    }
    mock_runner.get_service_name_and_api_version.return_value = ('ml', 'v1')
    mock_runner.create_model_for_aip_prediction_if_not_exist.return_value = True

    # Run executor.
    bulk_inferrer = executor.Executor(self._context)
    bulk_inferrer.Do(input_dict, output_dict, exec_properties)

    ai_platform_prediction_model_spec = (
        model_spec_pb2.AIPlatformPredictionModelSpec(
            project_id='project_id',
            model_name='model_name',
            version_name=self._model_version))
    ai_platform_prediction_model_spec.use_serialization_config = True
    inference_endpoint = model_spec_pb2.InferenceSpecType()
    inference_endpoint.ai_platform_prediction_model_spec.CopyFrom(
        ai_platform_prediction_model_spec)
    mock_run_model_inference.assert_called_once_with(mock.ANY, mock.ANY,
                                                     mock.ANY, mock.ANY,
                                                     mock.ANY,
                                                     inference_endpoint)
    executor_class_path = '%s.%s' % (bulk_inferrer.__class__.__module__,
                                     bulk_inferrer.__class__.__name__)
    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_EXECUTOR: executor_class_path}):
      job_labels = telemetry_utils.make_labels_dict()
    mock_runner.deploy_model_for_aip_prediction.assert_called_once_with(
        serving_path=path_utils.serving_model_path(self._model.uri),
        model_version_name=mock.ANY,
        ai_platform_serving_args=ai_platform_serving_args,
        labels=job_labels,
        api=mock.ANY,
        skip_model_endpoint_creation=True,
        set_default=False)
    mock_runner.delete_model_from_aip_if_exists.assert_called_once_with(
        model_version_name=mock.ANY,
        ai_platform_serving_args=ai_platform_serving_args,
        api=mock.ANY,
        delete_model_endpoint=True)

  @mock.patch(
      'tfx.extensions.google_cloud_ai_platform.bulk_inferrer.executor.discovery'
  )
  @mock.patch.object(executor.Executor, '_run_model_inference')
  @mock.patch.object(executor, 'runner', autospec=True)
  def testDoSkippedModelCreation(self, mock_runner, mock_run_model_inference,
                                 _):
    input_dict = {
        'examples': [self._examples],
        'model': [self._model],
        'model_blessing': [self._model_blessing],
    }
    output_dict = {
        'inference_result': [self._inference_result],
    }
    ai_platform_serving_args = {
        'model_name': 'model_name',
        'project_id': 'project_id'
    }
    # Create exe properties.
    exec_properties = {
        'data_spec':
            proto_utils.proto_to_json(bulk_inferrer_pb2.DataSpec()),
        'custom_config':
            json_utils.dumps(
                {executor.SERVING_ARGS_KEY: ai_platform_serving_args}),
    }
    mock_runner.get_service_name_and_api_version.return_value = ('ml', 'v1')
    mock_runner.create_model_for_aip_prediction_if_not_exist.return_value = False

    # Run executor.
    bulk_inferrer = executor.Executor(self._context)
    bulk_inferrer.Do(input_dict, output_dict, exec_properties)

    ai_platform_prediction_model_spec = (
        model_spec_pb2.AIPlatformPredictionModelSpec(
            project_id='project_id',
            model_name='model_name',
            version_name=self._model_version))
    ai_platform_prediction_model_spec.use_serialization_config = True
    inference_endpoint = model_spec_pb2.InferenceSpecType()
    inference_endpoint.ai_platform_prediction_model_spec.CopyFrom(
        ai_platform_prediction_model_spec)
    mock_run_model_inference.assert_called_once_with(mock.ANY, mock.ANY,
                                                     mock.ANY, mock.ANY,
                                                     mock.ANY,
                                                     inference_endpoint)
    executor_class_path = '%s.%s' % (bulk_inferrer.__class__.__module__,
                                     bulk_inferrer.__class__.__name__)
    with telemetry_utils.scoped_labels(
        {telemetry_utils.LABEL_TFX_EXECUTOR: executor_class_path}):
      job_labels = telemetry_utils.make_labels_dict()
    mock_runner.deploy_model_for_aip_prediction.assert_called_once_with(
        serving_path=path_utils.serving_model_path(self._model.uri),
        model_version_name=mock.ANY,
        ai_platform_serving_args=ai_platform_serving_args,
        labels=job_labels,
        api=mock.ANY,
        skip_model_endpoint_creation=True,
        set_default=False)
    mock_runner.delete_model_from_aip_if_exists.assert_called_once_with(
        model_version_name=mock.ANY,
        ai_platform_serving_args=ai_platform_serving_args,
        api=mock.ANY,
        delete_model_endpoint=False)

  @mock.patch(
      'tfx.extensions.google_cloud_ai_platform.bulk_inferrer.executor.discovery'
  )
  @mock.patch.object(executor.Executor, '_run_model_inference')
  @mock.patch.object(executor, 'runner', autospec=True)
  def testDoFailedModelDeployment(self, mock_runner, mock_run_model_inference,
                                  _):
    input_dict = {
        'examples': [self._examples],
        'model': [self._model],
        'model_blessing': [self._model_blessing],
    }
    output_dict = {
        'inference_result': [self._inference_result],
    }
    ai_platform_serving_args = {
        'model_name': 'model_name',
        'project_id': 'project_id'
    }
    # Create exe properties.
    exec_properties = {
        'data_spec':
            proto_utils.proto_to_json(bulk_inferrer_pb2.DataSpec()),
        'custom_config':
            json_utils.dumps(
                {executor.SERVING_ARGS_KEY: ai_platform_serving_args}),
    }
    mock_runner.deploy_model_for_aip_prediction.side_effect = (
        Exception('Deployment failed'))
    mock_runner.get_service_name_and_api_version.return_value = ('ml', 'v1')
    mock_runner.create_model_for_aip_prediction_if_not_exist.return_value = True

    bulk_inferrer = executor.Executor(self._context)
    with self.assertRaises(Exception):
      bulk_inferrer.Do(input_dict, output_dict, exec_properties)

    mock_runner.delete_model_from_aip_if_exists.assert_called_once_with(
        model_version_name=mock.ANY,
        ai_platform_serving_args=ai_platform_serving_args,
        api=mock.ANY,
        delete_model_endpoint=True)


if __name__ == '__main__':
  tf.test.main()
