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
"""Tests for tfx.orchestration.airflow.airflow_adapter."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import mock

import tensorflow as tf
from tfx.components.base import base_driver
from tfx.components.base import base_executor
from tfx.orchestration.airflow import airflow_adapter
from tfx.orchestration.airflow import airflow_component
from tfx.utils import logging_utils
from tfx.utils import types


class AirflowAdapterTest(tf.test.TestCase):

  def setUp(self):
    self.input_one = types.TfxType('INPUT_ONE')
    self.input_one.source = airflow_component._OrchestrationSource(
        'input_one_key', 'input_one_component_id')
    self.output_one = types.TfxType('OUTPUT_ONE')
    self.output_one.source = airflow_component._OrchestrationSource(
        'output_one_key', 'output_one_component_id')
    self.input_one_json = json.dumps([self.input_one.json_dict()])
    self.output_one_json = json.dumps([self.output_one.json_dict()])
    self._logger_config = logging_utils.LoggerConfig()

  def _setup_mocks(self, mock_metadata_class, mock_driver_class,
                   mock_executor_class, mock_docker_operator_class,
                   mock_get_logger):
    self._setup_mock_driver(mock_driver_class)
    self._setup_mock_executor(mock_executor_class)
    self._setup_mock_metadata(mock_metadata_class)
    self._setup_mock_task_instance()
    self._mock_get_logger = mock_get_logger

  def _setup_adapter_and_args(self):
    input_dict = {u'input_one': [self.input_one]}
    output_dict = {u'output_one': [self.output_one]}
    exec_properties = {}
    driver_options = {}

    adapter = airflow_adapter.AirflowAdapter(
        component_name='TfxComponent',
        input_dict=input_dict,
        output_dict=output_dict,
        exec_properties=exec_properties,
        driver_class=base_driver.BaseDriver,
        executor_class=base_executor.BaseExecutor,
        driver_options=driver_options,
        additional_pipeline_args=None,
        metadata_connection_config='metadata_connection_config',
        logger_config=self._logger_config)

    return adapter, input_dict, output_dict, exec_properties, driver_options

  def _setup_mock_metadata(self, mock_metadata_class):
    mock_metadata = mock.Mock()
    enter_mock = mock.Mock()
    exit_mock = mock.Mock()

    mock_metadata_class.return_value = mock_metadata
    mock_metadata.__enter__ = enter_mock
    mock_metadata.__exit__ = exit_mock
    enter_mock.return_value = mock_metadata

    self.mock_metadata = mock_metadata

  def _setup_mock_driver(self, mock_driver_class):
    mock_driver = mock.Mock()
    prepare_execution_mock = mock.Mock()

    mock_driver_class.return_value = mock_driver
    mock_driver.prepare_execution = prepare_execution_mock

    self.mock_driver = mock_driver

  def _setup_mock_executor(self, mock_executor_class):
    mock_executor = mock.Mock()
    mock_executor_class.return_value = mock_executor
    mock_executor_class.__name__ = 'mock_executor_class'

    self.mock_executor = mock_executor

  def _setup_mock_task_instance(self):
    mock_task_instance = mock.Mock()

    self.mock_task_instance = mock_task_instance

  @mock.patch('tfx.utils.logging_utils.get_logger')
  @mock.patch('airflow.operators.docker_operator.DockerOperator')
  @mock.patch('tfx.components.base.base_executor.BaseExecutor')
  @mock.patch('tfx.components.base.base_driver.BaseDriver')
  @mock.patch('tfx.orchestration.metadata.Metadata')
  def test_cached_execution(self, mock_metadata_class, mock_driver_class,
                            mock_executor_class, mock_docker_operator_class,
                            mock_get_logger):
    self._setup_mocks(mock_metadata_class, mock_driver_class,
                      mock_executor_class, mock_docker_operator_class,
                      mock_get_logger)
    adapter, input_dict, output_dict, exec_properties, driver_options = self._setup_adapter_and_args(
    )

    self.mock_task_instance.xcom_pull.side_effect = [self.input_one_json]

    self.mock_driver.prepare_execution.return_value = base_driver.ExecutionDecision(
        input_dict, output_dict, exec_properties)

    check_result = adapter.check_cache_and_maybe_prepare_execution(
        'cached_branch',
        'uncached_branch',
        ti=self.mock_task_instance)

    mock_get_logger.assert_called_with(self._logger_config)
    mock_driver_class.assert_called_with(
        logger=mock.ANY, metadata_handler=self.mock_metadata)
    self.mock_driver.prepare_execution.called_with(
        input_dict, output_dict, exec_properties, driver_options)
    self.mock_task_instance.xcom_pull.assert_called_with(
        dag_id='input_one_component_id', key='input_one_key')
    self.mock_task_instance.xcom_push.assert_called_with(
        key='output_one_key', value=self.output_one_json)

    self.assertEqual(check_result, 'cached_branch')

  @mock.patch('tfx.utils.logging_utils.get_logger')
  @mock.patch('airflow.operators.docker_operator.DockerOperator')
  @mock.patch('tfx.components.base.base_executor.BaseExecutor')
  @mock.patch('tfx.components.base.base_driver.BaseDriver')
  @mock.patch('tfx.orchestration.metadata.Metadata')
  def test_new_execution(self, mock_metadata_class, mock_driver_class,
                         mock_executor_class, mock_docker_operator_class,
                         mock_get_logger):
    self._setup_mocks(mock_metadata_class, mock_driver_class,
                      mock_executor_class, mock_docker_operator_class,
                      mock_get_logger)
    adapter, input_dict, output_dict, exec_properties, driver_options = self._setup_adapter_and_args(
    )

    self.mock_task_instance.xcom_pull.side_effect = [self.input_one_json]

    self.mock_driver.prepare_execution.return_value = base_driver.ExecutionDecision(
        input_dict, output_dict, exec_properties, execution_id=12345)

    check_result = adapter.check_cache_and_maybe_prepare_execution(
        'cached_branch',
        'uncached_branch',
        ti=self.mock_task_instance)

    mock_driver_class.assert_called_with(
        logger=mock.ANY, metadata_handler=self.mock_metadata)
    self.mock_driver.prepare_execution.called_with(
        input_dict, output_dict, exec_properties, driver_options)
    self.mock_task_instance.xcom_pull.assert_called_with(
        dag_id='input_one_component_id', key='input_one_key')

    calls = [
        mock.call(
            key='_exec_inputs', value=types.jsonify_tfx_type_dict(input_dict)),
        mock.call(
            key='_exec_outputs',
            value=types.jsonify_tfx_type_dict(output_dict)),
        mock.call(key='_exec_properties', value=json.dumps(exec_properties)),
        mock.call(key='_execution_id', value=12345)
    ]
    self.mock_task_instance.xcom_push.assert_has_calls(calls)

    self.assertEqual(check_result, 'uncached_branch')

  @mock.patch('tfx.utils.logging_utils.get_logger')
  @mock.patch('airflow.operators.docker_operator.DockerOperator')
  @mock.patch('tfx.components.base.base_executor.BaseExecutor')
  @mock.patch('tfx.components.base.base_driver.BaseDriver')
  @mock.patch('tfx.orchestration.metadata.Metadata')
  def test_python_exec(self, mock_metadata_class, mock_driver_class,
                       mock_executor_class, mock_docker_operator_class,
                       mock_get_logger):
    self._setup_mocks(mock_metadata_class, mock_driver_class,
                      mock_executor_class, mock_docker_operator_class,
                      mock_get_logger)
    adapter, input_dict, output_dict, exec_properties, _ = self._setup_adapter_and_args(
    )

    self.mock_task_instance.xcom_pull.side_effect = [
        types.jsonify_tfx_type_dict(input_dict),
        types.jsonify_tfx_type_dict(output_dict),
        json.dumps(exec_properties), 12345
    ]

    adapter.python_exec('cache_task_name', ti=self.mock_task_instance)

    calls = [
        mock.call(key='_exec_inputs', task_ids='cache_task_name'),
        mock.call(key='_exec_outputs', task_ids='cache_task_name'),
        mock.call(key='_exec_properties', task_ids='cache_task_name'),
        mock.call(key='_execution_id', task_ids='cache_task_name')
    ]

    self.assertEqual(
        json.dumps(exec_properties), json.dumps(adapter._exec_properties))
    mock_executor_class.assert_called_once()
    self.mock_executor.Do.assert_called_with(
        adapter._input_dict, adapter._output_dict, adapter._exec_properties)
    self.mock_task_instance.xcom_pull.assert_has_calls(calls)
    self.mock_task_instance.xcom_push.assert_called_once()

  @mock.patch('tfx.utils.logging_utils.get_logger')
  @mock.patch('airflow.operators.docker_operator.DockerOperator')
  @mock.patch('tfx.components.base.base_executor.BaseExecutor')
  @mock.patch('tfx.components.base.base_driver.BaseDriver')
  @mock.patch('tfx.orchestration.metadata.Metadata')
  def test_docker_exec(self, mock_metadata_class, mock_driver_class,
                       mock_executor_class, mock_docker_operator_class,
                       mock_get_logger):
    self._setup_mocks(mock_metadata_class, mock_driver_class,
                      mock_executor_class, mock_docker_operator_class,
                      mock_get_logger)
    adapter, _, _, _, _ = self._setup_adapter_and_args()

    adapter.docker_operator('task_id', 'parent_dag', {
        'volumes': ['root', 'base', 'taxi'],
    }, 'pusher_task')

    expected_command = ('--write-outputs-stdout '
                        '--executor_class_path=mock.mock.mock_executor_class '
                        '--inputs-base64={{ ti.xcom_pull(key="_exec_inputs", '
                        'task_ids="pusher_task") | b64encode }} '
                        '--outputs-base64={{ ti.xcom_pull(key="_exec_outputs", '
                        'task_ids="pusher_task") | b64encode }} '
                        '--exec-properties-base64={{ '
                        'ti.xcom_pull(key="_exec_properties", '
                        'task_ids="pusher_task") | b64encode }}')
    expected_volume = ['root:root:rw', 'base:base:rw', 'taxi:taxi:rw']

    mock_docker_operator_class.assert_called_with(
        dag='parent_dag',
        task_id='task_id',
        command=expected_command,
        volumes=expected_volume,
        xcom_push=True,
        image='tfx-executors-test:latest',
    )

  @mock.patch('tfx.utils.logging_utils.get_logger')
  @mock.patch('airflow.operators.docker_operator.DockerOperator')
  @mock.patch('tfx.components.base.base_executor.BaseExecutor')
  @mock.patch('tfx.components.base.base_driver.BaseDriver')
  @mock.patch('tfx.orchestration.metadata.Metadata')
  def test_publish_exec(self, mock_metadata_class, mock_driver_class,
                        mock_executor_class, mock_docker_operator_class,
                        mock_get_logger):
    self._setup_mocks(mock_metadata_class, mock_driver_class,
                      mock_executor_class, mock_docker_operator_class,
                      mock_get_logger)
    adapter, input_dict, output_dict, exec_properties, _ = self._setup_adapter_and_args(
    )

    self.mock_task_instance.xcom_pull.side_effect = [
        types.jsonify_tfx_type_dict(input_dict),
        types.jsonify_tfx_type_dict(output_dict),
        json.dumps(exec_properties), 12345,
        types.jsonify_tfx_type_dict(output_dict)
    ]
    output_artifact_published = types.TfxType('O')
    output_artifact_published.source = self.output_one.source
    self.mock_metadata.publish_execution.return_value = {
        u'output_one': [output_artifact_published]
    }

    adapter.publish_exec(
        'cache_task_name', 'exec_task_name', ti=self.mock_task_instance)

    calls = [
        mock.call(key='_exec_inputs', task_ids='cache_task_name'),
        mock.call(key='_exec_outputs', task_ids='cache_task_name'),
        mock.call(key='_exec_properties', task_ids='cache_task_name'),
        mock.call(key='_execution_id', task_ids='cache_task_name'),
        mock.call(key='return_value', task_ids='exec_task_name')
    ]

    self.mock_metadata.publish_execution.assert_called_with(
        12345, adapter._input_dict, adapter._output_dict)
    self.mock_task_instance.xcom_pull.assert_has_calls(calls)
    self.mock_task_instance.xcom_push.assert_called_once()


if __name__ == '__main__':
  tf.test.main()
