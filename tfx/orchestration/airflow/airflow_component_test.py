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
"""Tests for tfx.orchestration.airflow.airflow_component."""

import collections
import datetime
import os
from unittest import mock

from airflow import models
from airflow.operators import python_operator

import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration.airflow import airflow_component
from tfx.types import component_spec


class _ArtifactTypeA(types.Artifact):
  TYPE_NAME = 'ArtifactTypeA'


class _ArtifactTypeB(types.Artifact):
  TYPE_NAME = 'ArtifactTypeB'


class _FakeComponentSpec(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {
      'input': component_spec.ChannelParameter(type=_ArtifactTypeA),
  }
  OUTPUTS = {'output': component_spec.ChannelParameter(type=_ArtifactTypeB)}


class _FakeComponent(base_component.BaseComponent):

  SPEC_CLASS = types.ComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)

  def __init__(self, spec: types.ComponentSpec):
    super().__init__(spec=spec)


class AirflowComponentTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._component = _FakeComponent(
        _FakeComponentSpec(
            input=types.Channel(type=_ArtifactTypeA),
            output=types.Channel(type=_ArtifactTypeB)))
    self._pipeline_info = data_types.PipelineInfo('name', 'root')
    self._driver_args = data_types.DriverArgs(True)
    self._metadata_connection_config = metadata.sqlite_metadata_connection_config(
        os.path.join(
            os.environ.get('TEST_TMP_DIR', self.get_temp_dir()), 'metadata'))
    self._parent_dag = models.DAG(
        dag_id=self._pipeline_info.pipeline_name,
        start_date=datetime.datetime(2018, 1, 1),
        schedule_interval=None)

  def testAirflowAdaptor(self):
    fake_dagrun = collections.namedtuple('fake_dagrun', ['run_id'])
    mock_ti = mock.Mock()
    mock_ti.get_dagrun.return_value = fake_dagrun('run_id')
    mock_component_launcher = mock.Mock()
    mock_component_launcher_class = mock.Mock()
    mock_component_launcher_class.create.return_value = mock_component_launcher
    airflow_component._airflow_component_launcher(
        component=self._component,
        component_launcher_class=mock_component_launcher_class,
        pipeline_info=self._pipeline_info,
        driver_args=self._driver_args,
        metadata_connection_config=self._metadata_connection_config,
        beam_pipeline_args=[],
        additional_pipeline_args={},
        component_config=None,
        ti=mock_ti,
        exec_properties={})
    mock_component_launcher_class.create.assert_called_once()
    arg_list = mock_component_launcher_class.create.call_args_list
    self.assertEqual(arg_list[0][1]['pipeline_info'].run_id, 'run_id')
    mock_component_launcher.launch.assert_called_once()

  @mock.patch.object(python_operator.PythonOperator, '__init__')
  def testAirflowComponent(self, mock_python_operator_init):
    mock_component_launcher_class = mock.Mock()
    airflow_component.AirflowComponent(
        parent_dag=self._parent_dag,
        component=self._component,
        component_launcher_class=mock_component_launcher_class,
        pipeline_info=self._pipeline_info,
        enable_cache=True,
        metadata_connection_config=self._metadata_connection_config,
        beam_pipeline_args=[],
        additional_pipeline_args={},
        component_config=None)

    mock_python_operator_init.assert_called_once_with(
        task_id=self._component.id,
        provide_context=True,
        python_callable=mock.ANY,
        dag=self._parent_dag,
        op_kwargs={'exec_properties': {}})

    python_callable = mock_python_operator_init.call_args_list[0][1][
        'python_callable']
    self.assertEqual(python_callable.func,
                     airflow_component._airflow_component_launcher)
    self.assertTrue(python_callable.keywords.pop('driver_args').enable_cache)
    self.assertEqual(
        python_callable.keywords, {
            'component': self._component,
            'component_launcher_class': mock_component_launcher_class,
            'pipeline_info': self._pipeline_info,
            'metadata_connection_config': self._metadata_connection_config,
            'beam_pipeline_args': [],
            'additional_pipeline_args': {},
            'component_config': None,
        })


if __name__ == '__main__':
  tf.test.main()
