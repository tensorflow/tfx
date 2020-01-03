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
"""Tests for tfx.orchestration.airflow.airflow_component."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import os
from airflow import models
import mock

import tensorflow as tf
from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base import executor_spec
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
    super(_FakeComponent, self).__init__(spec=spec)


class AirflowComponentTest(tf.test.TestCase):

  def setUp(self):
    super(AirflowComponentTest, self).setUp()
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
        ti=mock_ti)
    mock_component_launcher_class.create.assert_called_once()
    arg_list = mock_component_launcher_class.create.call_args_list
    self.assertEqual(arg_list[0][1]['pipeline_info'].run_id, 'run_id')
    mock_component_launcher.launch.assert_called_once()

  @mock.patch('functools.partial')
  def testAirflowComponent(self, mock_functools_partial):
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
    mock_functools_partial.assert_called_once_with(
        airflow_component._airflow_component_launcher,
        component=self._component,
        component_launcher_class=mock_component_launcher_class,
        pipeline_info=self._pipeline_info,
        driver_args=mock.ANY,
        metadata_connection_config=self._metadata_connection_config,
        beam_pipeline_args=[],
        additional_pipeline_args={},
        component_config=None)
    arg_list = mock_functools_partial.call_args_list
    self.assertTrue(arg_list[0][1]['driver_args'].enable_cache)


if __name__ == '__main__':
  tf.test.main()
