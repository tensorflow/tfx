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
"""Tests for tfx.orchestration.airflow.airflow_dag_runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import mock
import tensorflow as tf

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base import executor_spec
from tfx.orchestration import pipeline
from tfx.orchestration.airflow import airflow_dag_runner
from tfx.types import component_spec


class _FakeComponentSpecA(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {}
  OUTPUTS = {'output': component_spec.ChannelParameter(type_name='a')}


class _FakeComponentSpecB(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {'a': component_spec.ChannelParameter(type_name='a')}
  OUTPUTS = {'output': component_spec.ChannelParameter(type_name='b')}


class _FakeComponentSpecC(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {
      'a': component_spec.ChannelParameter(type_name='a'),
      'b': component_spec.ChannelParameter(type_name='b')
  }
  OUTPUTS = {'output': component_spec.ChannelParameter(type_name='c')}


class _FakeComponentSpecD(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {
      'b': component_spec.ChannelParameter(type_name='b'),
      'c': component_spec.ChannelParameter(type_name='c'),
  }
  OUTPUTS = {'output': component_spec.ChannelParameter(type_name='d')}


class _FakeComponentSpecE(types.ComponentSpec):
  PARAMETERS = {}
  INPUTS = {
      'a': component_spec.ChannelParameter(type_name='a'),
      'b': component_spec.ChannelParameter(type_name='b'),
      'd': component_spec.ChannelParameter(type_name='d'),
  }
  OUTPUTS = {'output': component_spec.ChannelParameter(type_name='e')}


class _FakeComponent(base_component.BaseComponent):

  SPEC_CLASS = types.ComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)

  def __init__(self, spec: types.ComponentSpec):
    instance_name = spec.__class__.__name__.replace(
        '_FakeComponentSpec', '')
    super(_FakeComponent, self).__init__(
        spec=spec, instance_name=instance_name)


class AirflowDagRunnerTest(tf.test.TestCase):

  @mock.patch(
      'tfx.orchestration.airflow.airflow_component.AirflowComponent'
  )
  @mock.patch('airflow.models.DAG')
  def testAirflowDagRunner(self, mock_airflow_dag_class,
                           mock_airflow_component_class):
    mock_airflow_dag_class.return_value = 'DAG'
    mock_airflow_component_a = mock.Mock()
    mock_airflow_component_b = mock.Mock()
    mock_airflow_component_c = mock.Mock()
    mock_airflow_component_d = mock.Mock()
    mock_airflow_component_e = mock.Mock()
    mock_airflow_component_class.side_effect = [
        mock_airflow_component_a, mock_airflow_component_b,
        mock_airflow_component_c, mock_airflow_component_d,
        mock_airflow_component_e
    ]

    airflow_config = {
        'schedule_interval': '* * * * *',
        'start_date': datetime.datetime(2019, 1, 1)
    }
    component_a = _FakeComponent(
        _FakeComponentSpecA(output=types.Channel(type_name='a')))
    component_b = _FakeComponent(
        _FakeComponentSpecB(
            a=component_a.outputs['output'],
            output=types.Channel(type_name='b')))
    component_c = _FakeComponent(
        _FakeComponentSpecC(
            a=component_a.outputs['output'],
            b=component_b.outputs['output'],
            output=types.Channel(type_name='c')))
    component_d = _FakeComponent(
        _FakeComponentSpecD(
            b=component_b.outputs['output'],
            c=component_c.outputs['output'],
            output=types.Channel(type_name='d')))
    component_e = _FakeComponent(
        _FakeComponentSpecE(
            a=component_a.outputs['output'],
            b=component_b.outputs['output'],
            d=component_d.outputs['output'],
            output=types.Channel(type_name='e')))

    test_pipeline = pipeline.Pipeline(
        pipeline_name='x',
        pipeline_root='y',
        metadata_connection_config=None,
        components=[
            component_d, component_c, component_a, component_b, component_e
        ])
    runner = airflow_dag_runner.AirflowDagRunner(config=airflow_config)
    runner.run(test_pipeline)

    mock_airflow_component_a.set_upstream.assert_not_called()
    mock_airflow_component_b.set_upstream.assert_has_calls(
        [mock.call(mock_airflow_component_a)])
    mock_airflow_component_c.set_upstream.assert_has_calls([
        mock.call(mock_airflow_component_a),
        mock.call(mock_airflow_component_b)
    ],
                                                           any_order=True)
    mock_airflow_component_d.set_upstream.assert_has_calls([
        mock.call(mock_airflow_component_b),
        mock.call(mock_airflow_component_c)
    ],
                                                           any_order=True)
    mock_airflow_component_e.set_upstream.assert_has_calls([
        mock.call(mock_airflow_component_a),
        mock.call(mock_airflow_component_b),
        mock.call(mock_airflow_component_d)
    ],
                                                           any_order=True)


if __name__ == '__main__':
  tf.test.main()
