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

from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.orchestration import pipeline
from tfx.orchestration.airflow import airflow_dag_runner
from tfx.utils import channel


class _FakeComponentSpecA(base_component.ComponentSpec):
  COMPONENT_NAME = 'component_a'
  PARAMETERS = {}
  INPUTS = {}
  OUTPUTS = {'output': base_component.ChannelParameter(type_name='a')}


class _FakeComponentSpecB(base_component.ComponentSpec):
  COMPONENT_NAME = 'component_b'
  PARAMETERS = {}
  INPUTS = {'a': base_component.ChannelParameter(type_name='a')}
  OUTPUTS = {'output': base_component.ChannelParameter(type_name='b')}


class _FakeComponentSpecC(base_component.ComponentSpec):
  COMPONENT_NAME = 'component_c'
  PARAMETERS = {}
  INPUTS = {
      'a': base_component.ChannelParameter(type_name='a'),
      'b': base_component.ChannelParameter(type_name='b')
  }
  OUTPUTS = {'output': base_component.ChannelParameter(type_name='c')}


class _FakeComponentSpecD(base_component.ComponentSpec):
  COMPONENT_NAME = 'component_d'
  PARAMETERS = {}
  INPUTS = {
      'b': base_component.ChannelParameter(type_name='b'),
      'c': base_component.ChannelParameter(type_name='c'),
  }
  OUTPUTS = {'output': base_component.ChannelParameter(type_name='d')}


class _FakeComponentSpecE(base_component.ComponentSpec):
  COMPONENT_NAME = 'component_e'
  PARAMETERS = {}
  INPUTS = {
      'a': base_component.ChannelParameter(type_name='a'),
      'b': base_component.ChannelParameter(type_name='b'),
      'd': base_component.ChannelParameter(type_name='d'),
  }
  OUTPUTS = {'output': base_component.ChannelParameter(type_name='e')}


class _FakeComponent(base_component.BaseComponent):

  SPEC_CLASS = base_component.ComponentSpec
  EXECUTOR_CLASS = base_executor.BaseExecutor

  def __init__(self, spec: base_component.ComponentSpec):
    super(_FakeComponent, self).__init__(spec=spec)


class AirflowDagRunnerTest(tf.test.TestCase):

  @mock.patch(
      'tfx.orchestration.airflow.airflow_component.AirflowComponent'
  )
  @mock.patch('airflow.models.DAG')
  def test_airflow_dag_runner(self, mock_airflow_dag_class,
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
        _FakeComponentSpecA(output=channel.Channel(type_name='a')))
    component_b = _FakeComponent(
        _FakeComponentSpecB(
            a=component_a.outputs.output,
            output=channel.Channel(type_name='b')))
    component_c = _FakeComponent(
        _FakeComponentSpecC(
            a=component_a.outputs.output,
            b=component_b.outputs.output,
            output=channel.Channel(type_name='c')))
    component_d = _FakeComponent(
        _FakeComponentSpecD(
            b=component_b.outputs.output,
            c=component_c.outputs.output,
            output=channel.Channel(type_name='d')))
    component_e = _FakeComponent(
        _FakeComponentSpecE(
            a=component_a.outputs.output,
            b=component_b.outputs.output,
            d=component_d.outputs.output,
            output=channel.Channel(type_name='e')))

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
