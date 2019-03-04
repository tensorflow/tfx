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
"""Test for tfx.orchestration.airflow.airflow_runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import mock
import tensorflow as tf

from tfx.components.schema_gen import component as schema_component
from tfx.components.statistics_gen import component as stats_gen_component
from tfx.orchestration import pipeline
from tfx.orchestration.airflow import airflow_runner
from tfx.utils import channel


class AirflowRunnerTest(tf.test.TestCase):

  @mock.patch('tfx.orchestration.airflow.airflow_component.Component')
  @mock.patch('tfx.orchestration.airflow.airflow_pipeline.AirflowPipeline')
  def test_airflow_runner(self, mock_airflow_pipeline_class,
                          mock_airflow_component_class):
    mock_airflow_pipeline_class.return_value = 'DAG'

    c1 = stats_gen_component.StatisticsGen(
        input_data=channel.Channel(type_name='ExamplesPath'))
    c2 = schema_component.SchemaGen(
        stats=channel.Channel(type_name='ExampleStatisticsPath'))
    airflow_config = {'schedule_interval': '* * * * *',
                      'start_date': datetime.datetime(2019, 1, 1)}
    pipeline_config = {
        'pipeline_name': 'chicago_taxi_gcp',
        'log_root': '/var/tmp/tfx/logs',
        'metadata_db_root': 'var/tmp/tfx//metadata',
        'pipeline_root': '/var/tmp/tfx/pipelines'
    }
    # Simulate the runner's call to pipeline
    combined_config = pipeline_config.copy()
    combined_config.update(airflow_config)

    tfx_pipeline = pipeline.Pipeline(a='a', b='b', **pipeline_config)
    tfx_pipeline.components = [c1, c2]

    tfx_runner = airflow_runner.AirflowDAGRunner(airflow_config)
    tfx_runner.run(tfx_pipeline)

    mock_airflow_pipeline_class.assert_called_with(
        a='a', b='b', **combined_config)

    component_calls = [
        mock.call(
            'DAG',
            component_name=c1.component_name,
            unique_name=c1.unique_name,
            driver=c1.driver,
            executor=c1.executor,
            input_dict=mock.ANY,
            output_dict=mock.ANY,
            exec_properties=c1.exec_properties),
        mock.call(
            'DAG',
            component_name=c2.component_name,
            unique_name=c2.unique_name,
            driver=c2.driver,
            executor=c2.executor,
            input_dict=mock.ANY,
            output_dict=mock.ANY,
            exec_properties=c2.exec_properties)
    ]
    mock_airflow_component_class.assert_has_calls(component_calls)


if __name__ == '__main__':
  tf.test.main()
