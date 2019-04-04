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
"""Tests for tfx.examples.chicago_taxi_pipeline.taxi_pipeline_simple."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

import tensorflow as tf

from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_simple
from tfx.orchestration.airflow.airflow_pipeline import AirflowPipeline
from tfx.orchestration.airflow.airflow_runner import AirflowDAGRunner as TfxRunner


class TaxiPipelineSimpleTest(tf.test.TestCase):

  def setUp(self):
    self._original_home_value = os.environ.get('HOME', '')
    os.environ['HOME'] = '/tmp'

  def tearDown(self):
    os.environ['HOME'] = self._original_home_value

  def test_taxi_pipeline_check_dag_construction(self):
    airflow_config = {
        'schedule_interval': None,
        'start_date': datetime.datetime(2019, 1, 1),
    }
    logical_pipeline = taxi_pipeline_simple._create_pipeline()
    self.assertEqual(9, len(logical_pipeline.components))
    pipeline = TfxRunner(airflow_config).run(logical_pipeline)
    self.assertIsInstance(pipeline, AirflowPipeline)


if __name__ == '__main__':
  tf.test.main()
