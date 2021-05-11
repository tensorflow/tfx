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

import datetime
import os

from airflow import models

import tensorflow as tf

from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
from tfx.utils import test_case_utils


class TaxiPipelineSimpleTest(test_case_utils.TfxTest):

  def setUp(self):
    super(TaxiPipelineSimpleTest, self).setUp()
    self._test_dir = self.tmp_dir

  def testTaxiPipelineCheckDagConstruction(self):
    airflow_config = {
        'schedule_interval': None,
        'start_date': datetime.datetime(2019, 1, 1),
    }

    # Create directory structure and write expected user module file.
    os.makedirs(os.path.join(self._test_dir, 'taxi'))
    module_file = os.path.join(self._test_dir, 'taxi/taxi_utils.py')
    with open(module_file, 'w') as f:
      f.write('# Placeholder user module file.')

    # Patch $HOME directory for pipeline DAG construction.
    original_home = os.environ['HOME']
    os.environ['HOME'] = self._test_dir
    from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_simple  # pylint: disable=g-import-not-at-top
    os.environ['HOME'] = original_home

    logical_pipeline = taxi_pipeline_simple._create_pipeline(
        pipeline_name='Test',
        pipeline_root=self._test_dir,
        data_root=self._test_dir,
        module_file=module_file,
        serving_model_dir=self._test_dir,
        metadata_path=self._test_dir,
        beam_pipeline_args=[])
    self.assertEqual(9, len(logical_pipeline.components))
    pipeline = AirflowDagRunner(
        AirflowPipelineConfig(airflow_config)).run(logical_pipeline)
    self.assertIsInstance(pipeline, models.DAG)


if __name__ == '__main__':
  tf.test.main()
