# Lint as: python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""E2E tests for Kubeflow V2 runner."""

import os

from absl.testing import parameterized
import tensorflow as tf

from tfx.orchestration import test_utils
from tfx.orchestration.kubeflow.v2 import test_utils as kubeflow_v2_test_utils


class KubeflowV2E2ETest(
    kubeflow_v2_test_utils.BaseKubeflowV2Test,
    parameterized.TestCase):
  """Kubeflow V2 runner E2E test."""

  # The query to get data from BigQuery.
  _BIGQUERY_QUERY = """
          SELECT
            pickup_community_area,
            fare,
            EXTRACT(MONTH FROM trip_start_timestamp) AS trip_start_month,
            EXTRACT(HOUR FROM trip_start_timestamp) AS trip_start_hour,
            EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS trip_start_day,
            UNIX_SECONDS(trip_start_timestamp) AS trip_start_timestamp,
            pickup_latitude,
            pickup_longitude,
            dropoff_latitude,
            dropoff_longitude,
            trip_miles,
            pickup_census_tract,
            dropoff_census_tract,
            payment_type,
            company,
            trip_seconds,
            dropoff_community_area,
            tips
          FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
          WHERE (ABS(FARM_FINGERPRINT(unique_key)) / 0x7FFFFFFFFFFFFFFF)
            < 0.000001"""

  # The location of test data
  _TEST_DATA_ROOT = '/tfx-src/tfx/examples/chicago_taxi_pipeline/data/simple'

  @parameterized.named_parameters(
      {
          'testcase_name': 'BQEG',
          'bigquery_query': _BIGQUERY_QUERY,
          'csv_input_location': ''
      }, {
          'testcase_name': 'FBEG',
          'bigquery_query': '',
          'csv_input_location': _TEST_DATA_ROOT
      })
  def testSimpleEnd2EndPipeline(self, bigquery_query, csv_input_location):
    """End-to-End test for a simple pipeline."""
    pipeline_name = '-e2e-test-{}'.format(
        test_utils.random_id())

    components = kubeflow_v2_test_utils.create_pipeline_components(
        pipeline_root=self._pipeline_root(pipeline_name),
        transform_module=self._MODULE_FILE,
        trainer_module=self._MODULE_FILE,
        bigquery_query=bigquery_query,
        csv_input_location=csv_input_location)

    beam_pipeline_args = [
        '--temp_location=' +
        os.path.join(self._pipeline_root(pipeline_name), 'dataflow', 'temp'),
        '--project={}'.format(self._GCP_PROJECT_ID)
    ]

    pipeline = self._create_pipeline(pipeline_name, components,
                                     beam_pipeline_args)

    job_name = self._run_pipeline(pipeline)

    self._check_job_status(job_name)

  def testArtifactValuePlaceholders(self):
    component_instances = (
        kubeflow_v2_test_utils
        .tasks_for_pipeline_with_artifact_value_passing())

    pipeline_name = 'kubeflow-v2-test-artifact-value-{}'.format(
        test_utils.random_id())

    pipeline = self._create_pipeline(
        pipeline_name,
        pipeline_components=component_instances,
    )

    job_name = self._run_pipeline(pipeline)

    self._check_job_status(job_name)


if __name__ == '__main__':
  tf.test.main()
