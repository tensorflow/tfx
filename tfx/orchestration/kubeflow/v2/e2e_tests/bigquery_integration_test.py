# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.kubeflow.v2.e2e_tests.bigquery_integration."""

import os
from unittest import mock

import tensorflow as tf
from tfx.dsl.components.base import base_component
from tfx.orchestration import test_utils
from tfx.orchestration.kubeflow.v2 import test_utils as kubeflow_v2_test_utils
from tfx.orchestration.kubeflow.v2.e2e_tests import base_test_case

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


class BigqueryIntegrationTest(base_test_case.BaseKubeflowV2Test):

  @mock.patch.object(base_component.BaseComponent, '_resolve_pip_dependencies')
  def testSimpleEnd2EndPipeline(self, moke_resolve_dependencies):
    """End-to-End test for a simple pipeline."""
    moke_resolve_dependencies.return_value = None
    pipeline_name = 'kubeflow-v2-bqeg-test-{}'.format(test_utils.random_id())

    components = kubeflow_v2_test_utils.create_pipeline_components(
        pipeline_root=self._pipeline_root(pipeline_name),
        transform_module=self._MODULE_FILE,
        trainer_module=self._MODULE_FILE,
        bigquery_query=_BIGQUERY_QUERY)

    beam_pipeline_args = [
        '--temp_location=' +
        os.path.join(self._pipeline_root(pipeline_name), 'dataflow', 'temp'),
        '--project={}'.format(self._GCP_PROJECT_ID),
        # TODO(b/171733562): Remove `use_runner_v2` once it is the default for
        # Dataflow.
        '--experiments=use_runner_v2',
        '--worker_harness_container_image=%s' % self.container_image,
    ]

    pipeline = self._create_pipeline(pipeline_name, components,
                                     beam_pipeline_args)

    self._run_pipeline(pipeline)
    moke_resolve_dependencies.assert_called()


if __name__ == '__main__':
  tf.test.main()
