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
"""E2E tests for Kubeflow V2 runner with Vertex Pipelines."""

import datetime
import os
from typing import List, Optional

from absl.testing import parameterized

from kfp.v2.google import client

import tensorflow as tf

from tfx.dsl.components.base import base_node
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration import test_utils
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from tfx.orchestration.kubeflow.v2 import test_utils as kubeflow_v2_test_utils

_POLLING_INTERVAL_IN_SECONDS = 60

# TODO(b/184285790): Reduce timeout appropriately.
_MAX_JOB_EXECUTION_TIME = datetime.timedelta(minutes=90)


class KubeflowV2E2ETestCase(kubeflow_v2_test_utils.BaseKubeflowV2Test):
  """Integration tests of Kubeflow V2 runner on managed pipeline."""

  _GCP_PROJECT_ID = os.environ['KFP_E2E_GCP_PROJECT_ID']

  # The GCP region used to call the service.
  _GCP_REGION = os.environ.get('KFP_E2E_GCP_REGION')

  # The GCP bucket to use to write output artifacts.
  _BUCKET_NAME = os.environ['KFP_E2E_BUCKET_NAME']

  def setUp(self):
    super().setUp()
    self._client = client.AIPlatformClient(
        project_id=self._GCP_PROJECT_ID,
        region=self._GCP_REGION)

  def _create_pipeline(
      self,
      pipeline_name: str,
      pipeline_components: List[base_node.BaseNode],
      beam_pipeline_args: Optional[List[str]] = None) -> tfx_pipeline.Pipeline:
    """Creates a pipeline given name and list of components."""
    return tfx_pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=self._pipeline_root(pipeline_name),
        components=pipeline_components,
        beam_pipeline_args=beam_pipeline_args)

  def _run_pipeline(self, pipeline: tfx_pipeline.Pipeline, job_id: str) -> None:
    """Trigger the pipeline execution with a specific job ID."""
    # Ensure cleanup regardless of whether pipeline succeeds or fails.
    self.addCleanup(self._delete_pipeline_output,
                    pipeline.pipeline_info.pipeline_name)

    config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
        default_image=self.container_image)

    _ = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        config=config, output_filename='pipeline.json').run(
            pipeline, write_out=True)

    self._client.create_run_from_job_spec(
        job_spec_path='pipeline.json', job_id=job_id)

  def _check_job_status(self, job_id: str) -> None:
    kubeflow_v2_test_utils.poll_job_status(self._client, job_id,
                                           _MAX_JOB_EXECUTION_TIME,
                                           _POLLING_INTERVAL_IN_SECONDS)


class KubeflowV2E2ETest(KubeflowV2E2ETestCase, parameterized.TestCase):
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

  # The location of test data.
  # This location depends on install path of TFX in the docker image.
  _TEST_DATA_ROOT = '/opt/conda/lib/python3.7/site-packages/tfx/examples/chicago_taxi_pipeline/data/simple'

  @parameterized.named_parameters(
      {
          'testcase_name': 'BQEG',
          'bigquery_query': _BIGQUERY_QUERY,
          'csv_input_location': '',
          'use_custom_dataflow_image': True,
      }, {
          'testcase_name': 'FBEG',
          'bigquery_query': '',
          'csv_input_location': _TEST_DATA_ROOT,
          'use_custom_dataflow_image': False,
      })
  def testSimpleEnd2EndPipeline(self, bigquery_query, csv_input_location,
                                use_custom_dataflow_image):
    """End-to-End test for a simple pipeline."""
    pipeline_name = 'kubeflow-v2-e2e-test-{}'.format(test_utils.random_id())

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
    if use_custom_dataflow_image:
      beam_pipeline_args.extend([
          # TODO(b/171733562): Remove `use_runner_v2` once it is the default for
          # Dataflow.
          '--experiments=use_runner_v2',
          '--worker_harness_container_image=%s' % self.container_image,
      ])

    pipeline = self._create_pipeline(pipeline_name, components,
                                     beam_pipeline_args)

    self._run_pipeline(pipeline, pipeline_name)

    self._check_job_status(pipeline_name)

  def testArtifactValuePlaceholders(self):
    component_instances = (
        kubeflow_v2_test_utils.tasks_for_pipeline_with_artifact_value_passing())

    pipeline_name = 'kubeflow-v2-test-artifact-value-{}'.format(
        test_utils.random_id())

    pipeline = self._create_pipeline(
        pipeline_name,
        pipeline_components=component_instances,
    )

    self._run_pipeline(pipeline, pipeline_name)

    self._check_job_status(pipeline_name)


if __name__ == '__main__':
  tf.test.main()
