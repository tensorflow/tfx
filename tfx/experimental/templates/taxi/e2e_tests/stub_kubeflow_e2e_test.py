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
"""E2E test using kubeflow orchestrator with stub executors for taxi template."""

import os

from absl import logging
import kfp
import tensorflow as tf

from tfx.experimental.pipeline_testing import pipeline_recorder_utils
from tfx.experimental.templates.taxi.e2e_tests import kfp_test_utils
from tfx.orchestration import test_utils as orchestration_test_utils
from tfx.utils import io_utils


class StubKubeflowE2ETest(kfp_test_utils.KubeflowBaseEndToEndTest):

  _POLLING_INTERVAL_IN_SECONDS = 10
  _MAX_POLLING_COUNT = 20 * 6  # 20 min.

  _DATA_DIRECTORY_NAME = 'template_data'

  # The following environment variables need to be set prior to calling the test
  # in this file. All variables are required and do not have a default.

  # The base container image name to use when building the image used in tests.
  _BASE_CONTAINER_IMAGE = os.environ['KFP_E2E_BASE_CONTAINER_IMAGE']

  # The src path to use to build docker image
  _REPO_BASE = os.environ['KFP_E2E_SRC']

  # The project id to use to run tests.
  _GCP_PROJECT_ID = os.environ['KFP_E2E_GCP_PROJECT_ID']

  # The GCP bucket to use to write output artifacts.
  # This default bucket name is valid for KFP marketplace deployment since KFP
  # version 0.5.0.
  _BUCKET_NAME = _GCP_PROJECT_ID + '-kubeflowpipelines-default'

  _KFP_E2E_TEST_FORWARDING_PORT_BEGIN = 7230
  _KFP_E2E_TEST_FORWARDING_PORT_END = 7235
  _MAX_ATTEMPTS = 5

  def setUp(self):
    super().setUp()
    random_id = orchestration_test_utils.random_id()
    self._pipeline_name = 'taxi-template-kubeflow-e2e-test-' + random_id
    logging.info('Pipeline: %s', self._pipeline_name)
    self._endpoint = self._get_endpoint()
    self._kfp_client = kfp.Client(host=self._endpoint)
    logging.info('ENDPOINT: %s', self._endpoint)

    self._base_container_image = '{}:{}'.format(self._BASE_CONTAINER_IMAGE,
                                                random_id)
    self._target_container_image = 'gcr.io/{}/{}:{}'.format(
        self._GCP_PROJECT_ID, 'taxi-template-kubeflow-e2e-test', random_id)
    self._record_dir = "gs://{}/testdata".format(self._BUCKET_NAME)
    self._port_forwarding_process = self._setup_mlmd_port_forward()
    self._prepare_base_container_image()
    self._prepare_skaffold()

  def tearDown(self):
    super(StubKubeflowE2ETest, self).tearDown()
    logging.info('Killing the GRPC port-forwarding process.')
    self._port_forwarding_process.kill()
    io_utils.delete_dir(self._record_dir)
    self._cleanup_kfp()

  def _cleanup_kfp(self):
    self._cleanup_with_retry(self._delete_base_container_image)
    self._cleanup_with_retry(self._delete_target_container_image)
    self._cleanup_with_retry(self._delete_pipeline)
    self._cleanup_with_retry(self._delete_pipeline_data)
    self._cleanup_with_retry(self._delete_runs)

  def testPipeline(self):
    self._copy_template()
    os.environ['KUBEFLOW_HOME'] = os.path.join(self._temp_dir, 'kubeflow')

    # Uncomment all variables in config.
    self._uncomment_multiline_variables(
        os.path.join('pipeline', 'configs.py'), [
            'GOOGLE_CLOUD_REGION',
            'BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS',
            'BIG_QUERY_QUERY', 'DATAFLOW_BEAM_PIPELINE_ARGS',
            'GCP_AI_PLATFORM_TRAINING_ARGS', 'GCP_AI_PLATFORM_SERVING_ARGS'
        ])

    # Prepare data
    self._prepare_data()
    self._replace_file_content('kubeflow_dag_runner.py', [
        ('DATA_PATH = \'gs://{}/tfx-template/data/\'.format(configs.GCS_BUCKET_NAME)',  # pylint: disable=line-too-long
         'DATA_PATH = \'gs://{{}}/{}/{}\'.format(configs.GCS_BUCKET_NAME)'
         .format(self._DATA_DIRECTORY_NAME, self._pipeline_name)),
    ])

    # Create a pipeline with all components.
    updated_pipeline_file = self._add_all_components()
    logging.info('Updated %s to add all components to the pipeline.',
                 updated_pipeline_file)

    self._create_pipeline()
    self._run_pipeline()
    self._check_telemetry_label()
    logging.info("Successfully ran the pipeline.")
    # Record pipeline outputs to self._record_dir
    pipeline_recorder_utils.record_pipeline(
        output_dir=self._record_dir,
        metadata_db_uri=None,
        host='localhost',
        port=self._port,
        pipeline_name=self._pipeline_name,
        run_id=None)
    self.assertTrue(tf.io.gfile.exists(self._record_dir))
    logging.info("Pipeline has been recorded.")
    # Enable stub executors
    self._uncomment_multiline_variables('kubeflow_dag_runner.py',
                                        ['supported_launcher_classes'])

    # Update the pipeline to use stub executors.
    self._update_pipeline()
    self._run_pipeline()


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
