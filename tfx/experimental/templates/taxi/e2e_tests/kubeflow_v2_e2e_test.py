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
"""E2E test using Kubeflow V2 runner for taxi template."""

import os
from typing import List, Text
import urllib.request

from absl import logging
from click import testing as click_testing
from kfp.pipeline_spec import pipeline_spec_pb2
import tensorflow as tf
from tfx.experimental.templates.taxi.e2e_tests import test_utils
from tfx.orchestration import test_utils as orchestration_test_utils
from tfx.orchestration.kubeflow.v2 import test_utils as kubeflow_test_utils
from tfx.tools.cli.kubeflow.v2 import cli_main
from tfx.utils import io_utils

from google.cloud import storage


class TaxiTemplateKubeflowV2E2ETest(test_utils.BaseEndToEndTest,
                                    kubeflow_test_utils.BaseKubeflowV2Test):

  _DATA_DIRECTORY_NAME = 'template_data'

  def setUp(self):
    super().setUp()
    random_id = orchestration_test_utils.random_id()
    self._target_container_image = 'gcr.io/{}/{}:{}'.format(
        self._GCP_PROJECT_ID, 'taxi-template-kubeflow_v2-e2e-test', random_id)
    # Overriding the pipeline name to
    self._pipeline_name = 'taxi_template_kubeflow_v2_e2e_test_{}'.format(
        random_id)
    self._prepare_skaffold()

  def _prepare_skaffold(self):
    self._skaffold = os.path.join(self._temp_dir, 'skaffold')
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/skaffold/releases/latest/'
        'skaffold-linux-amd64', self._skaffold)
    os.chmod(self._skaffold, 0o775)

  def _run_cli(self, args: List[Text]) -> click_testing.Result:
    logging.info('Running cli: %s', args)
    result = self._cli_runner.invoke(cli_main.cli_group, args)
    logging.info('%s', result.output)
    if result.exit_code != 0:
      logging.error('Exit code from cli: %d, exception:%s', result.exit_code,
                    result.exception)
      logging.error('Traceback: %s', result.exc_info)

    return result

  def _create_pipeline(self):
    result = self._run_cli([
        'kubeflow_v2',
        'pipeline',
        'create',
        '--pipeline-path',
        'kubeflow_v2_dag_runner.py',
        '--build-base-image',
        self._CONTAINER_IMAGE,
        '--build-target-image',
        self._target_container_image,
        '--skaffold-cmd',
        self._skaffold,
    ])
    self.assertEqual(0, result.exit_code)
    self.addCleanup(self._delete_pipeline)

  def _delete_pipeline(self):
    self._run_cli([
        'kubeflow_v2', 'pipeline', 'delete', '--pipeline_name',
        self._pipeline_name
    ])

  def _prepare_data(self):
    """Uploads the csv data from local to GCS location."""
    client = storage.Client(project=self._GCP_PROJECT_ID)
    bucket = client.bucket(self._BUCKET_NAME)
    blob = bucket.blob('{}/{}/data.csv'.format(self._DATA_DIRECTORY_NAME,
                                               self._pipeline_name))
    blob.upload_from_filename('data/data.csv')
    self.addCleanup(self._delete_data)

  def _delete_data(self):
    """Deletes the uploaded csv data from GCS location."""
    client = storage.Client(project=self._GCP_PROJECT_ID)
    bucket = client.bucket(self._BUCKET_NAME)
    blob = bucket.blob('{}/{}/data.csv'.format(self._DATA_DIRECTORY_NAME,
                                               self._pipeline_name))
    blob.delete()

  def testPipeline(self):
    self._copyTemplate()

    # Uncomment all variables in config.
    self._uncommentMultiLineVariables(
        os.path.join('pipeline', 'configs.py'), [
            'GOOGLE_CLOUD_REGION',
            'BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS',
            'BIG_QUERY_QUERY', 'DATAFLOW_BEAM_PIPELINE_ARGS',
            'GCP_AI_PLATFORM_TRAINING_ARGS', 'GCP_AI_PLATFORM_SERVING_ARGS'
        ])

    # Prepare data.
    self._prepare_data()
    self._replaceFileContent('kubeflow_v2_dag_runner.py', [
        ('_DATA_PATH = \'gs://{}/tfx-template/data/\'.'
         'format(configs.GCS_BUCKET_NAME)',
         '_DATA_PATH = \'gs://{{}}/{}/{}\'.format(configs.GCS_BUCKET_NAME)'
         .format(self._DATA_DIRECTORY_NAME, self._pipeline_name)),
    ])

    # Create a pipeline with only one component.
    self._create_pipeline()

    # Extract the compiled pipeline spec.
    kubeflow_v2_pb = pipeline_spec_pb2.PipelineJob()
    io_utils.parse_json_file(
        file_name=os.path.join(os.getcwd(), 'pipeline.json'),
        message=kubeflow_v2_pb)
    # There should be one step in the compiled pipeline.
    self.assertLen(kubeflow_v2_pb.pipeline_spec['tasks'], 1)

    # TODO(b/159923274): Add the full workflow of Kubeflow V2 pipelines
    # template.
    # This includes uncommenting all the components end-to-end, and submitting
    # the pipeline for execution.


if __name__ == '__main__':
  tf.test.main()
