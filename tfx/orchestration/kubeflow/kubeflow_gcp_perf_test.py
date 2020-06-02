# Lint as: python2, python3
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
"""Integration tests for TFX-on-KFP and GCP services."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import subprocess
import time
from typing import Text

from absl import logging
import kfp
import tensorflow as tf

from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_kubeflow_gcp
from tfx.orchestration import data_types
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.orchestration.kubeflow import test_utils

# The endpoint of the KFP instance.
# This test fixture assumes an established KFP instance authenticated via
# inverse proxy.
_KFP_ENDPOINT = os.environ['KFP_E2E_ENDPOINT']

# The namespace where KFP is deployed.
_KFP_NAMESPACE = 'kubeflow'

# Timeout for a single pipeline run. Set to 6 hours.
# TODO(b/158009615): Tune this timeout to align with our observation.
# Note: the Chicago Taxi dataset is a dataset growing with time. The 6 hour
# timeout here was calibrated according to our empirical study in
# b/150222976. This might need to be adjusted occasionally.
_TIME_OUT_SECONDS = 6 * 3600

# The base container image name to use when building the image used in tests.
_BASE_CONTAINER_IMAGE = os.environ['KFP_E2E_BASE_CONTAINER_IMAGE']

# The project id to use to run tests.
_GCP_PROJECT_ID = os.environ['KFP_E2E_GCP_PROJECT_ID']

# The GCP region in which the end-to-end test is run.
_GCP_REGION = os.environ['KFP_E2E_GCP_REGION']

# The GCP bucket to use to write output artifacts.
_BUCKET_NAME = os.environ['KFP_E2E_BUCKET_NAME']

# Various execution status of a KFP pipeline.
_KFP_RUNNING_STATUS = 'running'
_KFP_SUCCESS_STATUS = 'succeeded'
_KFP_FAIL_STATUS = 'failed'
_KFP_SKIPPED_STATUS = 'skipped'
_KFP_ERROR_STATUS = 'error'

_KFP_FINAL_STATUS = frozenset((_KFP_SUCCESS_STATUS, _KFP_FAIL_STATUS,
                               _KFP_SKIPPED_STATUS, _KFP_ERROR_STATUS))

# The location of test user module file.
# It is retrieved from inside the container subject to testing.
_MODULE_FILE = '/tfx-src/tfx/examples/chicago_taxi_pipeline/taxi_utils.py'

# Parameterize worker type/count for easily ramping up the pipeline scale.
_WORKER_COUNT = data_types.RuntimeParameter(
    name='worker_count',
    default=2,
    ptype=int,
)

_WORKER_TYPE = data_types.RuntimeParameter(
    name='worker_type',
    default='standard',
    ptype=str,
)

# Parameterize parameter server count for easily ramping up the scale.
_PARAMETER_SERVER_COUNT = data_types.RuntimeParameter(
    name='parameter_server_count',
    default=1,
    ptype=int,
)

_AI_PLATFORM_SERVING_ARGS = {
    'model_name': 'chicago_taxi',
    'project_id': _GCP_PROJECT_ID,
    'regions': [_GCP_REGION],
}

_BEAM_PIPELINE_ARGS = [
    '--runner=DataflowRunner', '--experiments=shuffle_mode=auto',
    '--project=' + _GCP_PROJECT_ID,
    '--temp_location=gs://' + os.path.join(_BUCKET_NAME, 'dataflow', 'tmp'),
    '--region=' + _GCP_REGION, '--disk_size_gb=50', '--no_use_public_ips'
]


class KubeflowGcpPerfTest(test_utils.BaseKubeflowTest):

  @classmethod
  def setUpClass(cls):
    super(test_utils.BaseKubeflowTest, cls).setUpClass()
    # Create a container image for use by test pipelines.
    base_container_image = _BASE_CONTAINER_IMAGE

    cls._container_image = '{}:{}'.format(base_container_image,
                                          cls._random_id())
    cls._build_and_push_docker_image(cls._container_image)

  @classmethod
  def tearDownClass(cls):
    super(test_utils.BaseKubeflowTest, cls).tearDownClass()

  def _get_workflow_name(self, pipeline_name: Text) -> Text:
    """Gets the Argo workflow name using pipeline name."""
    get_workflow_name_command = (
        'argo --namespace %s list | grep -o "%s[^ ]*"' %
        (_KFP_NAMESPACE, pipeline_name))
    # Need to explicitly decode because the test fixture is running on
    # Python 3.5. Also need to remove the new line at the end of the string.
    return subprocess.check_output(
        get_workflow_name_command, shell=True).decode('utf-8')[:-1]

  def _get_workflow_log(self, pipeline_name: Text) -> Text:
    """Gets the workflow log for all the pods using pipeline name."""
    get_workflow_log_command = [
        'argo', '--namespace', _KFP_NAMESPACE, 'logs', '-w',
        self._get_workflow_name(pipeline_name)
    ]
    # Need to explicitly decode because the test fixture is running on
    # Python 3.5.
    return subprocess.check_output(get_workflow_log_command).decode('utf-8')

  # TODO(jxzheng): workaround for 1hr timeout limit in kfp.Client().
  # This should be changed after
  # https://github.com/kubeflow/pipelines/issues/3630 is fixed.
  # Currently gcloud authentication token has a one hour expiration by default
  # but kfp.Client() does not have a refreshing mechanism in place. This
  # causes failure when attempting to get running status for a long pipeline
  # execution (> 1 hour).
  # Instead of implementing a full-fledged authentication refreshing mechanism
  # here, we chose re-creating kfp.Client() frequently to make sure the
  # authentication does not expire. This is based on the fact that kfp.Client()
  # is very light-weight.
  # See more details at
  # https://github.com/kubeflow/pipelines/issues/3630
  def _assert_successful_run_completion(self, host: Text, run_id: Text,
                                        pipeline_name: Text, timeout: int):
    """Waits and asserts a successful KFP pipeline execution.

    Args:
      host: the endpoint of the KFP deployment.
      run_id: the run ID of the execution, can be obtained from the respoonse
        when submitting the pipeline.
      pipeline_name: the name of the pipeline under test.
      timeout: maximal waiting time for this execution, in seconds.

    Raises:
      RuntimeError: when timeout exceeds after waiting for specified duration.
    """
    status = None
    start_time = datetime.datetime.now()
    while True:
      client = kfp.Client(host=host)
      get_run_response = client._run_api.get_run(run_id=run_id)

      if (get_run_response and get_run_response.run and
          get_run_response.run.status and
          get_run_response.run.status.lower() in _KFP_FINAL_STATUS):
        # Break because final status is reached.
        status = get_run_response.run.status
        break
      elif (datetime.datetime.now() - start_time).seconds > timeout:
        # Timeout.
        raise RuntimeError('Waiting for run timeout at %s' %
                           datetime.datetime.now().strftime('%H:%M:%S'))
      else:
        logging.info('Waiting for the job to complete...')
        time.sleep(10)

    workflow_log = self._get_workflow_log(pipeline_name)

    self.assertEqual(
        status.lower(), _KFP_SUCCESS_STATUS,
        'Pipeline %s failed to complete successfully: %s' %
        (pipeline_name, workflow_log))

  def _compile_and_run_pipeline(self, pipeline: tfx_pipeline.Pipeline,
                                **kwargs):
    """Compiles and runs a KFP pipeline.

    In this method, provided TFX pipeline will be submitted via kfp.Client()
    instead of from Argo.

    Args:
      pipeline: The logical pipeline to run.
      **kwargs: Key-value pairs of runtime paramters passed to the pipeline
        execution.
    """
    client = kfp.Client(host=_KFP_ENDPOINT)

    pipeline_name = pipeline.pipeline_info.pipeline_name
    config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=self._get_kubeflow_metadata_config(),
        tfx_image=self._container_image)
    kubeflow_dag_runner.KubeflowDagRunner(config=config).run(pipeline)

    file_path = os.path.join(self._test_dir, '{}.tar.gz'.format(pipeline_name))
    self.assertTrue(tf.io.gfile.exists(file_path))

    run_result = client.create_run_from_pipeline_package(
        pipeline_file=file_path, arguments=kwargs)
    run_id = run_result.run_id

    self._assert_successful_run_completion(
        host=_KFP_ENDPOINT,
        run_id=run_id,
        pipeline_name=pipeline_name,
        timeout=_TIME_OUT_SECONDS)

  def testFullTaxiGcpPipeline(self):
    pipeline_name = 'gcp-perf-test-full-e2e-test-{}'.format(self._random_id())

    # Custom CAIP training job using a testing image.
    ai_platform_training_args = {
        'project': _GCP_PROJECT_ID,
        'region': _GCP_REGION,
        'scaleTier': 'CUSTOM',
        'masterType': 'large_model',
        'masterConfig': {
            'imageUri': self._container_image
        },
        'workerType': _WORKER_TYPE,
        'parameterServerType': 'standard',
        'workerCount': _WORKER_COUNT,
        'parameterServerCount': _PARAMETER_SERVER_COUNT
    }

    pipeline = taxi_pipeline_kubeflow_gcp.create_pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=self._pipeline_root(pipeline_name),
        module_file=_MODULE_FILE,
        ai_platform_training_args=ai_platform_training_args,
        ai_platform_serving_args=_AI_PLATFORM_SERVING_ARGS,
        beam_pipeline_args=_BEAM_PIPELINE_ARGS)
    self._compile_and_run_pipeline(
        pipeline=pipeline,
        query_sample_rate=1,
        # (1M * batch_size=200) / 200M records ~ 1 epoch
        train_steps=1000000,
        eval_steps=10000,
        worker_count=20,
        parameter_server_count=3,
    )


if __name__ == '__main__':
  tf.test.main()
