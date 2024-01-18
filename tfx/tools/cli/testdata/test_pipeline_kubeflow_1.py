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
"""Test pipeline for Kubeflow."""

import os

from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.tools.cli.e2e import test_utils

from ml_metadata.proto import metadata_store_pb2

# The base container image name to use when building the image used in tests.
_BASE_CONTAINER_IMAGE = os.environ['KFP_E2E_BASE_CONTAINER_IMAGE']

# The GCP bucket to use to write output artifacts.
_BUCKET_NAME = os.environ['KFP_E2E_BUCKET_NAME']

# The location of test data. The input files are copied to a test-local
# location for each invocation, and cleaned up at the end of test.
_TESTDATA_ROOT = os.environ['KFP_E2E_TEST_DATA_ROOT']


def _get_test_output_dir():
  return 'gs://{}/test_output'.format(_BUCKET_NAME)


def _get_csv_input_location():
  return os.path.join(_TESTDATA_ROOT, 'external', 'csv')


# Name of the pipeline
_PIPELINE_NAME = 'chicago_taxi_pipeline_kubeflow'


def _create_pipeline():
  pipeline_name = _PIPELINE_NAME
  pipeline_root = os.path.join(_get_test_output_dir(), pipeline_name)
  components = test_utils.create_e2e_components(_get_csv_input_location())
  return tfx_pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      metadata_connection_config=metadata_store_pb2.ConnectionConfig(),
      components=components[:2],  # Run two components only to reduce overhead.
      log_root='/var/tmp/tfx/logs',
      additional_pipeline_args={
          'WORKFLOW_ID': pipeline_name,
      },
  )


runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
    kubeflow_metadata_config=kubeflow_dag_runner
    .get_default_kubeflow_metadata_config(),
    tfx_image=_BASE_CONTAINER_IMAGE)
_ = kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
    _create_pipeline())
