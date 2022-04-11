
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
"""Tests for tfx.orchestration.kubeflow.v2.e2e_tests.exit_handler_e2e."""

import os

from kfp.pipeline_spec import pipeline_spec_pb2
import tensorflow as tf
from tfx import v1 as tfx
from tfx.orchestration import test_utils as orchestration_test_utils
from tfx.orchestration.kubeflow.v2 import test_utils
from tfx.orchestration.kubeflow.v2.e2e_tests import base_test_case
from tfx.orchestration.test_pipelines import custom_exit_handler
from tfx.utils import io_utils

from google.protobuf import json_format


# The location of test data.
# This location depends on install path of TFX in the docker image.
_TEST_DATA_ROOT = '/opt/conda/lib/python3.7/site-packages/tfx/examples/chicago_taxi_pipeline/data/simple'

_success_file_name = 'success_final_status.txt'


class ExitHandlerE2ETest(base_test_case.BaseKubeflowV2Test):

  # The GCP bucket to use to write output artifacts.
  _BUCKET_NAME = os.environ.get('KFP_E2E_BUCKET_NAME')

  def testExitHandlerPipelineSuccess(self):
    """End-to-End test for a successful pipeline with exit handler."""
    pipeline_name = 'kubeflow-v2-exit-handler-test-{}'.format(
        orchestration_test_utils.random_id())

    components = test_utils.simple_pipeline_components(_TEST_DATA_ROOT)

    beam_pipeline_args = [
        '--temp_location=' +
        os.path.join(self._pipeline_root(pipeline_name), 'dataflow', 'temp'),
        '--project={}'.format(self._GCP_PROJECT_ID)
    ]

    pipeline = self._create_pipeline(pipeline_name, components,
                                     beam_pipeline_args)

    output_file_dir = os.path.join(
        self._pipeline_root(pipeline_name), _success_file_name)

    exit_handler = custom_exit_handler.test_exit_handler(
        final_status=tfx.orchestration.experimental.FinalStatusStr(),
        file_dir=output_file_dir)

    self._run_pipeline(pipeline=pipeline, exit_handler=exit_handler)

    # verify execution results
    actual_final_status_str = io_utils.read_string_file(output_file_dir)
    expected_successful_final_status_str = """
      {
        "state":"SUCCEEDED",
        "error":{},
        "pipeline_task_name":"_tfx_dag"
      }
    """

    expected_successful_final_status = (pipeline_spec_pb2
                                        .PipelineTaskFinalStatus())
    json_format.Parse(expected_successful_final_status_str,
                      expected_successful_final_status)

    actual_final_status = pipeline_spec_pb2.PipelineTaskFinalStatus()
    json_format.Parse(actual_final_status_str, actual_final_status)

    self.assertProtoPartiallyEquals(expected_successful_final_status,
                                    actual_final_status,
                                    ignored_fields=[
                                        'pipeline_job_resource_name'])


if __name__ == '__main__':
  tf.test.main()
