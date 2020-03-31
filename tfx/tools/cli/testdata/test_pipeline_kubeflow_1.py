# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.tools.cli.e2e import test_utils

# Name of the pipeline
_PIPELINE_NAME = 'chicago_taxi_pipeline_kubeflow'


def _create_pipeline():
  pipeline_name = _PIPELINE_NAME
  pipeline_root = os.path.join(test_utils.get_test_output_dir(), pipeline_name)
  components = test_utils.create_e2e_components(
      pipeline_root,
      test_utils.get_csv_input_location(),
      test_utils.get_transform_module(),
      test_utils.get_trainer_module(),
  )
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
    tfx_image=test_utils.BASE_CONTAINER_IMAGE)
_ = kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
    _create_pipeline())
