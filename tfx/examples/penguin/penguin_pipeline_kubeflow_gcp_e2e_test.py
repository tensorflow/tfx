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
"""End-to-end tests for tfx.examples.penguin.penguin_pipeline_kubeflow_gcp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import tensorflow as tf

from tfx.examples.penguin import penguin_pipeline_kubeflow_gcp
from tfx.orchestration import test_utils
from tfx.orchestration.kubeflow import test_utils as kubeflow_test_utils


class PenguinPipelineKubeflowGcpEndToEndTest(
    kubeflow_test_utils.BaseKubeflowTest):

  def testEndToEndPipelineRun(self):
    """End-to-end test for pipeline with Tuner."""
    pipeline_name = 'penguin-kubeflow-e2e-test-parameter-{}'.format(
        test_utils.random_id())
    pipeline = penguin_pipeline_kubeflow_gcp.create_pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=penguin_pipeline_kubeflow_gcp._pipeline_root,
        enable_cache=True,
        beam_pipeline_args=penguin_pipeline_kubeflow_gcp._beam_pipeline_args)

    parameters = {
        'pipeline-root': self._pipeline_root(pipeline_name),
        'transform-module': self._transform_module,
        'trainer-module': self._trainer_module,
        'data-root': self._data_root,
        'train-steps': 10,
        'eval-steps': 5,
    }

    self._compile_and_run_pipeline(pipeline=pipeline, parameters=parameters)

    # TODO(wuamy) Call AIP Training Service and Vizier service?

if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  tf.test.main()
