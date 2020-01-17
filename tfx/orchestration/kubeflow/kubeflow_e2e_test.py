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
"""End to end tests for Kubeflow-based orchestrator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys

import tensorflow as tf

from tfx.orchestration.kubeflow import test_utils


class KubeflowEndToEndTest(test_utils.BaseKubeflowTest):

  def testSimpleEnd2EndPipeline(self):
    """End-to-End test for simple pipeline."""
    pipeline_name = 'kubeflow-e2e-test-{}'.format(self._random_id())
    components = test_utils.create_e2e_components(
        self._pipeline_root(pipeline_name),
        self._data_root,
        self._transform_module,
        self._trainer_module,
    )
    pipeline = self._create_pipeline(pipeline_name, components)

    self._compile_and_run_pipeline(pipeline)


if __name__ == '__main__':
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  tf.test.main()
