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

from tfx.orchestration import metadata
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
    self._assert_infra_validator_passed(pipeline_name)

  def testPrimitiveEnd2EndPipeline(self):
    """End-to-End test for primitive artifacts passing."""
    pipeline_name = 'kubeflow-primitive-e2e-test-{}'.format(self._random_id())
    components = test_utils.create_primitive_type_components(pipeline_name)
    # Test that the pipeline can be executed successfully.
    pipeline = self._create_pipeline(pipeline_name, components)
    self._compile_and_run_pipeline(
        pipeline=pipeline, workflow_name=pipeline_name + '-run-1')
    # Test if the correct value has been passed.
    str_artifacts = self._get_artifacts_with_type_and_pipeline(
        type_name='String', pipeline_name=pipeline_name)
    # There should be exactly one string artifact.
    self.assertEqual(1, len(str_artifacts))
    self.assertEqual(
        self._get_value_of_string_artifact(str_artifacts[0]),
        'hello %s\n' % pipeline_name)
    # Test caching.
    self._compile_and_run_pipeline(
        pipeline=pipeline, workflow_name=pipeline_name + '-run-2')
    cached_execution = self._get_executions_by_pipeline_name_and_state(
        pipeline_name=pipeline_name, state=metadata.EXECUTION_STATE_CACHED)
    self.assertEqual(2, len(cached_execution))


if __name__ == '__main__':
  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  tf.test.main()
