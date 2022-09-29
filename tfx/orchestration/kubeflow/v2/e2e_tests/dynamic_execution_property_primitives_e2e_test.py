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

import os

import tensorflow as tf
from tfx.orchestration import test_utils as orchestration_test_utils
from tfx.orchestration.kubeflow.v2 import test_utils
from tfx.orchestration.kubeflow.v2.e2e_tests import base_test_case


class DynamicExecutionPropertyPrimitivesE2ETest(
    base_test_case.BaseKubeflowV2Test):

  _BUCKET_NAME = os.environ.get('KFP_E2E_BUCKET_NAME')

  def testDynamicExecutionPropertySuccess(self):

    # Create test ID and pipeline name
    test_id = orchestration_test_utils.random_id()
    pipeline_name = 'vertex-dynamic-execution-e2e-test-{}'.format(test_id)

    # Create two step pipeline with custom components
    upstream = test_utils.custom_component(input_parameter='input_parameter')
    upstream.id = 'upstream_task'
    downstream = test_utils.custom_component(
        input_parameter=upstream.outputs['output_value_artifact'].future()
        [0].value)
    downstream.id = 'downstream_task'
    components = [upstream, downstream]

    # Create and run the pipeline; assert success
    pipeline = self._create_pipeline(pipeline_name, components)
    self._run_pipeline(pipeline=pipeline)
    assert downstream.outputs['output_value_artifact'] == 'input_parameter'


if __name__ == '__main__':
  tf.test.main()
