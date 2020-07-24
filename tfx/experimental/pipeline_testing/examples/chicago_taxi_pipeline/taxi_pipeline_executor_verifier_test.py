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
"""E2E Tests for taxi pipeline beam with executor verifiers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tfx.experimental.pipeline_testing import executor_verifier_utils
from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_beam
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

class TaxiPipelineExecutorVerifier(tf.test.TestCase):
  """Test for verifying executors using Chicago Taxi dataset."""

  def setUp(self):
    super(ExecutorVerifier, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self._pipeline_name = 'beam_test'
    taxi_root = os.path.dirname(taxi_pipeline_beam.__file__)
    self._data_root = os.path.join(taxi_root, 'data', 'simple')
    self._module_file = os.path.join(taxi_root, 'taxi_utils.py')
    self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')
    # This example assumes that the pipeline outputs are recorded in
    # tfx/experimental/pipeline_testing/examples/chicago_taxi_pipeline/testdata.
    # Feel free to customize this as needed.
    self._record_dir = os.path.join(os.path.dirname(__file__), 'testdata')

  def testExecutorVerifier(self):
    taxi_pipeline = taxi_pipeline_beam._create_pipeline(  # pylint:disable=protected-access
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        module_file=self._module_file,
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._metadata_path,
        beam_pipeline_args=[])

    BeamDagRunner().run(taxi_pipeline)
    pipeline_outputs = executor_verifier_utils.get_pipeline_outputs(
        taxi_pipeline.metadata_connection_config,
        taxi_pipeline.pipeline_info)

    verify_component_ids = ['Transform', 'Trainer', 'Evaluator']
    for component_id in verify_component_ids:
      for key, artifact in pipeline_outputs[component_id].items():
        output_uri = os.path.join(self._record_dir, component_id, key)
        executor_verifier_utils.verify(output_uri, artifact, 0.5)

if __name__ == '__main__':
  tf.test.main()
