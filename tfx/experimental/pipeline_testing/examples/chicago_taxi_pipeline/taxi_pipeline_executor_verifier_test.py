# Lint as: python3
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

from absl import logging
from typing import Text
import tensorflow as tf

from tfx.experimental.pipeline_testing import executor_verifier_utils
from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_beam
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

class TaxiPipelineExecutorVerifier(tf.test.TestCase):
  """Test for verifying executors using Chicago Taxi dataset."""

  def setUp(self):
    super(TaxiPipelineExecutorVerifier, self).setUp()
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
    # self._record_dir = os.path.join(os.path.dirname(__file__), 'testdata')
    self._record_dir = os.path.join('/Users/sujipark/tfx', 'testdata')

    self.taxi_pipeline = taxi_pipeline_beam._create_pipeline(  # pylint:disable=protected-access
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        module_file=self._module_file,
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._metadata_path,
        beam_pipeline_args=[])

    BeamDagRunner().run(self.taxi_pipeline)
    self.pipeline_outputs = executor_verifier_utils.get_pipeline_outputs(
        self.taxi_pipeline.metadata_connection_config,
        self.taxi_pipeline.pipeline_info)

    self._verifier_map = {'Trainer': self._verify_trainer,
                          'Evaluator': self._verify_trainer,
                          'CsvExampleGen': self._verify_examples,
                          'SchemaGen': self._verify_schema,
                          'ExampleValidator': self._verify_validator}

  def _verify_file_path(self, output_uri: Text, artifact_uri: Text):
    self.assertTrue(
        executor_verifier_utils.verify_file_dir(output_uri, artifact_uri))

  def _verify_evaluator(self, output_uri: Text, expected_uri: Text):
    self.assertTrue(executor_verifier_utils.compare_eval_results(
        output_uri,
        expected_uri, .5))

  def _verify_schema(self, output_uri: Text, expected_uri: Text):
    self.assertTrue(
        executor_verifier_utils.compare_file_sizes(output_uri,
                                                   expected_uri, .5))

  def _verify_examples(self, output_uri: Text, expected_uri: Text):
    self.assertTrue(
        executor_verifier_utils.compare_file_sizes(output_uri,
                                                   expected_uri, .5))

  def _verify_trainer(self, output_uri: Text, expected_uri: Text):
    self.assertTrue(
        executor_verifier_utils.compare_model_file_sizes(output_uri,
                                                   expected_uri, .5))

  def _verify_validator(self, output_uri: Text, expected_uri: Text):
    self.assertTrue(
        executor_verifier_utils.compare_anomalies(output_uri,
                                                  expected_uri))

  def testExecutorVerifier(self):
    # Calls verifier for artifacts
    model_resolver_id = 'ResolverNode.latest_blessed_model_resolver'
    # Components to verify
    verify_component_ids = [component.id
                            for component in self.taxi_pipeline.components
                            if component.id != model_resolver_id]

    for component_id in verify_component_ids:
      for key, artifact_dict in self.pipeline_outputs[component_id].items():
        for idx, artifact in artifact_dict.items():
          logging.info("Verifying {}".format(component_id))
          recorded_uri = os.path.join(self._record_dir, component_id,
                                      key, str(idx))
          self._verifier_map.get(component_id,
                                 self._verify_file_path)(artifact.uri,
                                                         recorded_uri)

if __name__ == '__main__':
  tf.test.main()
