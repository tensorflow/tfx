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

import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma

from tensorflow_metadata.proto.v0 import anomalies_pb2
from tfx.utils import io_utils
from tfx.experimental.pipeline_testing.examples.chicago_taxi_pipeline import taxi_pipeline_verifier_utils
from tfx.examples.chicago_taxi_pipeline import taxi_pipeline_beam
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

class TaxiPipelineExecutorVerifierTest(tf.test.TestCase):

  def setUp(self):
    super(TaxiPipelineExecutorVerifierTest, self).setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self._pipeline_name = 'beam_test'
    taxi_root = os.path.join(os.environ['HOME'],
                             "tfx/tfx/examples/chicago_taxi_pipeline")
    self._data_root = os.path.join(taxi_root, 'data', 'simple')
    self._module_file = os.path.join(taxi_root, 'taxi_utils.py')
    self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')
    self._threshold = 0.5
    self._record_dir = os.path.join(os.environ['HOME'],
                                    'tfx/tfx/experimental/pipeline_testing/',
                                    'examples/chicago_taxi_pipeline/testdata1')

  def verify_trainer(self, output_dict):
    """compares two model files"""
    absl.logging.info("verifying Trainer")
    model_artifact = output_dict['model']
    model_uri = model_artifact.uri

    path = os.path.join(self._record_dir, 'Trainer', 'model')
    taxi_pipeline_verifier_utils.compare_model_file_sizes(
        model_uri,
        path,
        self._threshold)

  def verify_evaluator(self, output_dict):
    """compares two evaluation proto files"""
    absl.logging.info("verifying Evaluator")
    eval_result = tfma.load_eval_result(output_dict['evaluation'].uri)
    expected_eval_result = tfma.load_eval_result(os.path.join(self._record_dir,
                                                              'Evaluator',
                                                              'evaluation'))
    taxi_pipeline_verifier_utils.compare_eval_results(eval_result,
                                                      expected_eval_result,
                                                      self._threshold)

  def verify_validator(self, output_dict):
    """compares two validation proto files"""
    absl.logging.info("verifying Validator")
    anomalies = io_utils.parse_pbtxt_file(
        os.path.join(output_dict['anomalies'].uri, 'anomalies.pbtxt'),
        anomalies_pb2.Anomalies())
    expected_anomalies = io_utils.parse_pbtxt_file(
        os.path.join(self._record_dir,
                     'ExampleValidator',
                     'anomalies',
                     'anomalies.pbtxt'),
        anomalies_pb2.Anomalies())
    self.assertEqual(expected_anomalies.anomaly_info, anomalies.anomaly_info)

  def testExecutorVerifier(self):
    taxi_pipeline = taxi_pipeline_beam._create_pipeline(  # pylint:disable=protected-access, unexpected-keyword-arg
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        module_file=self._module_file,
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._metadata_path,
        beam_pipeline_args=[])

    BeamDagRunner().run(taxi_pipeline)
    component_output_map = \
        taxi_pipeline_verifier_utils.get_component_output_map(
            taxi_pipeline.metadata_connection_config,
            taxi_pipeline.pipeline_info)
    self.verify_validator(component_output_map['ExampleValidator'])
    self.verify_trainer(component_output_map['Trainer'])
    self.verify_evaluator(component_output_map['Evaluator'])

if __name__ == '__main__':
  tf.test.main()
