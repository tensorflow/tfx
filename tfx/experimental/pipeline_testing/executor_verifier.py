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
"""Recording pipeline from MLMD metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import absl
import os
import types
from typing import Text

from tensorflow_metadata.proto.v0 import anomalies_pb2
import tensorflow_model_analysis as tfma
from tfx.utils import io_utils
from tfx.experimental.pipeline_testing import verifier_utils

class ExecutorVerifier(object):
  """ExecutorVerifier for verifying executor outputs"""
  def __init__(self, record_dir,
               pipeline_info,
               metadata_connection_config,
               threshold=0.5):
    """
    threshold: between 0 and 1
    components_id: components to verify
    pipeline_info: for current pipeline
    metadata_connection_config: for current pipeline
    """
    self._record_dir = record_dir
    # self._metadata_connection_config = metadata_connection_config
    # self._pipeline_info = pipeline_info
    self._threshold = threshold

    # default verifier
    self._verifier_map = {'ExampleValidator': self.validator_verifier,
                          'Trainer': self.trainer_verifier,
                          'Evaluator': self.evaluator_verifier}
    self.component_output_map = verifier_utils.get_component_output_map(
        metadata_connection_config, pipeline_info)

  def trainer_verifier(self, component_id, output_dict):
    """compares two model files"""
    print("trainer_verifier")
    model_artifact = output_dict['model']
    model_uri = model_artifact.uri

    component_id = \
          model_artifact.custom_properties['producer_component'].string_value
    path = os.path.join(self._record_dir, component_id, 'model')
    verifier_utils.compare_model_file_sizes(model_uri, path, self._threshold)

  def evaluator_verifier(self, component_id, output_dict):
    """compares two evaluation proto files"""
    print("evaluator_verifier")
    eval_result = tfma.load_eval_result(output_dict['evaluation'].uri)
    expected_eval_result = tfma.load_eval_result(os.path.join(self._record_dir,
                                                              component_id,
                                                              'evaluation'))
    verifier_utils.compare_eval_results(eval_result,
                                        expected_eval_result,
                                        self._threshold)
    # tfma.load_validation_result(output_dict['blessing'].uri, "BLESSED")
    # tfma.load_validation_result(os.path.join(record_dir, component_id, 'blessing'))

  def validator_verifier(self, component_id, output_dict):
    """compares two validation proto files"""
    print("validator_verifier", component_id)
    # print("output_dict", output_dict)
    anomalies = io_utils.parse_pbtxt_file(
        os.path.join(output_dict['anomalies'].uri, 'anomalies.pbtxt'),
        anomalies_pb2.Anomalies())
    expected_anomalies = io_utils.parse_pbtxt_file(
        os.path.join(output_dict['anomalies'].uri, 'anomalies.pbtxt'),
        anomalies_pb2.Anomalies())
    if expected_anomalies.anomaly_info != anomalies.anomaly_info:
      absl.logging.warning("anomaly info different")

  def set_verifier(self, component_id: Text, verifier_fn: types.FunctionType):
    # compares user verifier
    self._verifier_map[component_id] = verifier_fn

  def verify(self, component_ids):
    for component_id in component_ids:
      verifier_fn = self._verifier_map.get(component_id, None)
      if verifier_fn:
        print("verifying {}".format(component_id))
        assert component_id in self.component_output_map
        verifier_fn(component_id, self.component_output_map[component_id])
