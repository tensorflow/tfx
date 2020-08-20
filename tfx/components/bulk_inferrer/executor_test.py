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
"""Tests for bulk_inferrer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from google.protobuf import json_format
from tensorflow_serving.apis import prediction_log_pb2
from tfx.components.bulk_inferrer import executor
from tfx.proto import bulk_inferrer_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self._source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self.component_id = 'test_component'

    # Create input dict.
    self._examples = standard_artifacts.Examples()
    self._examples.uri = os.path.join(self._source_data_dir, 'csv_example_gen')
    self._examples.split_names = artifact_utils.encode_split_names(
        ['unlabelled'])
    self._model = standard_artifacts.Model()
    self._model.uri = os.path.join(self._source_data_dir, 'trainer/current')

    self._model_blessing = standard_artifacts.ModelBlessing()
    self._model_blessing.uri = os.path.join(self._source_data_dir,
                                            'model_validator/blessed')
    self._model_blessing.set_int_custom_property('blessed', 1)

    self._inference_result = standard_artifacts.InferenceResult()
    self._prediction_log_dir = os.path.join(self._output_data_dir,
                                            'prediction_logs')
    self._inference_result.uri = self._prediction_log_dir

    # Create context
    self._tmp_dir = os.path.join(self._output_data_dir, '.temp')
    self._context = executor.Executor.Context(
        tmp_dir=self._tmp_dir, unique_id='2')

  def _get_results(self, prediction_log_path):
    results = []
    filepattern = os.path.join(
        prediction_log_path,
        executor._PREDICTION_LOGS_DIR_NAME) + '-?????-of-?????.gz'
    for f in tf.io.gfile.glob(filepattern):
      record_iterator = tf.compat.v1.python_io.tf_record_iterator(
          path=f,
          options=tf.compat.v1.python_io.TFRecordOptions(
              tf.compat.v1.python_io.TFRecordCompressionType.GZIP))
      for record_string in record_iterator:
        prediction_log = prediction_log_pb2.PredictionLog()
        prediction_log.MergeFromString(record_string)
        results.append(prediction_log)
    return results

  def testDoWithBlessedModel(self):
    input_dict = {
        'examples': [self._examples],
        'model': [self._model],
        'model_blessing': [self._model_blessing],
    }
    output_dict = {
        'inference_result': [self._inference_result],
    }
    # Create exe properties.
    exec_properties = {
        'data_spec':
            json_format.MessageToJson(
                bulk_inferrer_pb2.DataSpec(), preserving_proto_field_name=True),
        'model_spec':
            json_format.MessageToJson(
                bulk_inferrer_pb2.ModelSpec(),
                preserving_proto_field_name=True),
        'component_id':
            self.component_id,
    }

    # Run executor.
    bulk_inferrer = executor.Executor(self._context)
    bulk_inferrer.Do(input_dict, output_dict, exec_properties)

    # Check outputs.
    self.assertTrue(tf.io.gfile.exists(self._prediction_log_dir))
    results = self._get_results(self._prediction_log_dir)
    self.assertTrue(results)
    self.assertEqual(
        len(results[0].classify_log.response.result.classifications), 1)
    self.assertEqual(
        len(results[0].classify_log.response.result.classifications[0].classes),
        2)


if __name__ == '__main__':
  tf.test.main()
