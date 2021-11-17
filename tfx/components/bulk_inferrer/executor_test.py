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

import os
import unittest

import tensorflow as tf
from tfx.components.bulk_inferrer import executor
from tfx.dsl.io import fileio
from tfx.proto import bulk_inferrer_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import proto_utils

from google.protobuf import text_format
from tensorflow_serving.apis import prediction_log_pb2


@unittest.skipIf(tf.__version__ < '2',
                 'This test uses testdata only compatible with TF 2.x')
class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    self._output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self.component_id = 'test_component'

    # Create input dict.
    self._examples = standard_artifacts.Examples()
    unlabelled_path = os.path.join(self._source_data_dir, 'csv_example_gen',
                                   'unlabelled')
    self._examples.uri = os.path.join(self._output_data_dir, 'csv_example_gen')
    io_utils.copy_dir(unlabelled_path,
                      os.path.join(self._examples.uri, 'Split-unlabelled'))
    io_utils.copy_dir(unlabelled_path,
                      os.path.join(self._examples.uri, 'Split-unlabelled2'))
    self._examples.split_names = artifact_utils.encode_split_names(
        ['unlabelled', 'unlabelled2'])
    self._model = standard_artifacts.Model()
    self._model.uri = os.path.join(self._source_data_dir, 'trainer/current')

    self._model_blessing = standard_artifacts.ModelBlessing()
    self._model_blessing.uri = os.path.join(self._source_data_dir,
                                            'model_validator/blessed')
    self._model_blessing.set_int_custom_property('blessed', 1)

    self._input_dict = {
        standard_component_specs.EXAMPLES_KEY: [self._examples],
        standard_component_specs.MODEL_KEY: [self._model],
        standard_component_specs.MODEL_BLESSING_KEY: [self._model_blessing],
    }

    # Create output dict.
    self._inference_result = standard_artifacts.InferenceResult()
    self._prediction_log_dir = os.path.join(self._output_data_dir,
                                            'prediction_logs')
    self._inference_result.uri = self._prediction_log_dir

    self._output_examples = standard_artifacts.Examples()
    self._output_examples_dir = os.path.join(self._output_data_dir,
                                             'output_examples')
    self._output_examples.uri = self._output_examples_dir

    self._output_dict_ir = {
        standard_component_specs.INFERENCE_RESULT_KEY: [self._inference_result],
    }
    self._output_dict_oe = {
        standard_component_specs.OUTPUT_EXAMPLES_KEY: [
            self._output_examples
        ],
    }

    # Create exe properties.
    self._exec_properties = {
        standard_component_specs.DATA_SPEC_KEY:
            proto_utils.proto_to_json(bulk_inferrer_pb2.DataSpec()),
        standard_component_specs.MODEL_SPEC_KEY:
            proto_utils.proto_to_json(bulk_inferrer_pb2.ModelSpec()),
        'component_id':
            self.component_id,
    }

    # Create context
    self._tmp_dir = os.path.join(self._output_data_dir, '.temp')
    self._context = executor.Executor.Context(
        tmp_dir=self._tmp_dir, unique_id='2')

  def _get_results(self, path, file_name, proto_type):
    results = []
    filepattern = os.path.join(path, file_name) + '-?????-of-?????.gz'
    for f in fileio.glob(filepattern):
      record_iterator = tf.compat.v1.python_io.tf_record_iterator(
          path=f,
          options=tf.compat.v1.python_io.TFRecordOptions(
              tf.compat.v1.python_io.TFRecordCompressionType.GZIP))
      for record_string in record_iterator:
        prediction_log = proto_type()
        prediction_log.MergeFromString(record_string)
        results.append(prediction_log)
    return results

  def _verify_example_split(self, split_name):
    self.assertTrue(
        fileio.exists(
            os.path.join(self._output_examples_dir, f'Split-{split_name}')))
    results = self._get_results(
        os.path.join(self._output_examples_dir, f'Split-{split_name}'),
        executor._EXAMPLES_FILE_NAME, tf.train.Example)
    self.assertTrue(results)
    self.assertIn('classify_label', results[0].features.feature)
    self.assertIn('classify_score', results[0].features.feature)

  def testDoWithBlessedModel(self):
    # Run executor.
    bulk_inferrer = executor.Executor(self._context)
    bulk_inferrer.Do(self._input_dict, self._output_dict_ir,
                     self._exec_properties)

    # Check outputs.
    self.assertTrue(fileio.exists(self._prediction_log_dir))
    results = self._get_results(self._prediction_log_dir,
                                executor._PREDICTION_LOGS_FILE_NAME,
                                prediction_log_pb2.PredictionLog)
    self.assertTrue(results)
    self.assertEqual(
        len(results[0].classify_log.response.result.classifications), 1)
    self.assertEqual(
        len(results[0].classify_log.response.result.classifications[0].classes),
        2)

  def testDoWithOutputExamplesAllSplits(self):
    self._exec_properties[standard_component_specs
                          .OUTPUT_EXAMPLE_SPEC_KEY] = proto_utils.proto_to_json(
                              text_format.Parse(
                                  """
                output_columns_spec {
                  classify_output {
                    label_column: 'classify_label'
                    score_column: 'classify_score'
                  }
                }
            """, bulk_inferrer_pb2.OutputExampleSpec()))

    # Run executor.
    bulk_inferrer = executor.Executor(self._context)
    bulk_inferrer.Do(self._input_dict, self._output_dict_oe,
                     self._exec_properties)

    # Check outputs.
    self.assertTrue(fileio.exists(self._output_examples_dir))
    self._verify_example_split('unlabelled')
    self._verify_example_split('unlabelled2')

  def testDoWithOutputExamplesSpecifiedSplits(self):
    self._exec_properties[
        standard_component_specs.DATA_SPEC_KEY] = proto_utils.proto_to_json(
            text_format.Parse(
                """
                example_splits: 'unlabelled'
            """, bulk_inferrer_pb2.DataSpec()))
    self._exec_properties[standard_component_specs
                          .OUTPUT_EXAMPLE_SPEC_KEY] = proto_utils.proto_to_json(
                              text_format.Parse(
                                  """
                output_columns_spec {
                  classify_output {
                    label_column: 'classify_label'
                    score_column: 'classify_score'
                  }
                }
            """, bulk_inferrer_pb2.OutputExampleSpec()))

    # Run executor.
    bulk_inferrer = executor.Executor(self._context)
    bulk_inferrer.Do(self._input_dict, self._output_dict_oe,
                     self._exec_properties)

    # Check outputs.
    self.assertTrue(fileio.exists(self._output_examples_dir))
    self._verify_example_split('unlabelled')
    self.assertFalse(
        fileio.exists(
            os.path.join(self._output_examples_dir, 'Split-unlabelled2')))


if __name__ == '__main__':
  tf.test.main()
