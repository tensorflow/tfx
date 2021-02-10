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
"""Tests for tfx.examples.ranking.ranking_pipeline."""
import os

import tensorflow as tf

from tfx.examples.ranking import ranking_pipeline
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from google.protobuf import text_format
from tensorflow_serving.apis import input_pb2


_ELWC = text_format.Parse("""
context {
  features {
    feature {
      key: "query_tokens"
      value {
        bytes_list { value: ["cat", "dog"] }
      }
    }
  }
}
examples {
  features {
    feature {
      key: "document_tokens"
      value {
        bytes_list { value: ["red", "blue"] }
      }
    }
  }
}
""", input_pb2.ExampleListWithContext())


class RankingPipelineTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self._pipeline_name = 'tf_ranking_test'
    self._data_root = os.path.join(os.path.dirname(__file__),
                                   'testdata', 'input')
    self._tfx_root = os.path.join(self._test_dir, 'tfx')
    self._module_file = os.path.join(os.path.dirname(__file__),
                                     'ranking_utils.py')
    self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
    self._metadata_path = os.path.join(self._tfx_root, 'metadata',
                                       self._pipeline_name, 'metadata.db')
    print('TFX ROOT: ', self._tfx_root)

  def _AssertAllComponentsExecuted(
      self, metadata_path, num_expected_components):
     # Make sure that all the components ran successfully.
    metadata_config = metadata.sqlite_metadata_connection_config(metadata_path)
    with metadata.Metadata(metadata_config) as m:
      artifact_count = len(m.store.get_artifacts())
      execution_count = len(m.store.get_executions())
      self.assertGreaterEqual(artifact_count, execution_count)
      self.assertEqual(execution_count, num_expected_components)

  def _ValidateServingSignatures(self, saved_model_path):
    serving_model = tf.saved_model.load(saved_model_path)
    self.assertLen(serving_model.signatures, 3)
    self.assertIn('tensorflow/serving/regress', serving_model.signatures)
    self.assertIn('tensorflow/serving/predict', serving_model.signatures)
    self.assertIn('serving_default', serving_model.signatures)

    def validate_output_dict(d):
      self.assertIsInstance(d, dict)
      self.assertLen(d, 1)
      self.assertIn('outputs', d)

    predict = serving_model.signatures['tensorflow/serving/predict']
    predict_output = predict(tf.convert_to_tensor([_ELWC.SerializeToString()]))
    validate_output_dict(predict_output)

    regress = serving_model.signatures['tensorflow/serving/regress']
    tf_example = tf.train.Example()
    tf_example.MergeFrom(_ELWC.context)
    tf_example.MergeFrom(_ELWC.examples[0])
    regress_output = regress(
        tf.convert_to_tensor([tf_example.SerializeToString()]))
    validate_output_dict(regress_output)
    self.assertAllEqual(predict_output['outputs'],
                        tf.expand_dims(regress_output['outputs'], axis=0))

    serving_default = serving_model.signatures['serving_default']
    serving_default_output = serving_default(
        tf.convert_to_tensor([_ELWC.SerializeToString()]))
    validate_output_dict(serving_default_output)
    self.assertAllEqual(predict_output['outputs'],
                        serving_default_output['outputs'])

  def testPipeline(self):
    BeamDagRunner().run(
        ranking_pipeline._create_pipeline(
            pipeline_name=self._pipeline_name,
            pipeline_root=self._tfx_root,
            data_root=self._data_root,
            module_file=self._module_file,
            serving_model_dir=self._serving_model_dir,
            metadata_path=self._metadata_path,
            beam_pipeline_args=['--direct_num_workers=1']))

    self.assertTrue(tf.io.gfile.exists(self._metadata_path))
    self._AssertAllComponentsExecuted(self._metadata_path, 9)

    self.assertTrue(tf.io.gfile.exists(self._serving_model_dir))
    serving_model_dir_contents = tf.io.gfile.listdir(self._serving_model_dir)
    self.assertLen(serving_model_dir_contents, 1)
    self._ValidateServingSignatures(
        os.path.join(self._serving_model_dir, serving_model_dir_contents[0]))


if __name__ == '__main__':
  tf.test.main()
