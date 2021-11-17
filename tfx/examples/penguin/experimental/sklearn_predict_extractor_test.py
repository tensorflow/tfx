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
"""Tests for the custom scikit-learn Evaluator module."""

import os
import pickle

import apache_beam as beam
from apache_beam.testing import util
from sklearn import neural_network as nn
import tensorflow_model_analysis as tfma
from tfx.examples.penguin.experimental import sklearn_predict_extractor
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import test_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


class SklearnPredictExtractorTest(tfma.test.TestCase):

  def setUp(self):
    super().setUp()
    self._eval_export_dir = os.path.join(self._getTempDir(), 'eval_export')
    self._create_sklearn_model(self._eval_export_dir)
    self._eval_config = tfma.EvalConfig(model_specs=[tfma.ModelSpec()])
    self._eval_shared_model = (
        sklearn_predict_extractor.custom_eval_shared_model(
            eval_saved_model_path=self._eval_export_dir,
            model_name=None,
            eval_config=self._eval_config))
    self._schema = text_format.Parse(
        """
        feature {
          name: "age"
          type: FLOAT
        }
        feature {
          name: "language"
          type: FLOAT
        }
        feature {
          name: "label"
          type: INT
        }
        """, schema_pb2.Schema())
    self._tfx_io = test_util.InMemoryTFExampleRecord(
        schema=self._schema,
        raw_record_column_name=tfma.ARROW_INPUT_COLUMN)
    self._tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=self._tfx_io.ArrowSchema(),
        tensor_representations=self._tfx_io.TensorRepresentations())
    self._examples = [
        self._makeExample(age=3.0, language=1.0, label=1),
        self._makeExample(age=3.0, language=0.0, label=0),
        self._makeExample(age=4.0, language=1.0, label=1),
        self._makeExample(age=5.0, language=0.0, label=0),
    ]

  def testMakeSklearnPredictExtractor(self):
    """Tests that predictions are made from extracts for a single model."""
    feature_extractor = tfma.extractors.FeaturesExtractor(self._eval_config)
    prediction_extractor = (
        sklearn_predict_extractor._make_sklearn_predict_extractor(
            self._eval_shared_model))
    with beam.Pipeline() as pipeline:
      predict_extracts = (
          pipeline
          | 'Create' >> beam.Create(
              [e.SerializeToString() for e in self._examples])
          | 'BatchExamples' >> self._tfx_io.BeamSource()
          | 'InputsToExtracts' >> tfma.BatchedInputsToExtracts()  # pylint: disable=no-value-for-parameter
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
      )

      def check_result(actual):
        try:
          for item in actual:
            self.assertEqual(item['labels'].shape, item['predictions'].shape)

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(predict_extracts, check_result)

  def testMakeSklearnPredictExtractorWithMultiModels(self):
    """Tests that predictions are made from extracts for multiple models."""
    eval_config = tfma.EvalConfig(model_specs=[
        tfma.ModelSpec(name='model1'),
        tfma.ModelSpec(name='model2'),
    ])
    eval_export_dir_1 = os.path.join(self._eval_export_dir, '1')
    self._create_sklearn_model(eval_export_dir_1)
    eval_shared_model_1 = sklearn_predict_extractor.custom_eval_shared_model(
        eval_saved_model_path=eval_export_dir_1,
        model_name='model1',
        eval_config=eval_config)
    eval_export_dir_2 = os.path.join(self._eval_export_dir, '2')
    self._create_sklearn_model(eval_export_dir_2)
    eval_shared_model_2 = sklearn_predict_extractor.custom_eval_shared_model(
        eval_saved_model_path=eval_export_dir_2,
        model_name='model2',
        eval_config=eval_config)

    feature_extractor = tfma.extractors.FeaturesExtractor(self._eval_config)
    prediction_extractor = (
        sklearn_predict_extractor._make_sklearn_predict_extractor(
            eval_shared_model={
                'model1': eval_shared_model_1,
                'model2': eval_shared_model_2,
            }))
    with beam.Pipeline() as pipeline:
      predict_extracts = (
          pipeline
          | 'Create' >> beam.Create(
              [e.SerializeToString() for e in self._examples])
          | 'BatchExamples' >> self._tfx_io.BeamSource()
          | 'InputsToExtracts' >> tfma.BatchedInputsToExtracts()  # pylint: disable=no-value-for-parameter
          | feature_extractor.stage_name >> feature_extractor.ptransform
          | prediction_extractor.stage_name >> prediction_extractor.ptransform
      )

      def check_result(actual):
        try:
          for item in actual:
            self.assertEqual(item['labels'].shape, item['predictions'].shape)
            self.assertIn('model1', item['predictions'][0])
            self.assertIn('model2', item['predictions'][0])

        except AssertionError as err:
          raise util.BeamAssertException(err)

      util.assert_that(predict_extracts, check_result)

  def test_custom_eval_shared_model(self):
    """Tests that an EvalSharedModel is created with a custom sklearn loader."""
    model_file = os.path.basename(self._eval_shared_model.model_path)
    self.assertEqual(model_file, 'model.pkl')
    model = self._eval_shared_model.model_loader.construct_fn()
    self.assertIsInstance(model, nn.MLPClassifier)

  def test_custom_extractors(self):
    """Tests that the sklearn extractor is used when creating extracts."""
    extractors = sklearn_predict_extractor.custom_extractors(
        self._eval_shared_model, self._eval_config, self._tensor_adapter_config)
    self.assertLen(extractors, 6)
    self.assertIn(
        'SklearnPredict', [extractor.stage_name for extractor in extractors])

  def _create_sklearn_model(self, eval_export_dir):
    """Creates and pickles a toy scikit-learn model.

    Args:
        eval_export_dir: Directory to store a pickled scikit-learn model. This
            directory is created if it does not exist.
    """
    x_train = [[3, 0], [4, 1]]
    y_train = [0, 1]
    model = nn.MLPClassifier(max_iter=1)
    model.feature_keys = ['age', 'language']
    model.label_key = 'label'
    model.fit(x_train, y_train)

    os.makedirs(eval_export_dir)
    model_path = os.path.join(eval_export_dir, 'model.pkl')
    with open(model_path, 'wb+') as f:
      pickle.dump(model, f)
