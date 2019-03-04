# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tfx.examples.chicago_taxi_pipeline.taxi_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import types

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import dataset_schema
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx.examples.chicago_taxi_pipeline import taxi_utils
from tfx.utils import io_utils


class TaxiUtilsTest(tf.test.TestCase):

  def setUp(self):
    self._testdata_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'components/testdata')

  def test_utils(self):
    key = 'fare'
    xfm_key = taxi_utils._transformed_name(key)
    self.assertEqual(xfm_key, 'fare_xf')

  def test_preprocessing_fn(self):
    schema_file = os.path.join(self._testdata_path, 'schema_gen/schema.pbtxt')
    schema = io_utils.parse_pbtxt_file(schema_file, schema_pb2.Schema())
    feature_spec = taxi_utils._get_raw_feature_spec(schema)
    working_dir = self.get_temp_dir()
    transform_output_path = os.path.join(working_dir, 'transform_output')
    transformed_examples_path = os.path.join(
        working_dir, 'transformed_examples')

    # Run very simplified version of executor logic.
    # TODO(kestert): Replace with tft_unit.assertAnalyzeAndTransformResults.
    # Generate legacy `DatasetMetadata` object.  Future version of Transform
    # will accept the `Schema` proto directly.
    legacy_metadata = dataset_metadata.DatasetMetadata(
        dataset_schema.from_feature_spec(feature_spec))
    decoder = tft.coders.ExampleProtoCoder(legacy_metadata.schema)
    with beam.Pipeline() as p:
      with tft_beam.Context(temp_dir=os.path.join(working_dir, 'tmp')):
        examples = (
            p
            | 'ReadTrainData' >> beam.io.ReadFromTFRecord(
                os.path.join(self._testdata_path, 'csv_example_gen/train/*'),
                coder=beam.coders.BytesCoder(),
                # TODO(b/114938612): Eventually remove this override.
                validate=False)
            | 'DecodeTrainData' >> beam.Map(decoder.decode))
        (transformed_examples, transformed_metadata), transform_fn = (
            (examples, legacy_metadata)
            | 'AnalyzeAndTransform' >> tft_beam.AnalyzeAndTransformDataset(
                taxi_utils.preprocessing_fn))

        # WriteTransformFn writes transform_fn and metadata to subdirectories
        # tensorflow_transform.SAVED_MODEL_DIR and
        # tensorflow_transform.TRANSFORMED_METADATA_DIR respectively.
        # pylint: disable=expression-not-assigned
        (transform_fn
         | 'WriteTransformFn' >> tft_beam.WriteTransformFn(
             transform_output_path))

        encoder = tft.coders.ExampleProtoCoder(transformed_metadata.schema)
        (transformed_examples
         | 'EncodeTrainData' >> beam.Map(encoder.encode)
         | 'WriteTrainData' >> beam.io.WriteToTFRecord(
             os.path.join(transformed_examples_path,
                          'train/transformed_exmaples.gz'),
             coder=beam.coders.BytesCoder()))
        # pylint: enable=expression-not-assigned

    # Verify the output matches golden output.
    # NOTE: we don't verify that transformed examples match golden output.
    expected_transformed_schema = io_utils.parse_pbtxt_file(
        os.path.join(
            self._testdata_path,
            'transform/transform_output/transformed_metadata/schema.pbtxt'),
        schema_pb2.Schema())
    transformed_schema = io_utils.parse_pbtxt_file(
        os.path.join(transform_output_path,
                     'transformed_metadata/schema.pbtxt'),
        schema_pb2.Schema())
    self.assertEqual(transformed_schema, expected_transformed_schema)

  def test_trainer_fn(self):
    temp_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    schema_file = os.path.join(self._testdata_path, 'schema_gen/schema.pbtxt')
    hparams = tf.contrib.training.HParams(
        train_files=os.path.join(temp_dir, 'train_files'),
        transform_output=os.path.join(self._testdata_path,
                                      'transform/transform_output/'),
        output_dir=os.path.join(temp_dir, 'output_dir'),
        serving_model_dir=os.path.join(temp_dir, 'serving_model_dir'),
        eval_files=os.path.join(temp_dir, 'eval_files'),
        schema_file=schema_file,
        train_steps=10001,
        eval_steps=5000,
        verbosity='INFO',
        warm_start_from=os.path.join(temp_dir, 'serving_model_dir'))
    schema = io_utils.parse_pbtxt_file(schema_file, schema_pb2.Schema())
    training_spec = taxi_utils.trainer_fn(hparams, schema)

    self.assertIsInstance(training_spec['estimator'],
                          tf.estimator.DNNLinearCombinedClassifier)
    self.assertIsInstance(training_spec['train_spec'], tf.estimator.TrainSpec)
    self.assertIsInstance(training_spec['eval_spec'], tf.estimator.EvalSpec)
    self.assertIsInstance(training_spec['eval_input_receiver_fn'],
                          types.FunctionType)


if __name__ == '__main__':
  tf.test.main()
