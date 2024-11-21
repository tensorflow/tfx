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

import os

import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform import beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from tfx.examples.chicago_taxi_pipeline import taxi_utils
from tfx.utils import io_utils
from tfx_bsl.tfxio import tf_example_record

from tensorflow_metadata.proto.v0 import schema_pb2


class TaxiUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._testdata_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'components/testdata')

  def testUtils(self):
    key = 'fare'
    xfm_key = taxi_utils._transformed_name(key)
    self.assertEqual(xfm_key, 'fare_xf')

  def testPreprocessingFn(self):
    schema_file = os.path.join(self._testdata_path, 'schema_gen/schema.pbtxt')
    schema = io_utils.parse_pbtxt_file(schema_file, schema_pb2.Schema())
    feature_spec = taxi_utils._get_raw_feature_spec(schema)
    working_dir = self.get_temp_dir()
    transform_graph_path = os.path.join(working_dir, 'transform_graph')
    transformed_examples_path = os.path.join(
        working_dir, 'transformed_examples')

    # Run very simplified version of executor logic.
    # TODO(kestert): Replace with tft_unit.assertAnalyzeAndTransformResults.
    # Generate legacy `DatasetMetadata` object.  Future version of Transform
    # will accept the `Schema` proto directly.
    legacy_metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(feature_spec))
    tfxio = tf_example_record.TFExampleRecord(
        file_pattern=os.path.join(self._testdata_path,
                                  'csv_example_gen/Split-train/*'),
        telemetry_descriptors=['Tests'],
        schema=legacy_metadata.schema)
    with beam.Pipeline() as p:
      with tft_beam.Context(temp_dir=os.path.join(working_dir, 'tmp')):
        examples = p | 'ReadTrainData' >> tfxio.BeamSource()
        (transformed_examples, transformed_metadata), transform_fn = (
            (examples, tfxio.TensorAdapterConfig())
            | 'AnalyzeAndTransform' >> tft_beam.AnalyzeAndTransformDataset(
                taxi_utils.preprocessing_fn))

        # WriteTransformFn writes transform_fn and metadata to subdirectories
        # tensorflow_transform.SAVED_MODEL_DIR and
        # tensorflow_transform.TRANSFORMED_METADATA_DIR respectively.
        # pylint: disable=expression-not-assigned
        (transform_fn
         |
         'WriteTransformFn' >> tft_beam.WriteTransformFn(transform_graph_path))

        encoder = tft.coders.ExampleProtoCoder(transformed_metadata.schema)
        (transformed_examples
         | 'EncodeTrainData' >> beam.Map(encoder.encode)
         | 'WriteTrainData' >> beam.io.WriteToTFRecord(
             os.path.join(transformed_examples_path,
                          'Split-train/transformed_examples.gz'),
             coder=beam.coders.BytesCoder()))
        # pylint: enable=expression-not-assigned

    # Verify the output matches golden output.
    # NOTE: we don't verify that transformed examples match golden output.
    expected_transformed_schema = io_utils.parse_pbtxt_file(
        os.path.join(
            self._testdata_path,
            'transform/transform_graph/transformed_metadata/schema.pbtxt'),
        schema_pb2.Schema())
    transformed_schema = io_utils.parse_pbtxt_file(
        os.path.join(transform_graph_path, 'transformed_metadata/schema.pbtxt'),
        schema_pb2.Schema())
    # Clear annotations so we only have to test main schema.
    transformed_schema.ClearField('annotation')
    for feature in transformed_schema.feature:
      feature.ClearField('annotation')
    self.assertEqual(transformed_schema, expected_transformed_schema)
