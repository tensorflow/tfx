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
"""Tests for tfx.utils.tfxio_utils."""

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import unittest

from absl.testing import parameterized
import tensorflow as tf
from tfx.components.experimental.data_view import constants
from tfx.components.util import examples_utils
from tfx.components.util import tfxio_utils
from tfx.proto import example_gen_pb2
from tfx.types import standard_artifacts
import tfx_bsl
from tfx_bsl.tfxio import raw_tf_record
from tfx_bsl.tfxio import record_based_tfxio
from tfx_bsl.tfxio import tf_example_record
from tfx_bsl.tfxio import tf_sequence_example_record

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2

# TODO(b/161449255): clean this up after a release post tfx_bsl 0.22.1.
if getattr(tfx_bsl, 'HAS_TF_GRAPH_RECORD_DECODER', False):
  from tfx_bsl.coders import tf_graph_record_decoder  # pylint: disable=g-import-not-at-top
  from tfx_bsl.coders.tf_graph_record_decoder import TFGraphRecordDecoder  # pylint: disable=g-import-not-at-top
  from tfx_bsl.tfxio.record_to_tensor_tfxio import TFRecordToTensorTFXIO  # pylint: disable=g-import-not-at-top
else:
  tf_graph_record_decoder = None
  TFRecordToTensorTFXIO = None  # pylint: disable=invalid-name
  TFGraphRecordDecoder = object  # pylint: disable=invalid-name

_RAW_RECORD_COLUMN_NAME = 'raw_record'
_MAKE_TFXIO_TEST_CASES = [
    dict(
        testcase_name='tf_example_record',
        payload_format=example_gen_pb2.PayloadFormat.FORMAT_TF_EXAMPLE,
        expected_tfxio_type=tf_example_record.TFExampleRecord),
    dict(
        testcase_name='tf_example_record_also_read_raw_records',
        payload_format=example_gen_pb2.PayloadFormat.FORMAT_TF_EXAMPLE,
        raw_record_column_name=_RAW_RECORD_COLUMN_NAME,
        expected_tfxio_type=tf_example_record.TFExampleRecord),
    dict(
        testcase_name='tf_example_record_default_payload_format',
        payload_format=None,
        expected_tfxio_type=tf_example_record.TFExampleRecord),
    dict(
        testcase_name='tf_sequence_example_record',
        payload_format=example_gen_pb2.PayloadFormat.FORMAT_TF_SEQUENCE_EXAMPLE,
        expected_tfxio_type=tf_sequence_example_record.TFSequenceExampleRecord),
    dict(
        testcase_name='proto_with_data_view',
        payload_format=example_gen_pb2.PayloadFormat.FORMAT_PROTO,
        provide_data_view_uri=True,
        expected_tfxio_type=TFRecordToTensorTFXIO),
    dict(
        testcase_name='tf_example_raw_record',
        payload_format=example_gen_pb2.PayloadFormat.FORMAT_TF_EXAMPLE,
        read_as_raw_records=True,
        raw_record_column_name=_RAW_RECORD_COLUMN_NAME,
        expected_tfxio_type=raw_tf_record.RawTfRecordTFXIO),
    dict(
        testcase_name='proto_raw_record',
        payload_format=example_gen_pb2.PayloadFormat.FORMAT_PROTO,
        read_as_raw_records=True,
        raw_record_column_name=_RAW_RECORD_COLUMN_NAME,
        expected_tfxio_type=raw_tf_record.RawTfRecordTFXIO),
]

_FAKE_FILE_PATTERN = '/input/data'
_TELEMETRY_DESCRIPTORS = ['my', 'component']
_SCHEMA = text_format.Parse("""
feature {
  name: "foo"
  type: INT
}
""", schema_pb2.Schema())


class _SimpleTfGraphRecordDecoder(TFGraphRecordDecoder):
  """A simple DataView Decoder used for testing."""

  def _decode_record_internal(self, record):
    indices = tf.transpose(
        tf.stack([
            tf.range(tf.size(record), dtype=tf.int64),
            tf.zeros(tf.size(record), dtype=tf.int64)
        ]))

    return {
        'sparse_tensor':
            tf.SparseTensor(
                values=record,
                indices=indices,
                dense_shape=[tf.size(record), 1])
    }


@unittest.skipIf(not getattr(tfx_bsl, 'HAS_TF_GRAPH_RECORD_DECODER', False),
                 'tfx-bsl installed does not have modules required to run '
                 'this test.')
class TfxioUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(*_MAKE_TFXIO_TEST_CASES)
  def test_make_tfxio(self, payload_format, expected_tfxio_type,
                      raw_record_column_name=None,
                      provide_data_view_uri=False,
                      read_as_raw_records=False):
    if payload_format is None:
      payload_format = 'FORMAT_TF_EXAMPLE'
    data_view_uri = None
    if provide_data_view_uri:
      data_view_uri = tempfile.mkdtemp(dir=self.get_temp_dir())
      tf_graph_record_decoder.save_decoder(_SimpleTfGraphRecordDecoder(),
                                           data_view_uri)
    tfxio = tfxio_utils.make_tfxio(
        _FAKE_FILE_PATTERN, _TELEMETRY_DESCRIPTORS, payload_format,
        data_view_uri, _SCHEMA, read_as_raw_records, raw_record_column_name)
    self.assertIsInstance(tfxio, expected_tfxio_type)
    # We currently only create RecordBasedTFXIO and the check below relies on
    # that.
    self.assertIsInstance(tfxio, record_based_tfxio.RecordBasedTFXIO)
    self.assertEqual(tfxio.telemetry_descriptors, _TELEMETRY_DESCRIPTORS)
    self.assertEqual(tfxio.raw_record_column_name, raw_record_column_name)
    # Since we provide a schema, ArrowSchema() should not raise.
    _ = tfxio.ArrowSchema()

  @parameterized.named_parameters(*_MAKE_TFXIO_TEST_CASES)
  def test_get_tfxio_factory_from_artifact(self,
                                           payload_format,
                                           expected_tfxio_type,
                                           raw_record_column_name=None,
                                           provide_data_view_uri=False,
                                           read_as_raw_records=False):
    examples = standard_artifacts.Examples()
    if payload_format is not None:
      examples_utils.set_payload_format(examples, payload_format)
    data_view_uri = None
    if provide_data_view_uri:
      data_view_uri = tempfile.mkdtemp(dir=self.get_temp_dir())
      tf_graph_record_decoder.save_decoder(_SimpleTfGraphRecordDecoder(),
                                           data_view_uri)
    if data_view_uri is not None:
      examples.set_string_custom_property(
          constants.DATA_VIEW_URI_PROPERTY_KEY, data_view_uri)
    tfxio_factory = tfxio_utils.get_tfxio_factory_from_artifact(
        examples,
        _TELEMETRY_DESCRIPTORS,
        _SCHEMA,
        read_as_raw_records,
        raw_record_column_name)
    tfxio = tfxio_factory(_FAKE_FILE_PATTERN)
    self.assertIsInstance(tfxio, expected_tfxio_type)
    # We currently only create RecordBasedTFXIO and the check below relies on
    # that.
    self.assertIsInstance(tfxio, record_based_tfxio.RecordBasedTFXIO)
    self.assertEqual(tfxio.telemetry_descriptors, _TELEMETRY_DESCRIPTORS)
    self.assertEqual(tfxio.raw_record_column_name, raw_record_column_name)
    # Since we provide a schema, ArrowSchema() should not raise.
    _ = tfxio.ArrowSchema()

  def test_raise_if_not_example(self):
    artifact = standard_artifacts.DataView()
    with self.assertRaisesRegex(AssertionError,
                                'must be of type standard_artifacts.Example'):
      tfxio_utils.get_tfxio_factory_from_artifact(artifact,
                                                  _TELEMETRY_DESCRIPTORS)

  def test_raise_if_data_view_uri_not_available(self):
    examples = standard_artifacts.Examples()
    examples_utils.set_payload_format(
        examples, example_gen_pb2.PayloadFormat.FORMAT_PROTO)
    with self.assertRaisesRegex(AssertionError, 'requires a DataView'):
      tfxio_utils.get_tfxio_factory_from_artifact(
          examples, _TELEMETRY_DESCRIPTORS)(_FAKE_FILE_PATTERN)

  def test_raise_if_read_as_raw_but_raw_column_name_not_provided(self):
    examples = standard_artifacts.Examples()
    with self.assertRaisesRegex(AssertionError,
                                'must provide raw_record_column_name'):
      tfxio_utils.get_tfxio_factory_from_artifact(
          examples, _TELEMETRY_DESCRIPTORS, read_as_raw_records=True)(
              _FAKE_FILE_PATTERN)


if __name__ == '__main__':
  tf.test.main()
