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
"""Tests for tfx.orchestration.data_types_utils."""

from absl.testing import parameterized

import tensorflow as tf
from tfx.orchestration import data_types_utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import artifact_utils

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2


class DataTypesUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.artifact_struct_dict = {
        'a1':
            text_format.Parse(
                """
                elements {
                  artifact {
                    artifact {
                      id: 123
                    }
                    type {
                      name: 't1'
                    }
                  }
                }
                """, metadata_store_service_pb2.ArtifactStructList()),
        'a2':
            text_format.Parse(
                """
                elements {
                  artifact {
                    artifact {
                      id: 456
                    }
                    type {
                      name: 't2'
                    }
                  }
                }
                """, metadata_store_service_pb2.ArtifactStructList())
    }

    self.artifact_dict = {
        'a1': [
            artifact_utils.deserialize_artifact(
                metadata_store_pb2.ArtifactType(name='t1'),
                metadata_store_pb2.Artifact(id=123))
        ],
        'a2': [
            artifact_utils.deserialize_artifact(
                metadata_store_pb2.ArtifactType(name='t2'),
                metadata_store_pb2.Artifact(id=456))
        ]
    }

    self.metadata_value_dict = {
        'p1': metadata_store_pb2.Value(int_value=1),
        'p2': metadata_store_pb2.Value(string_value='hello')
    }
    self.value_dict = {'p1': 1, 'p2': 'hello'}

  def testBuildArtifactDict(self):
    actual_artifact_dict = data_types_utils.build_artifact_dict(
        self.artifact_struct_dict)
    for k, v in actual_artifact_dict.items():
      self.assertLen(self.artifact_dict[k], len(v))
      self.assertEqual(self.artifact_dict[k][0].id, v[0].id)
      self.assertEqual(self.artifact_dict[k][0].type_name, v[0].type_name)

  def testBuildArtifactStructDict(self):
    actual_artifact_struct_dict = data_types_utils.build_artifact_struct_dict(
        self.artifact_dict)
    self.assertEqual(self.artifact_struct_dict, actual_artifact_struct_dict)

  def testBuildValueDict(self):
    actual_value_dict = data_types_utils.build_value_dict(
        self.metadata_value_dict)
    self.assertEqual(self.value_dict, actual_value_dict)

  def testBuildMetadataValueDict(self):
    actual_metadata_value_dict = (
        data_types_utils.build_metadata_value_dict(self.value_dict))
    self.assertEqual(self.metadata_value_dict, actual_metadata_value_dict)

  def testGetMetadataValueType(self):
    tfx_value = pipeline_pb2.Value()
    text_format.Parse(
        """
        field_value {
          int_value: 1
        }""", tfx_value)
    self.assertEqual(
        data_types_utils.get_metadata_value_type(tfx_value),
        metadata_store_pb2.INT)

  def testGetMetadataValueTypePrimitiveValue(self):
    self.assertEqual(
        data_types_utils.get_metadata_value_type(1), metadata_store_pb2.INT)

  def testGetMetadataValueTypeFailed(self):
    tfx_value = pipeline_pb2.Value()
    text_format.Parse(
        """
        runtime_parameter {
          name: 'rp'
        }""", tfx_value)
    with self.assertRaisesRegex(RuntimeError, 'Expecting field_value but got'):
      data_types_utils.get_metadata_value_type(tfx_value)

  def testGetValue(self):
    tfx_value = pipeline_pb2.Value()
    text_format.Parse(
        """
        field_value {
          int_value: 1
        }""", tfx_value)
    self.assertEqual(data_types_utils.get_value(tfx_value), 1)

  def testGetValueFailed(self):
    tfx_value = pipeline_pb2.Value()
    text_format.Parse(
        """
        runtime_parameter {
          name: 'rp'
        }""", tfx_value)
    with self.assertRaisesRegex(RuntimeError, 'Expecting field_value but got'):
      data_types_utils.get_value(tfx_value)

  def testSetMetadataValueWithTfxValue(self):
    tfx_value = pipeline_pb2.Value()
    metadata_property = metadata_store_pb2.Value()
    text_format.Parse(
        """
        field_value {
            int_value: 1
        }""", tfx_value)
    data_types_utils.set_metadata_value(
        metadata_value=metadata_property, value=tfx_value)
    self.assertProtoEquals('int_value: 1', metadata_property)

  def testSetMetadataValueWithTfxValueFailed(self):
    tfx_value = pipeline_pb2.Value()
    metadata_property = metadata_store_pb2.Value()
    text_format.Parse(
        """
        runtime_parameter {
          name: 'rp'
        }""", tfx_value)
    with self.assertRaisesRegex(ValueError, 'Expecting field_value but got'):
      data_types_utils.set_metadata_value(
          metadata_value=metadata_property, value=tfx_value)

  @parameterized.named_parameters(
      ('IntValue', 42, metadata_store_pb2.Value(int_value=42)),
      ('FloatValue', 42.0, metadata_store_pb2.Value(double_value=42.0)),
      ('StrValue', '42', metadata_store_pb2.Value(string_value='42')))
  def testSetMetadataValueWithPrimitiveValue(self, value, expected_pb):
    pb = metadata_store_pb2.Value()
    data_types_utils.set_metadata_value(pb, value)
    self.assertEqual(pb, expected_pb)

  def testSetMetadataValueUnsupportedType(self):
    pb = metadata_store_pb2.Value()
    with self.assertRaises(ValueError):
      data_types_utils.set_metadata_value(pb, True)


if __name__ == '__main__':
  tf.test.main()
