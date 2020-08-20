# Lint as: python3
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
"""Tests for tfx.orchestration.portable.mlmd.common_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import tensorflow as tf

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import common_utils
from tfx.proto.orchestration import pipeline_pb2


def _create_type(metadata_type_class):
  type_with_two_properties = metadata_type_class()
  text_format.Parse(
      """
      name: 'my_type'
      properties {
        key: 'p1'
        value: INT
      }
      properties {
        key: 'p2'
        value: STRING
      }
      """, type_with_two_properties)
  return type_with_two_properties


class CommonUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()

  def testGetMetadataValueType(self):
    tfx_value = pipeline_pb2.Value()
    text_format.Parse(
        """
        field_value {
          int_value: 1
        }""", tfx_value)
    self.assertEqual(
        common_utils.get_metadata_value_type(tfx_value), metadata_store_pb2.INT)

  def testGetMetadataValueTypePrimitiveValue(self):
    self.assertEqual(
        common_utils.get_metadata_value_type(1), metadata_store_pb2.INT)

  def testGetMetadataValueTypeFailed(self):
    tfx_value = pipeline_pb2.Value()
    text_format.Parse(
        """
        runtime_parameter {
          name: 'rp'
        }""", tfx_value)
    with self.assertRaisesRegex(RuntimeError, 'Expecting field_value but got'):
      common_utils.get_metadata_value_type(tfx_value)

  def testGetValue(self):
    tfx_value = pipeline_pb2.Value()
    text_format.Parse(
        """
        field_value {
          int_value: 1
        }""", tfx_value)
    self.assertEqual(common_utils.get_value(tfx_value), 1)

  def testGetValueFailed(self):
    tfx_value = pipeline_pb2.Value()
    text_format.Parse(
        """
        runtime_parameter {
          name: 'rp'
        }""", tfx_value)
    with self.assertRaisesRegex(RuntimeError, 'Expecting field_value but got'):
      common_utils.get_value(tfx_value)

  def testSetMetadataValueWithTfxValue(self):
    tfx_value = pipeline_pb2.Value()
    metadata_property = metadata_store_pb2.Value()
    text_format.Parse(
        """
        field_value {
            int_value: 1
        }""", tfx_value)
    common_utils.set_metadata_value(
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
    with self.assertRaisesRegex(RuntimeError, 'Expecting field_value but got'):
      common_utils.set_metadata_value(
          metadata_value=metadata_property, value=tfx_value)

  def testSetMetadataValueWithPrimitiveValue(self):
    metadata_property = metadata_store_pb2.Value()
    common_utils.set_metadata_value(metadata_value=metadata_property, value=1)
    self.assertProtoEquals('int_value: 1', metadata_property)

  @parameterized.parameters(metadata_store_pb2.ArtifactType,
                            metadata_store_pb2.ContextType,
                            metadata_store_pb2.ExecutionType)
  def testRegisterType(self, metadata_type_class):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      type_with_two_properties = _create_type(metadata_type_class)
      result_one = common_utils.register_type_if_not_exist(
          m, type_with_two_properties)
      self.assertProtoEquals(
          """
          id: 1
          name: 'my_type'
          properties {
            key: 'p1'
            value: INT
          }
          properties {
            key: 'p2'
            value: STRING
          }
          """, result_one)

  @parameterized.parameters(metadata_store_pb2.ArtifactType,
                            metadata_store_pb2.ContextType,
                            metadata_store_pb2.ExecutionType)
  def testRegisterTypeReuseExisting(self, metadata_type_class):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      type_with_two_properties = _create_type(metadata_type_class)
      result_one = common_utils.register_type_if_not_exist(
          m, type_with_two_properties)
      # Tries to register a type that shares the same name of the type
      # previously registered but with no properties. We expect the previously
      # registered type to be reused.
      type_without_properties = metadata_type_class()
      text_format.Parse("name: 'my_type'", type_without_properties)
      result_two = common_utils.register_type_if_not_exist(
          m, type_without_properties)
      self.assertProtoEquals(result_one, result_two)

  @parameterized.parameters(metadata_store_pb2.ArtifactType,
                            metadata_store_pb2.ContextType,
                            metadata_store_pb2.ExecutionType)
  def testRegisterTypeModifiedKey(self, metadata_type_class):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      type_with_two_properties = _create_type(metadata_type_class)
      common_utils.register_type_if_not_exist(m, type_with_two_properties)
      # Tries to register a type that shares the same name of the type
      # previously registered but with conflicting property types. We expect
      # this to fail.
      type_with_different_properties = metadata_type_class()
      text_format.Parse(
          """
          name: 'my_type'
          properties {
            key: 'p1'
            value: STRING  # This is different from the original registered type
          }
          """, type_with_different_properties)
      with self.assertRaisesRegex(RuntimeError, 'Missing or modified key'):
        common_utils.register_type_if_not_exist(m,
                                                type_with_different_properties)


if __name__ == '__main__':
  tf.test.main()
