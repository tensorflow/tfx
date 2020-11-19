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

import tensorflow as tf
from tfx.orchestration import data_types_utils
from tfx.types import artifact_utils

from google.protobuf import text_format
from ml_metadata.proto import metadata_store_pb2
from ml_metadata.proto import metadata_store_service_pb2


class DataTypesUtilsTest(tf.test.TestCase):

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

    self.exec_property_value_dict = {
        'p1': metadata_store_pb2.Value(int_value=1),
        'p2': metadata_store_pb2.Value(string_value='hello')
    }
    self.exec_property_dict = {'p1': 1, 'p2': 'hello'}

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

  def testBuildExecPropertyDict(self):
    actual_exec_property_dict = data_types_utils.build_exec_property_dict(
        self.exec_property_value_dict)
    self.assertEqual(self.exec_property_dict, actual_exec_property_dict)

  def testBuildExecPropertyValueDict(self):
    actual_exec_property_value_dict = (
        data_types_utils.build_exec_property_value_dict(
            self.exec_property_dict))
    self.assertEqual(self.exec_property_value_dict,
                     actual_exec_property_value_dict)


if __name__ == '__main__':
  tf.test.main()
