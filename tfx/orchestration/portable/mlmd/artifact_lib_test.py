# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.portable.mlmd.artifact_lib."""

import os

import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import artifact_lib
from tfx.types import artifact_utils

from ml_metadata.proto import metadata_store_pb2


class ArtifactLibTest(tf.test.TestCase):

  def _get_mlmd_handle(self):
    metadata_path = os.path.join(self.get_temp_dir(), 'metadata', 'metadata.db')
    connection_config = metadata.sqlite_metadata_connection_config(
        metadata_path)
    connection_config.sqlite.SetInParent()
    return metadata.Metadata(connection_config=connection_config)

  def testUpdateArtifactForTargetMLMD(self):
    mlmd_handle = self._get_mlmd_handle()
    with mlmd_handle:
      # Runs the function.
      source_artifact_type = metadata_store_pb2.ArtifactType(
          id=1000, name='type-name')
      source_artifact = metadata_store_pb2.Artifact(
          id=2000, type_id=1000, name='artifact-name')
      result_artifact = artifact_lib.update_artifact_for_target_mlmd(
          artifact_utils.deserialize_artifact(source_artifact_type,
                                              source_artifact),
          source_owner='owner',
          source_name='name',
          target_mlmd_handle=mlmd_handle)

      # Tests whether the function has updated the artifact.
      result_artifact_types = mlmd_handle.store.get_artifact_types()
      self.assertLen(result_artifact_types, 1)
      self.assertEqual(result_artifact_types[0].name, 'type-name')
      self.assertEqual(result_artifact.type_id, result_artifact_types[0].id)

  def testUpdateArtifactForTargetMLMD_ArtifactTypeExists(self):
    mlmd_handle = self._get_mlmd_handle()
    with mlmd_handle:
      # Lets the mlmd db has an artifact type.
      existing_artifact_type = metadata_store_pb2.ArtifactType(name='type-name')
      existing_artifact_type_id = mlmd_handle.store.put_artifact_type(
          existing_artifact_type)

      # Runs the function.
      source_artifact_type = metadata_store_pb2.ArtifactType(
          id=1000, name='type-name')
      source_artifact = metadata_store_pb2.Artifact(
          id=2000, type_id=1000, name='artifact-name')
      result_artifact = artifact_lib.update_artifact_for_target_mlmd(
          artifact_utils.deserialize_artifact(source_artifact_type,
                                              source_artifact),
          source_owner='owner',
          source_name='name',
          target_mlmd_handle=mlmd_handle)

      # Tests whether the function has copied the artifact.
      result_artifact_types = mlmd_handle.store.get_artifact_types()
      self.assertLen(result_artifact_types, 1)
      self.assertEqual(result_artifact_types[0].id, existing_artifact_type_id)
      self.assertEqual(result_artifact_types[0].name, 'type-name')
      self.assertEqual(result_artifact.type_id, existing_artifact_type_id)


if __name__ == '__main__':
  tf.test.main()
