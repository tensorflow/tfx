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
from typing import Optional, Sequence

import tensorflow as tf
from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import artifact_lib
from tfx.types import artifact as tfx_artifact_lib
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from ml_metadata.proto import metadata_store_pb2

_DEFAULT_ARTIFACT_TYPE = standard_artifacts.Examples


def _create_tfx_artifact(uri: str,
                         state: Optional[str] = None
                        ) -> tfx_artifact_lib.Artifact:
  tfx_artifact = _DEFAULT_ARTIFACT_TYPE()
  tfx_artifact.uri = uri
  if state:
    tfx_artifact.state = state
  return tfx_artifact


def _write_tfx_artifacts(
    mlmd_handle: metadata.Metadata,
    tfx_artifacts: Sequence[tfx_artifact_lib.Artifact]) -> Sequence[int]:
  """Writes TFX artifacts to MLMD and updates their IDs in-place."""
  artifact_type_id = mlmd_handle.store.put_artifact_type(
      artifact_type=metadata_store_pb2.ArtifactType(
          name=_DEFAULT_ARTIFACT_TYPE.TYPE_NAME))
  for tfx_artifact in tfx_artifacts:
    tfx_artifact.type_id = artifact_type_id

  created_artifact_ids = mlmd_handle.store.put_artifacts(
      [tfx_artifact.mlmd_artifact for tfx_artifact in tfx_artifacts])
  for idx, artifact_id in enumerate(created_artifact_ids):
    tfx_artifacts[idx].id = artifact_id
  return created_artifact_ids


def _read_tfx_artifacts(
    mlmd_handle: metadata.Metadata,
    artifact_ids: Sequence[int]) -> Sequence[tfx_artifact_lib.Artifact]:
  """Reads TFX artifacts from MLMD by artifact ID."""
  mlmd_artifacts = mlmd_handle.store.get_artifacts_by_id(artifact_ids)
  mlmd_artifact_types = mlmd_handle.store.get_artifact_types_by_id(
      [mlmd_artifact.type_id for mlmd_artifact in mlmd_artifacts])
  mlmd_artifact_type_by_id = {
      artifact_type.id: artifact_type for artifact_type in mlmd_artifact_types
  }
  return [
      artifact_utils.deserialize_artifact(
          mlmd_artifact_type_by_id[mlmd_artifact.type_id], mlmd_artifact)
      for mlmd_artifact in mlmd_artifacts
  ]


class ArtifactLibTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()

  def testUpdateArtifactsWithoutNewState(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      tfx_artifacts = [
          _create_tfx_artifact('a/b/1'),
          _create_tfx_artifact('a/b/2'),
      ]
      created_artifact_ids = _write_tfx_artifacts(m, tfx_artifacts)

      for tfx_artifact in tfx_artifacts:
        tfx_artifact.set_string_custom_property('foo', 'bar')

      artifact_lib.update_artifacts(m, tfx_artifacts)

      updated_tfx_artifacts = _read_tfx_artifacts(m, created_artifact_ids)
      self.assertLen(updated_tfx_artifacts, len(tfx_artifacts))
      for tfx_artifact in updated_tfx_artifacts:
        self.assertEqual(tfx_artifact.get_string_custom_property('foo'), 'bar')

  def testUpdateArtifactsWithNewState(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      tfx_artifacts = [
          _create_tfx_artifact('foo/1', tfx_artifact_lib.ArtifactState.PENDING),
          _create_tfx_artifact('foo/2', tfx_artifact_lib.ArtifactState.PENDING),
      ]
      created_artifact_ids = _write_tfx_artifacts(m, tfx_artifacts)

      artifact_lib.update_artifacts(m, tfx_artifacts,
                                    tfx_artifact_lib.ArtifactState.MISSING)

      updated_tfx_artifacts = _read_tfx_artifacts(m, created_artifact_ids)
      self.assertLen(updated_tfx_artifacts, len(tfx_artifacts))
      for tfx_artifact in updated_tfx_artifacts:
        self.assertEqual(tfx_artifact.state,
                         tfx_artifact_lib.ArtifactState.MISSING)

  def testUpdateArtifactsWithoutIdRaisesError(self):
    with metadata.Metadata(connection_config=self._connection_config) as m:
      tfx_artifacts = [
          _create_tfx_artifact('x/y/1'),
          _create_tfx_artifact('x/y/2'),
      ]
      _ = _write_tfx_artifacts(m, tfx_artifacts)
      tfx_artifacts[1].mlmd_artifact.ClearField('id')

      with self.assertRaisesRegex(ValueError, 'Artifact must have an MLMD ID'):
        artifact_lib.update_artifacts(m, tfx_artifacts)


if __name__ == '__main__':
  tf.test.main()
