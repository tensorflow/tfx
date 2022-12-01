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
from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.portable.mlmd import artifact_lib
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from ml_metadata.proto import metadata_store_pb2

_DEFAULT_ARTIFACT_TYPE = standard_artifacts.Examples


def _create_tfx_artifact(uri: str,
                         state: Optional[str] = None) -> types.Artifact:
  tfx_artifact = _DEFAULT_ARTIFACT_TYPE()
  tfx_artifact.uri = uri
  if state:
    tfx_artifact.state = state
  return tfx_artifact


def _write_tfx_artifacts(
    mlmd_handle: metadata.Metadata,
    tfx_artifacts: Sequence[types.Artifact]) -> Sequence[int]:
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


class ArtifactLibTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    connection_config = metadata_store_pb2.ConnectionConfig()
    connection_config.sqlite.SetInParent()
    mlmd_connection = metadata.Metadata(connection_config=connection_config)
    self._mlmd_handle = self.enter_context(mlmd_connection)

  def testGetArtifactsByIdsSuccessfullyReadsAndDeserializes(self):
    original_artifact = _create_tfx_artifact(
        uri='a/b/c', state=types.artifact.ArtifactState.PENDING)
    original_artifact.set_string_custom_property('foo', 'bar')
    [artifact_id] = _write_tfx_artifacts(self._mlmd_handle, [original_artifact])
    [actual_artifact
    ] = artifact_lib.get_artifacts_by_ids(self._mlmd_handle, [artifact_id])
    self.assertIsInstance(actual_artifact, types.Artifact)
    self.assertEqual(actual_artifact.uri, 'a/b/c')
    self.assertEqual(actual_artifact.id, artifact_id)
    self.assertEqual(actual_artifact.state,
                     types.artifact.ArtifactState.PENDING)
    self.assertEqual(actual_artifact.get_string_custom_property('foo'), 'bar')

  def testGetArtifactsByIdsMissingIdsRaisesError(self):
    tfx_artifacts = [
        _create_tfx_artifact('a/b/1'),
        _create_tfx_artifact('a/b/2'),
    ]
    [artifact_id1, artifact_id2] = _write_tfx_artifacts(self._mlmd_handle,
                                                        tfx_artifacts)
    unknown_artifact_id = artifact_id2 + 1
    with self.assertRaisesRegex(ValueError,
                                'Could not find all MLMD artifacts'):
      artifact_lib.get_artifacts_by_ids(
          self._mlmd_handle, [artifact_id1, unknown_artifact_id, artifact_id2])

  def testUpdateArtifactsWithoutNewState(self):
    artifact1 = _create_tfx_artifact('a/b/1')
    artifact2 = _create_tfx_artifact('a/b/2')
    created_artifact_ids = _write_tfx_artifacts(self._mlmd_handle,
                                                [artifact1, artifact2])

    artifact1.set_string_custom_property('foo', 'bar')
    artifact2.set_string_custom_property('foo', 'bar')

    artifact_lib.update_artifacts(self._mlmd_handle, {
        'key1': [artifact1],
        'key2': [artifact2],
    })

    updated_tfx_artifacts = artifact_lib.get_artifacts_by_ids(
        self._mlmd_handle, created_artifact_ids)
    self.assertLen(updated_tfx_artifacts, 2)
    for tfx_artifact in updated_tfx_artifacts:
      self.assertEqual(tfx_artifact.get_string_custom_property('foo'), 'bar')

  def testUpdateArtifactsWithNewState(self):
    artifact1 = _create_tfx_artifact('a/b/1',
                                     types.artifact.ArtifactState.PENDING)
    artifact2 = _create_tfx_artifact('a/b/2',
                                     types.artifact.ArtifactState.PENDING)
    created_artifact_ids = _write_tfx_artifacts(self._mlmd_handle,
                                                [artifact1, artifact2])

    artifact_lib.update_artifacts(self._mlmd_handle, {
        'key1': [artifact1],
        'key2': [artifact2],
    }, types.artifact.ArtifactState.MISSING)

    updated_tfx_artifacts = artifact_lib.get_artifacts_by_ids(
        self._mlmd_handle, created_artifact_ids)
    self.assertLen(updated_tfx_artifacts, 2)
    for tfx_artifact in updated_tfx_artifacts:
      self.assertEqual(tfx_artifact.state, types.artifact.ArtifactState.MISSING)

  def testUpdateArtifactsWithoutIdRaisesError(self):
    artifact1 = _create_tfx_artifact('x/y/1')
    artifact2 = _create_tfx_artifact('x/y/2')
    _ = _write_tfx_artifacts(self._mlmd_handle, [artifact1, artifact2])
    artifact1.mlmd_artifact.ClearField('id')

    with self.assertRaisesRegex(ValueError, 'Artifact must have an MLMD ID'):
      artifact_lib.update_artifacts(self._mlmd_handle, {
          'key': [artifact1, artifact2],
      })


if __name__ == '__main__':
  tf.test.main()
