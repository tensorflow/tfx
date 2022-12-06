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
"""Tests for tfx.orchestration.portable.merge_utils."""
from typing import Dict, Mapping, Optional, Sequence

from absl.testing import parameterized
import tensorflow as tf
from tfx import types
from tfx.orchestration.portable import merge_utils
from tfx.orchestration.portable import outputs_utils
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils
from tfx.utils import typing_utils

from ml_metadata.proto import metadata_store_pb2

_DEFAULT_ARTIFACT_TYPE = standard_artifacts.Examples
_RUNTIME_RESOLVED_URI = outputs_utils.RESOLVED_AT_RUNTIME


def _tfx_artifact(
    uri: str,
    artifact_id: Optional[int] = None,
    type_id: Optional[int] = None,
    custom_properties: Optional[Dict[str, str]] = None) -> types.Artifact:
  artifact = _DEFAULT_ARTIFACT_TYPE()
  artifact.uri = uri
  artifact.is_external = False
  if artifact_id is not None:
    artifact.id = artifact_id
  if type_id is not None:
    artifact.type_id = type_id
  if custom_properties:
    for key, value in custom_properties.items():
      artifact.set_string_custom_property(key, value)
  return artifact


def _build_output_artifact_dict(
    output_artifacts: typing_utils.ArtifactMultiMap
) -> Mapping[str, Sequence[metadata_store_pb2.Artifact]]:
  return {
      k: [artifact.mlmd_artifact for artifact in artifacts
         ] for k, artifacts in output_artifacts.items()
  }


class MergeUtilsTest(test_case_utils.TfxTest, parameterized.TestCase):

  def testMergeOutputArtifactsAppliesUpdates(self):
    original_artifact = standard_artifacts.Examples()
    original_artifact.uri = '/a/b/c'

    updated_artifact_proto = metadata_store_pb2.Artifact()
    updated_artifact_proto.uri = '/a/b/c/d'
    updated_artifact_proto.properties['span'].int_value = 5

    merged_artifact = merge_utils._merge_output_artifact(
        original_artifact, updated_artifact_proto)

    expected_artifact = standard_artifacts.Examples()
    expected_artifact.uri = '/a/b/c/d'
    expected_artifact.span = 5
    expected_artifact.is_external = False

    self.assertEqual(expected_artifact.uri, merged_artifact.uri)
    self.assertEqual(expected_artifact.span, merged_artifact.span)
    self.assertProtoEquals(expected_artifact.mlmd_artifact,
                           merged_artifact.mlmd_artifact)
    self.assertProtoEquals(expected_artifact.artifact_type,
                           merged_artifact.artifact_type)

  def testMergeOutputArtifactsOverwritesProperties(self):
    original_artifact = standard_artifacts.Examples()
    original_artifact.uri = '/a/b/c'
    original_artifact.span = 4

    updated_artifact_proto = metadata_store_pb2.Artifact()
    updated_artifact_proto.uri = '/a/b/c'
    updated_artifact_proto.properties['span'].int_value = 5

    merged_artifact = merge_utils._merge_output_artifact(
        original_artifact, updated_artifact_proto)

    self.assertEqual(4, original_artifact.span)
    self.assertEqual(5, merged_artifact.span)

  def testMergeOutputArtifacts_PreservesExternalArtifact(self):
    original_artifact = standard_artifacts.Examples()
    original_artifact.uri = '/a/b/c'
    original_artifact.is_external = True
    updated_artifact_proto = metadata_store_pb2.Artifact(uri='/a/b/c')
    merged_artifact = merge_utils._merge_output_artifact(
        original_artifact, updated_artifact_proto)
    self.assertTrue(merged_artifact.is_external)

  def testMergeOutputArtifacts_DoesntAllowComponentToModifyExternality(self):
    original_artifact = standard_artifacts.Examples()
    original_artifact.uri = '/a/b/c'
    updated_artifact_proto = metadata_store_pb2.Artifact(uri='/a/b/c')
    updated_artifact_proto.custom_properties['is_external'].int_value = 1
    merged_artifact = merge_utils._merge_output_artifact(
        original_artifact, updated_artifact_proto)
    self.assertFalse(merged_artifact.is_external)

  def testMergeOutputArtifactsPreservesOriginalArtifactId(self):
    original_artifact = standard_artifacts.Examples()
    updated_artifact_proto = metadata_store_pb2.Artifact(id=2)
    merged_artifact = merge_utils._merge_output_artifact(
        original_artifact, updated_artifact_proto)
    self.assertFalse(merged_artifact.mlmd_artifact.HasField('id'))

    original_artifact.id = 1
    merged_artifact = merge_utils._merge_output_artifact(
        original_artifact, updated_artifact_proto)
    self.assertEqual(1, merged_artifact.id)

  def testMergeOutputArtifactsPreservesOriginalArtifactTypeId(self):
    original_artifact = standard_artifacts.Examples()
    original_artifact.artifact_type.id = 1
    updated_artifact_proto = metadata_store_pb2.Artifact(type_id=2)
    merged_artifact = merge_utils._merge_output_artifact(
        original_artifact, updated_artifact_proto)
    self.assertEqual(1, merged_artifact.type_id)

  @parameterized.named_parameters([
      dict(
          testcase_name='WithoutUpdatedArtifactsPreservesOriginal',
          original_artifacts={
              'key': [_tfx_artifact(uri='foo/bar')],
          },
          updated_artifacts={},
          expected_merged_artifacts={
              'key': [_tfx_artifact(uri='foo/bar')],
          }),
      dict(
          testcase_name='WithoutKeyInUpdatedArtifactsPreservesOriginal',
          original_artifacts={
              'key1': [_tfx_artifact(uri='/a/1')],
              'key2': [_tfx_artifact(uri='/a/2')],
          },
          updated_artifacts={
              'key1': [_tfx_artifact(uri='/a/1')],
          },
          expected_merged_artifacts={
              'key1': [_tfx_artifact(uri='/a/1')],
              'key2': [_tfx_artifact(uri='/a/2')],
          }),
      dict(
          testcase_name='AppliesUpdatedProperties',
          original_artifacts={
              'key1': [
                  _tfx_artifact(uri='/a/1', custom_properties={'foo': 'bar'})
              ],
          },
          updated_artifacts={
              'key1': [
                  _tfx_artifact(uri='/a/1', custom_properties={'bar': 'foo'})
              ],
          },
          expected_merged_artifacts={
              'key1': [
                  _tfx_artifact(uri='/a/1', custom_properties={'bar': 'foo'})
              ],
          }),
      dict(
          testcase_name='RuntimeResolvedUriIsRemovedFromOriginalDict',
          original_artifacts={
              'key1': [_tfx_artifact(uri='/a/1')],
              'key2': [_tfx_artifact(uri=_RUNTIME_RESOLVED_URI)],
          },
          updated_artifacts={
              'key1': [_tfx_artifact(uri='/a/1')],
          },
          expected_merged_artifacts={
              'key1': [_tfx_artifact(uri='/a/1')],
              'key2': [],
          }),
      dict(
          testcase_name='RuntimeResolvedUriIsRemovedFromUpdatedDict',
          original_artifacts={
              'key1': [_tfx_artifact(uri='/a/1')],
              'key2': [_tfx_artifact(uri=_RUNTIME_RESOLVED_URI)],
          },
          updated_artifacts={
              'key1': [_tfx_artifact(uri='/a/1')],
              'key2': [_tfx_artifact(uri=_RUNTIME_RESOLVED_URI)],
          },
          expected_merged_artifacts={
              'key1': [_tfx_artifact(uri='/a/1')],
              'key2': [],
          }),
      dict(
          testcase_name='ReusesOriginalIdIfUriMatchesUpdatedArtifact',
          original_artifacts={
              'key1': [_tfx_artifact(uri='/x/1', artifact_id=1)],
              'key2': [_tfx_artifact(uri='/x/2', artifact_id=2)],
          },
          updated_artifacts={
              'key1': [_tfx_artifact(uri='/x/1', artifact_id=1)],
              'key2': [
                  _tfx_artifact(uri='/x/2/a', artifact_id=2),
                  _tfx_artifact(uri='/x/2/b')
              ],
          },
          expected_merged_artifacts={
              'key1': [_tfx_artifact(uri='/x/1', artifact_id=1)],
              'key2': [
                  _tfx_artifact(uri='/x/2/a'),
                  _tfx_artifact(uri='/x/2/b')
              ],
          }),
      dict(
          testcase_name='UpdatedDictOmitsArtifactInListIsOmittedInResult',
          original_artifacts={
              'key1': [_tfx_artifact(uri='/x/1')],
              'key2': [_tfx_artifact(uri='/x/2')],
          },
          updated_artifacts={
              'key1': [_tfx_artifact(uri='/x/1')],
              'key2': [],
          },
          expected_merged_artifacts={
              'key1': [_tfx_artifact(uri='/x/1')],
              'key2': [],
          }),
  ])
  def testMergeOutputArtifacts(
      self, original_artifacts: typing_utils.ArtifactMultiMap,
      updated_artifacts: typing_utils.ArtifactMultiMap,
      expected_merged_artifacts: typing_utils.ArtifactMultiMap):
    merged_output_artifacts = merge_utils.merge_updated_output_artifacts(
        original_artifacts, _build_output_artifact_dict(updated_artifacts))
    self.assertArtifactMapsEqual(expected_merged_artifacts,
                                 merged_output_artifacts)

  def testMergeOutputArtifactsUpdatedDictChangesArtifactTypeRaisesError(self):
    original_artifacts = {'key1': [_tfx_artifact(uri='/x/1', type_id=1)]}
    updated_artifacts = {'key1': [_tfx_artifact(uri='/x/2', type_id=2)]}
    with self.assertRaisesRegex(
        RuntimeError, 'Executor output should not change artifact type'):
      merge_utils.merge_updated_output_artifacts(
          original_artifacts, _build_output_artifact_dict(updated_artifacts))

  def testMergeOutputArtifactsUnrecognizedKeyInUpdatedDictRaisesError(self):
    original_artifacts = {'key1': [_tfx_artifact(uri='/x/1')]}
    updated_artifacts = {'key2': [_tfx_artifact(uri='/x/2')]}
    with self.assertRaisesRegex(RuntimeError,
                                'contains more keys than output skeleton'):
      merge_utils.merge_updated_output_artifacts(
          original_artifacts, _build_output_artifact_dict(updated_artifacts))

  def testMergeOutputArtifactsUpdatedArtifactUriNotSubdirectoryRaisesError(
      self):
    original_artifacts = {'key1': [_tfx_artifact(uri='/x/1')]}
    updated_artifacts = {
        'key1': [_tfx_artifact(uri='/y/1'),
                 _tfx_artifact(uri='/y/2')]
    }
    with self.assertRaisesRegex(RuntimeError,
                                'URIs should be direct sub-directories'):
      merge_utils.merge_updated_output_artifacts(
          original_artifacts, _build_output_artifact_dict(updated_artifacts))


if __name__ == '__main__':
  tf.test.main()
