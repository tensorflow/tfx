# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Tests for tfx.types.artifact_utils."""

import copy
from unittest import mock


from absl import logging
import tensorflow as tf
from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.types import standard_artifacts

from ml_metadata.proto import metadata_store_pb2


class _MyArtifact(artifact.Artifact):
  TYPE_NAME = 'ArtifactUtilsTypeName'
  PROPERTIES = {
      'dummy_int': artifact.Property(artifact.PropertyType.INT),
      'dummy_string': artifact.Property(artifact.PropertyType.STRING),
  }


class ArtifactUtilsTest(tf.test.TestCase):

  def testGetFromSingleList(self):
    """Test various retrieval utilities on a single list of Artifact."""
    artifacts = [standard_artifacts.Examples()]
    artifacts[0].uri = '/tmp/evaluri'
    artifacts[0].split_names = '["eval"]'
    self.assertEqual(artifacts[0],
                     artifact_utils.get_single_instance(artifacts))
    self.assertEqual('/tmp/evaluri', artifact_utils.get_single_uri(artifacts))
    self.assertEqual('/tmp/evaluri/Split-eval',
                     artifact_utils.get_split_uri(artifacts, 'eval'))
    with self.assertRaises(ValueError):
      artifact_utils.get_split_uri(artifacts, 'train')

  def testReplicateArtifacts(self):
    an_artifact = standard_artifacts.Examples()
    an_artifact.uri = '/tmp/evaluri'
    an_artifact.split_names = '["eval"]'
    replicated = artifact_utils.replicate_artifacts(an_artifact, 4)
    self.assertLen(replicated, 4)
    self.assertEqual(replicated[0].uri, '/tmp/evaluri/0')
    self.assertEqual(replicated[3].uri, '/tmp/evaluri/3')
    self.assertEqual(replicated[0].split_names, an_artifact.split_names)
    self.assertEqual(replicated[0].split_names, an_artifact.split_names)

  def testGetFromSplits(self):
    """Test various retrieval utilities on a list of split Artifact."""
    artifacts = [standard_artifacts.Examples()]
    artifacts[0].uri = '/tmp'
    artifacts[0].split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])

    self.assertEqual(artifacts[0].split_names, '["train", "eval"]')

    self.assertIs(artifact_utils.get_single_instance(artifacts), artifacts[0])
    self.assertEqual('/tmp', artifact_utils.get_single_uri(artifacts))
    self.assertEqual('/tmp/Split-train',
                     artifact_utils.get_split_uri(artifacts, 'train'))
    self.assertEqual('/tmp/Split-eval',
                     artifact_utils.get_split_uri(artifacts, 'eval'))

  def testGetFromSplitsMultipleArtifacts(self):
    """Test split retrieval utility on a multiple list of split Artifacts."""
    artifacts = [standard_artifacts.Examples(), standard_artifacts.Examples()]
    artifacts[0].uri = '/tmp1'
    artifacts[0].split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    artifacts[1].uri = '/tmp2'
    artifacts[1].split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    # When creating new splits, use 'Split-<split_name>' format.
    self.assertEqual(['/tmp1/Split-train', '/tmp2/Split-train'],
                     artifact_utils.get_split_uris(artifacts, 'train'))
    self.assertEqual(['/tmp1/Split-eval', '/tmp2/Split-eval'],
                     artifact_utils.get_split_uris(artifacts, 'eval'))
    # When reading artifacts without version.
    artifacts[0].mlmd_artifact.state = metadata_store_pb2.Artifact.LIVE
    artifacts[1].mlmd_artifact.state = metadata_store_pb2.Artifact.LIVE
    self.assertEqual(['/tmp1/train', '/tmp2/train'],
                     artifact_utils.get_split_uris(artifacts, 'train'))
    self.assertEqual(['/tmp1/eval', '/tmp2/eval'],
                     artifact_utils.get_split_uris(artifacts, 'eval'))
    # When reading artifacts with old version.
    artifacts[0].set_string_custom_property(
        artifact_utils.ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY, '0.1')
    artifacts[1].set_string_custom_property(
        artifact_utils.ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY, '0.1')
    self.assertEqual(['/tmp1/train', '/tmp2/train'],
                     artifact_utils.get_split_uris(artifacts, 'train'))
    self.assertEqual(['/tmp1/eval', '/tmp2/eval'],
                     artifact_utils.get_split_uris(artifacts, 'eval'))
    # When reading artifacts with new version.
    artifacts[0].set_string_custom_property(
        artifact_utils.ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY,
        artifact_utils._ARTIFACT_VERSION_FOR_SPLIT_UPDATE)
    artifacts[1].set_string_custom_property(
        artifact_utils.ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY,
        artifact_utils._ARTIFACT_VERSION_FOR_SPLIT_UPDATE)
    self.assertEqual(['/tmp1/Split-train', '/tmp2/Split-train'],
                     artifact_utils.get_split_uris(artifacts, 'train'))
    self.assertEqual(['/tmp1/Split-eval', '/tmp2/Split-eval'],
                     artifact_utils.get_split_uris(artifacts, 'eval'))

  def testArtifactTypeRoundTrip(self):
    mlmd_artifact_type = standard_artifacts.Examples._get_artifact_type()
    self.assertIs(standard_artifacts.Examples,
                  artifact_utils.get_artifact_type_class(mlmd_artifact_type))
    mlmd_artifact_type = _MyArtifact._get_artifact_type()
    # Test that the ID is ignored for type comparison purposes during
    # deserialization.
    mlmd_artifact_type.id = 123
    self.assertIs(_MyArtifact,
                  artifact_utils.get_artifact_type_class(mlmd_artifact_type))

  @mock.patch.object(logging, 'warning', autospec=True)
  def testArtifactTypeRoundTripUnknownArtifactClass(self, mock_warning):
    mlmd_artifact_type = copy.deepcopy(
        standard_artifacts.Examples._get_artifact_type())
    self.assertIs(standard_artifacts.Examples,
                  artifact_utils.get_artifact_type_class(mlmd_artifact_type))
    mlmd_artifact_type.name = 'UnknownTypeName'

    reconstructed_class = artifact_utils.get_artifact_type_class(
        mlmd_artifact_type)
    mock_warning.assert_called_once()

    self.assertIsNot(standard_artifacts.Examples, reconstructed_class)
    self.assertTrue(issubclass(reconstructed_class, artifact.Artifact))
    self.assertEqual('UnknownTypeName', reconstructed_class.TYPE_NAME)
    self.assertEqual(mlmd_artifact_type,
                     reconstructed_class._get_artifact_type())

  def testIsArtifactVersionOlderThan(self):
    examples = standard_artifacts.Examples()
    self.assertFalse(
        artifact_utils.is_artifact_version_older_than(examples, '0.1'))
    examples.mlmd_artifact.state = metadata_store_pb2.Artifact.LIVE
    self.assertTrue(
        artifact_utils.is_artifact_version_older_than(examples, '0.1'))
    examples.set_string_custom_property(
        artifact_utils.ARTIFACT_TFX_VERSION_CUSTOM_PROPERTY_KEY, '0.2')
    self.assertTrue(
        artifact_utils.is_artifact_version_older_than(examples, '0.10'))
    self.assertTrue(
        artifact_utils.is_artifact_version_older_than(examples, '0.3'))
    self.assertFalse(
        artifact_utils.is_artifact_version_older_than(examples, '0.2'))
    self.assertFalse(
        artifact_utils.is_artifact_version_older_than(examples, '0.1'))


if __name__ == '__main__':
  tf.test.main()
