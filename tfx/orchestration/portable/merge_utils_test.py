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

import tensorflow as tf
from tfx.orchestration.portable import merge_utils
from tfx.types import standard_artifacts

from ml_metadata.proto import metadata_store_pb2


class MergeUtilsTest(tf.test.TestCase):

  def testMergeOutputArtifactsAppliesUpdates(self):
    original_artifact = standard_artifacts.Examples()
    original_artifact.uri = '/a/b/c'

    updated_artifact_proto = metadata_store_pb2.Artifact()
    updated_artifact_proto.uri = '/a/b/c/d'
    updated_artifact_proto.properties['span'].int_value = 5

    merged_artifact = merge_utils.merge_output_artifact(
        original_artifact, updated_artifact_proto)

    expected_artifact = standard_artifacts.Examples()
    expected_artifact.uri = '/a/b/c/d'
    expected_artifact.span = 5

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

    merged_artifact = merge_utils.merge_output_artifact(
        original_artifact, updated_artifact_proto)

    self.assertEqual(4, original_artifact.span)
    self.assertEqual(5, merged_artifact.span)


if __name__ == '__main__':
  tf.test.main()
