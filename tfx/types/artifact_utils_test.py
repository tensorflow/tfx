# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Standard Imports

import tensorflow as tf
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class ArtifactUtilsTest(tf.test.TestCase):

  def testGetFromSingleList(self):
    """Test various retrieval utilities on a single list of Artifact."""
    artifacts = [standard_artifacts.Examples()]
    artifacts[0].uri = '/tmp/evaluri'
    artifacts[0].split_names = '["eval"]'
    self.assertEqual(artifacts[0],
                     artifact_utils.get_single_instance(artifacts))
    self.assertEqual('/tmp/evaluri', artifact_utils.get_single_uri(artifacts))
    self.assertEqual('/tmp/evaluri/eval',
                     artifact_utils.get_split_uri(artifacts, 'eval'))
    with self.assertRaises(ValueError):
      artifact_utils.get_split_uri(artifacts, 'train')

  def testGetFromSplits(self):
    """Test various retrieval utilities on a list of split Artifact."""
    artifacts = [standard_artifacts.Examples()]
    artifacts[0].uri = '/tmp'
    artifacts[0].split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])

    self.assertEqual(artifacts[0].split_names, '["train", "eval"]')

    self.assertIs(artifact_utils.get_single_instance(artifacts), artifacts[0])
    self.assertEqual('/tmp', artifact_utils.get_single_uri(artifacts))
    self.assertEqual('/tmp/train',
                     artifact_utils.get_split_uri(artifacts, 'train'))
    self.assertEqual('/tmp/eval',
                     artifact_utils.get_split_uri(artifacts, 'eval'))


if __name__ == '__main__':
  tf.test.main()
