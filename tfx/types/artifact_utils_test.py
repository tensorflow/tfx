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
from tfx.types import artifact
from tfx.types import artifact_utils


class ArtifactUtilsTest(tf.test.TestCase):

  def testGetFromSingleList(self):
    """Test various retrieval utilities on a single list of Artifact."""
    single_list = [artifact.Artifact('MyTypeName', split='eval')]
    single_list[0].uri = '/tmp/evaluri'
    self.assertEqual(single_list[0],
                     artifact_utils.get_single_instance(single_list))
    self.assertEqual('/tmp/evaluri', artifact_utils.get_single_uri(single_list))
    self.assertEqual(single_list[0],
                     artifact_utils._get_split_instance(single_list, 'eval'))
    self.assertEqual('/tmp/evaluri',
                     artifact_utils.get_split_uri(single_list, 'eval'))
    with self.assertRaises(ValueError):
      artifact_utils._get_split_instance(single_list, 'train')
    with self.assertRaises(ValueError):
      artifact_utils.get_split_uri(single_list, 'train')

  def testGetFromSplitList(self):
    """Test various retrieval utilities on a list of split Artifact."""
    split_list = []
    for split in ['train', 'eval']:
      instance = artifact.Artifact('MyTypeName', split=split)
      instance.uri = '/tmp/' + split
      split_list.append(instance)

    with self.assertRaises(ValueError):
      artifact_utils.get_single_instance(split_list)

    with self.assertRaises(ValueError):
      artifact_utils.get_single_uri(split_list)

    self.assertEqual(split_list[0],
                     artifact_utils._get_split_instance(split_list, 'train'))
    self.assertEqual('/tmp/train',
                     artifact_utils.get_split_uri(split_list, 'train'))
    self.assertEqual(split_list[1],
                     artifact_utils._get_split_instance(split_list, 'eval'))
    self.assertEqual('/tmp/eval',
                     artifact_utils.get_split_uri(split_list, 'eval'))


if __name__ == '__main__':
  tf.test.main()
