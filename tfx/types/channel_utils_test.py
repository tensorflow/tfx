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
"""Tests for tfx.utils.channel."""

import tensorflow as tf
from tfx.types import artifact
from tfx.types import channel
from tfx.types import channel_utils


class _MyArtifact(artifact.Artifact):
  TYPE_NAME = 'MyTypeName'


class ChannelUtilsTest(tf.test.TestCase):

  def testArtifactCollectionAsChannel(self):
    instance_a = _MyArtifact()
    instance_b = _MyArtifact()
    chnl = channel_utils.as_channel([instance_a, instance_b])
    self.assertEqual(chnl.type, _MyArtifact)
    self.assertEqual(chnl.type_name, 'MyTypeName')
    self.assertCountEqual(chnl.get(), [instance_a, instance_b])

  def testEmptyArtifactCollectionAsChannelFail(self):
    with self.assertRaises(ValueError):
      channel_utils.as_channel([])

  def testInvalidSourceAsChannelFail(self):
    with self.assertRaises(ValueError):
      channel_utils.as_channel(artifacts='invalid artifacts')

  def testUnwrapChannelDict(self):
    instance_a = _MyArtifact()
    instance_b = _MyArtifact()
    channel_dict = {
        'id':
            channel.Channel(_MyArtifact).set_artifacts([instance_a, instance_b])
    }
    result = channel_utils.unwrap_channel_dict(channel_dict)
    self.assertDictEqual(result, {'id': [instance_a, instance_b]})

  def testGetInidividualChannels(self):
    instance_a = _MyArtifact()
    instance_b = _MyArtifact()
    one_channel = channel.Channel(_MyArtifact).set_artifacts([instance_a])
    another_channel = channel.Channel(_MyArtifact).set_artifacts([instance_b])

    result = channel_utils.get_individual_channels(one_channel)
    self.assertEqual(result, [one_channel])

    result = channel_utils.get_individual_channels(
        channel.union([one_channel, another_channel]))
    self.assertEqual(result, [one_channel, another_channel])


if __name__ == '__main__':
  tf.test.main()
