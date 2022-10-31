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

from unittest import mock

import tensorflow as tf
from tfx.dsl.input_resolution import resolver_op
from tfx.types import artifact
from tfx.types import channel
from tfx.types import channel_utils
from tfx.types import resolved_channel


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
    one_channel = channel.Channel(_MyArtifact)
    another_channel = channel.Channel(_MyArtifact)

    result = channel_utils.get_individual_channels(one_channel)
    self.assertEqual(result, [one_channel])

    result = channel_utils.get_individual_channels(
        channel.union([one_channel, another_channel]))
    self.assertEqual(result, [one_channel, another_channel])

  def testGetDependentNodeIds(self):
    x1 = mock.MagicMock()
    x1.id = 'x1'
    x2 = mock.MagicMock()
    x2.id = 'x2'
    p = mock.MagicMock()
    p.id = 'p'

    just_channel = channel.Channel(type=_MyArtifact)
    output_channel_x1 = channel.OutputChannel(
        artifact_type=_MyArtifact, producer_component=x1,
        output_key='out1')
    output_channel_x2 = channel.OutputChannel(
        artifact_type=_MyArtifact, producer_component=x2,
        output_key='out1')
    pipeline_input_channel = channel.PipelineInputChannel(
        output_channel_x1, output_key='out2')
    pipeline_output_channel = channel.PipelineOutputChannel(
        output_channel_x2, p, output_key='out3')
    pipeline_input_channel.pipeline = p
    union_channel = channel.union([output_channel_x1, output_channel_x2])
    resolved_channel_ = resolved_channel.ResolvedChannel(
        _MyArtifact, resolver_op.InputNode(
            union_channel, output_data_type=resolver_op.DataType.ARTIFACT_LIST))

    class DummyChannel(channel.BaseChannel):
      pass

    unknown_channel = DummyChannel(_MyArtifact)

    def check(ch, expected):
      with self.subTest(channel_type=type(ch).__name__):
        if isinstance(expected, list):
          actual = list(channel_utils.get_dependent_node_ids(ch))
          self.assertCountEqual(
              actual, expected, f'Expected {expected} but got {actual}.')
        else:
          with self.assertRaises(expected):
            list(channel_utils.get_dependent_node_ids(ch))

    check(just_channel, [])
    check(output_channel_x1, ['x1'])
    check(output_channel_x2, ['x2'])
    check(pipeline_input_channel, ['p'])
    check(pipeline_output_channel, ['p'])
    check(union_channel, ['x1', 'x2'])
    check(resolved_channel_, ['x1', 'x2'])
    check(unknown_channel, TypeError)


if __name__ == '__main__':
  tf.test.main()
