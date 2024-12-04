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

from absl.testing import absltest
from tfx.dsl.components.base.testing import test_node
from tfx.dsl.placeholder import placeholder as ph
from tfx.types import artifact
from tfx.types import channel
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class _MyArtifact(artifact.Artifact):
  TYPE_NAME = 'MyTypeName'


class ChannelUtilsTest(absltest.TestCase):

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
    one_channel = channel.OutputChannel(
        artifact_type=_MyArtifact,
        producer_component=test_node.TestNode('a'),
        output_key='foo',
    )
    another_channel = channel.OutputChannel(
        artifact_type=_MyArtifact,
        producer_component=test_node.TestNode('b'),
        output_key='bar',
    )

    result = channel_utils.get_individual_channels(one_channel)
    self.assertEqual(result, [one_channel])

    result = channel_utils.get_individual_channels(
        channel.union([one_channel, another_channel]))
    self.assertEqual(result, [one_channel, another_channel])

  def testPredicateDependentChannels(self):
    int1 = channel.OutputChannel(
        artifact_type=standard_artifacts.Integer,
        producer_component=test_node.TestNode('a'),
        output_key='foo',
    )
    int2 = channel.OutputChannel(
        artifact_type=standard_artifacts.Integer,
        producer_component=test_node.TestNode('b'),
        output_key='bar',
    )
    pred1 = int1.future().value == 1
    pred2 = int1.future().value == int2.future().value
    pred3 = ph.logical_not(pred1)
    pred4 = ph.logical_and(pred1, pred2)

    self.assertEqual(set(channel_utils.get_dependent_channels(pred1)), {int1})
    self.assertEqual(
        set(channel_utils.get_dependent_channels(pred2)), {int1, int2}
    )
    self.assertEqual(set(channel_utils.get_dependent_channels(pred3)), {int1})
    self.assertEqual(
        set(channel_utils.get_dependent_channels(pred4)), {int1, int2}
    )

  def testUnwrapSimpleChannelPlaceholder(self):
    int1 = channel.OutputChannel(
        artifact_type=standard_artifacts.Integer,
        producer_component=test_node.TestNode('a'),
        output_key='foo',
    )
    self.assertEqual(
        channel_utils.unwrap_simple_channel_placeholder(int1.future()[0].value),
        int1,
    )
    self.assertEqual(
        channel_utils.unwrap_simple_channel_placeholder(int1.future().value),
        int1,
    )

  def testUnwrapSimpleChannelPlaceholderRejectsMultiChannel(self):
    str1 = channel.OutputChannel(
        artifact_type=standard_artifacts.String,
        producer_component=test_node.TestNode('a'),
        output_key='foo',
    )
    str2 = channel.OutputChannel(
        artifact_type=standard_artifacts.String,
        producer_component=test_node.TestNode('b'),
        output_key='bar',
    )
    with self.assertRaisesRegex(ValueError, '.*placeholder of shape.*'):
      channel_utils.unwrap_simple_channel_placeholder(
          str1.future()[0].value + str2.future()[0].value
      )
    with self.assertRaisesRegex(ValueError, '.*placeholder of shape.*'):
      channel_utils.unwrap_simple_channel_placeholder(
          ph.join([str1.future()[0].value, str2.future()[0].value], ',')
      )

  def testUnwrapSimpleChannelPlaceholderRejectsNoChannel(self):
    with self.assertRaisesRegex(ValueError, '.*placeholder of shape.*'):
      channel_utils.unwrap_simple_channel_placeholder(ph.make_list([]))
    with self.assertRaisesRegex(ValueError, '.*placeholder of shape.*'):
      channel_utils.unwrap_simple_channel_placeholder(ph.input('disallowed'))
    with self.assertRaisesRegex(ValueError, '.*placeholder of shape.*'):
      channel_utils.unwrap_simple_channel_placeholder(ph.output('disallowed'))

  def testUnwrapSimpleChannelPlaceholderRejectsComplexPlaceholders(self):
    str1 = channel.OutputChannel(
        artifact_type=standard_artifacts.String,
        producer_component=test_node.TestNode('a'),
        output_key='foo',
    )
    with self.assertRaisesRegex(ValueError, '.*placeholder of shape.*'):
      channel_utils.unwrap_simple_channel_placeholder(
          str1.future()[0].value + 'foo'
      )
    with self.assertRaisesRegex(ValueError, '.*placeholder of shape.*'):
      channel_utils.unwrap_simple_channel_placeholder(
          str1.future()[0].value + ph.execution_invocation().pipeline_run_id
      )
