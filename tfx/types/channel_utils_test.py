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
from tfx.dsl.placeholder import placeholder as ph
from tfx.proto.orchestration import placeholder_pb2
from tfx.types import artifact
from tfx.types import channel
from tfx.types import channel_utils
from tfx.types import standard_artifacts


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

  def testPredicateDependentChannels(self):
    int1 = channel.Channel(type=standard_artifacts.Integer)
    int2 = channel.Channel(type=standard_artifacts.Integer)
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
    int1 = channel.Channel(type=standard_artifacts.Integer)
    self.assertEqual(
        channel_utils.unwrap_simple_channel_placeholder(int1.future()[0].value),
        int1,
    )
    self.assertEqual(
        channel_utils.unwrap_simple_channel_placeholder(int1.future().value),
        int1,
    )

  def testUnwrapSimpleChannelPlaceholderRejectsMultiChannel(self):
    str1 = channel.Channel(type=standard_artifacts.String)
    str2 = channel.Channel(type=standard_artifacts.String)
    with self.assertRaisesRegex(ValueError, '.*single channel.*'):
      channel_utils.unwrap_simple_channel_placeholder(
          str1.future()[0].value + str2.future()[0].value
      )
    with self.assertRaisesRegex(ValueError, '.*single channel.*'):
      channel_utils.unwrap_simple_channel_placeholder(
          ph.join([str1.future()[0].value, str2.future()[0].value], ',')
      )

  def testUnwrapSimpleChannelPlaceholderRejectsNoChannel(self):
    # To test the no-channels case, we need to provide _some_ kind of
    # placeholder that is not a ChannelWrappedPlaceholder but also not
    # denylisted. We do that with a custom class declared in this test,
    # although the _real_ behavior we're testing for is rejection of
    # placeholders that are declared in the future or elsewhere and are thus not
    # covered by the denylist.
    class TestPlaceholder(ph.Placeholder):

      def __init__(self):
        super().__init__(placeholder_pb2.Placeholder.Type.RUNTIME_INFO)

    with self.assertRaisesRegex(ValueError, '.*no channels.*'):
      channel_utils.unwrap_simple_channel_placeholder(TestPlaceholder())

  def testUnwrapSimpleChannelPlaceholderRejectsComplexPlaceholders(self):
    str1 = channel.Channel(type=standard_artifacts.String)
    with self.assertRaisesRegex(ValueError, '.*complex.*'):
      channel_utils.unwrap_simple_channel_placeholder(ph.to_list([]))
    with self.assertRaisesRegex(ValueError, '.*complex.*'):
      channel_utils.unwrap_simple_channel_placeholder(ph.input('disallowed'))
    with self.assertRaisesRegex(ValueError, '.*complex.*'):
      channel_utils.unwrap_simple_channel_placeholder(ph.output('disallowed'))
    with self.assertRaisesRegex(ValueError, '.*complex.*'):
      channel_utils.unwrap_simple_channel_placeholder(
          str1.future()[0].value + ph.execution_invocation().pipeline_run_id
      )


if __name__ == '__main__':
  tf.test.main()
