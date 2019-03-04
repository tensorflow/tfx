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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Standard Imports

import tensorflow as tf
from tfx.utils import channel
from tfx.utils import types


class ChannelTest(tf.test.TestCase):

  def test_valid_channel(self):
    instance_a = types.TfxType('MyTypeName')
    instance_b = types.TfxType('MyTypeName')
    chnl = channel.Channel(
        'MyTypeName', static_artifact_collection=[instance_a, instance_b])
    self.assertEqual(chnl.type_name, 'MyTypeName')
    self.assertItemsEqual(chnl.get(), [instance_a, instance_b])

  def test_invalid_channel_type(self):
    instance_a = types.TfxType('MyTypeName')
    instance_b = types.TfxType('MyTypeName')
    with self.assertRaises(ValueError):
      channel.Channel(
          'AnotherTypeName',
          static_artifact_collection=[instance_a, instance_b])

  def test_artifact_collection_as_channel(self):
    instance_a = types.TfxType('MyTypeName')
    instance_b = types.TfxType('MyTypeName')
    chnl = channel.as_channel([instance_a, instance_b])
    self.assertEqual(chnl.type_name, 'MyTypeName')
    self.assertItemsEqual(chnl.get(), [instance_a, instance_b])

  def test_channel_as_channel_success(self):
    instance_a = types.TfxType('MyTypeName')
    instance_b = types.TfxType('MyTypeName')
    chnl_original = channel.Channel(
        'MyTypeName', static_artifact_collection=[instance_a, instance_b])
    chnl_result = channel.as_channel(chnl_original)
    self.assertEqual(chnl_original, chnl_result)

  def test_empty_artifact_collection_as_channel_fail(self):
    with self.assertRaises(ValueError):
      channel.as_channel([])

  def test_invalid_source_as_channel_fail(self):
    with self.assertRaises(ValueError):
      channel.as_channel(source='invalid source')

  def test_type_check_success(self):
    chnl = channel.Channel('MyTypeName')
    chnl.type_check('MyTypeName')

  def test_type_check_fail(self):
    chnl = channel.Channel('MyTypeName')
    with self.assertRaises(TypeError):
      chnl.type_check('AnotherTypeName')


if __name__ == '__main__':
  tf.test.main()
