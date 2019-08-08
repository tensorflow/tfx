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
from tfx.types.artifact import Artifact
from tfx.types.channel import Channel


class ChannelTest(tf.test.TestCase):

  def testValidChannel(self):
    instance_a = Artifact('MyTypeName')
    instance_b = Artifact('MyTypeName')
    chnl = Channel('MyTypeName', artifacts=[instance_a, instance_b])
    self.assertEqual(chnl.type_name, 'MyTypeName')
    self.assertItemsEqual(chnl.get(), [instance_a, instance_b])

  def testInvalidChannelType(self):
    instance_a = Artifact('MyTypeName')
    instance_b = Artifact('MyTypeName')
    with self.assertRaises(ValueError):
      Channel('AnotherTypeName', artifacts=[instance_a, instance_b])


if __name__ == '__main__':
  tf.test.main()
