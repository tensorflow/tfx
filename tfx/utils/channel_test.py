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
"""Tests for tfx.utils.types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# Standard Imports

import mock
import tensorflow as tf
from tensorflow.python.platform import tf_logging  # pylint:disable=g-direct-tensorflow-import
from tfx.types import standard_artifacts
from tfx.utils import channel


class ChannelTest(tf.test.TestCase):

  def testChannelDeprecated(self):
    with mock.patch.object(tf_logging, 'warning'):
      warn_mock = mock.MagicMock()
      tf_logging.warning = warn_mock
      channel.Channel(type=standard_artifacts.Examples)
      warn_mock.assert_called_once()
      self.assertIn(
          'tfx.utils.types.Channel has been renamed to tfx.types.Channel',
          warn_mock.call_args[0][5])

  def testAsChannelDeprecated(self):
    with mock.patch.object(tf_logging, 'warning'):
      warn_mock = mock.MagicMock()
      tf_logging.warning = warn_mock
      channel.as_channel([standard_artifacts.Model()])
      warn_mock.assert_called_once()
      self.assertIn('tfx.utils.channel.as_channel has been renamed to',
                    warn_mock.call_args[0][5])

  def testUnwrapChannelDictDeprecated(self):
    with mock.patch.object(tf_logging, 'warning'):
      warn_mock = mock.MagicMock()
      tf_logging.warning = warn_mock
      channel.unwrap_channel_dict({})
      warn_mock.assert_called_once()
      self.assertIn('tfx.utils.channel.unwrap_channel_dict has been renamed to',
                    warn_mock.call_args[0][5])


if __name__ == '__main__':
  tf.test.main()
