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

from unittest import mock

import tensorflow as tf
from tfx.types import standard_artifacts
from tfx.utils import channel
from tfx.utils import deprecation_utils
from tfx.utils import test_case_utils


class ChannelTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._mock_warn = self.enter_context(mock.patch('warnings.warn'))

  def _assertDeprecatedWarningRegex(self, expected_regex):
    self._mock_warn.assert_called()
    (message, warning_cls), unused_kwargs = self._mock_warn.call_args
    self.assertEqual(warning_cls, deprecation_utils.TfxDeprecationWarning)
    self.assertRegex(message, expected_regex)

  def testChannelDeprecated(self):
    channel.Channel(type=standard_artifacts.Examples)
    self._assertDeprecatedWarningRegex(
        'tfx.utils.types.Channel has been renamed to tfx.types.Channel')

  def testAsChannelDeprecated(self):
    channel.as_channel([standard_artifacts.Model()])
    self._assertDeprecatedWarningRegex(
        'tfx.utils.channel.as_channel has been renamed to '
        'tfx.types.channel_utils.as_channel')

  def testUnwrapChannelDictDeprecated(self):
    channel.unwrap_channel_dict({})
    self._assertDeprecatedWarningRegex(
        'tfx.utils.channel.unwrap_channel_dict has been renamed to '
        'tfx.types.channel_utils.unwrap_channel_dict')


if __name__ == '__main__':
  tf.test.main()
