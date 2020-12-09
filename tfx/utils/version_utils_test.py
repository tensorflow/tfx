# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Tests for tfx.utils.version_utils."""

import tensorflow as tf
from tfx.utils import version_utils


class VersionUtilsTest(tf.test.TestCase):

  def testImageVersion(self):
    self.assertEqual(version_utils.get_image_version('0.25.0'), '0.25.0')
    self.assertEqual(version_utils.get_image_version('0.25.0-rc1'), '0.25.0rc1')
    self.assertEqual(
        version_utils.get_image_version('0.25.0.dev20201101'),
        '0.25.0.dev20201101')
    self.assertEqual(version_utils.get_image_version('0.26.0.dev'), 'latest')


if __name__ == '__main__':
  tf.test.main()
