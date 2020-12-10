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
"""Tests for tfx.tools.cli.pip_utils."""

import subprocess
from unittest import mock

import tensorflow as tf

from tfx.tools.cli import pip_utils


_SAMPLE_PIP_FREEZE_RESULT = b"""# some comment.

-f a link
-e .
absl-py==0.10.0
aiohttp==3.7.3
alembic==1.4.3"""


class PipUtilsTest(tf.test.TestCase):

  @mock.patch.object(
      subprocess,
      'check_output',
      return_value=_SAMPLE_PIP_FREEZE_RESULT,
      autospec=True)
  def test_get_package_names(self, mock_subprocess):
    self.assertSameElements(pip_utils.get_package_names(),
                            ['absl-py', 'aiohttp', 'alembic'])
    mock_subprocess.assert_called_once()


if __name__ == '__main__':
  tf.test.main()
