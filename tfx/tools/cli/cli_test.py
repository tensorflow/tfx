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
"""Tests for tfx.tools.cli.cli."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import locale
import os

from click.testing import CliRunner
import tensorflow as tf
from tfx.tools.cli.cli import cli


class CliTest(tf.test.TestCase):

  def setUp(self):
    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'
    self.runner = CliRunner()

  def test_cli_pipeline(self):
    result = self.runner.invoke(cli, ['pipeline'])
    self.assertIn('CLI', result.output)

  def test_cli_invalid_command(self):
    result = self.runner.invoke(cli, ['pipelin'])
    self.assertNotEqual(0, result.exit_code)


if __name__ == '__main__':
  tf.test.main()
