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
"""Tests for tfx.orchestration.experimental.interactive.notebook_formatters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import tensorflow as tf

from tfx import types
from tfx.orchestration.experimental.interactive import notebook_formatters
from tfx.types import standard_artifacts


class NotebookFormattersTest(tf.test.TestCase):

  def _format(self, obj):
    for cls in obj.__class__.mro():
      if cls in notebook_formatters.FORMATTER_REGISTRY:
        formatter = notebook_formatters.FORMATTER_REGISTRY[cls]
        return formatter.render(obj)

  def testBasicFormatter(self):
    # Basic artifact.
    examples = standard_artifacts.Examples()
    examples.uri = '/tmp/123'
    self.assertIsNotNone(
        re.search('Artifact.*of type.*Examples.*/tmp/123',
                  self._format(examples)))

    # Channel containing artifact.
    channel = types.Channel(
        type=standard_artifacts.Examples,
        artifacts=[examples])
    self.assertIsNotNone(
        re.search(('.*Channel.*of type.*Examples'
                   '(.|\n)*Artifact.*of type.*Examples'),
                  self._format(channel)))

  def testFormatterTypeCheck(self):
    formatter = notebook_formatters.FORMATTER_REGISTRY[types.Artifact]
    with self.assertRaisesRegexp(
        ValueError,
        'Expected object of type .*Artifact.* but got .*object object'):
      formatter.render(object())

if __name__ == '__main__':
  tf.test.main()
