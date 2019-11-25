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
"""Tests for TFX RuntimeStringParameter type."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text
import tensorflow as tf

from tfx.orchestration.experimental.runtime_parameter import runtime_string_parameter


class RuntimeStringParameterTest(tf.test.TestCase):

  def test_sweetpath(self):
    parameter = runtime_string_parameter.RuntimeStringParameter(
        name='test-parameter',
        default=u'test-default',
        ptype=Text,
        description='test-description')
    parsed = runtime_string_parameter.RuntimeStringParameter.parse(parameter)
    self.assertEqual(parsed.name, 'test-parameter')
    self.assertEqual(parsed.default, 'test-default')
    self.assertEqual(parsed.ptype, Text)
    self.assertEqual(parsed.description, 'test-description')

  def test_non_string_ptype(self):
    with self.assertRaises(TypeError):
      parameter = runtime_string_parameter.RuntimeStringParameter(  # pylint: disable=unused-variable
          name='test-parameter', ptype=int)

  def test_mismatch_default_ptype(self):
    with self.assertRaises(TypeError):
      parameter = runtime_string_parameter.RuntimeStringParameter(  # pylint: disable=unused-variable
          name='test-parameter', ptype=Text, default=42)


if __name__ == '__main__':
  tf.test.main()
