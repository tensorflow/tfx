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
"""Tests for tfx.components.util.value_utils."""

import tensorflow as tf

from tfx.components.util import value_utils


def DummyFunctionWithNoArg():
  pass


def DummyFunctionWithArgs(arg1: int, arg2: int = 42) -> int:
  return arg1 + arg2


class ValueUtilsTest(tf.test.TestCase):

  def testFunctionHasArg(self):
    self.assertFalse(value_utils.FunctionHasArg(DummyFunctionWithNoArg, 'arg1'))
    self.assertTrue(value_utils.FunctionHasArg(DummyFunctionWithArgs, 'arg1'))
    self.assertTrue(value_utils.FunctionHasArg(DummyFunctionWithArgs, 'arg2'))
    self.assertFalse(value_utils.FunctionHasArg(DummyFunctionWithArgs, 'arg3'))


if __name__ == '__main__':
  tf.test.main()
