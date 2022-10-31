# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.experimental.interactive.notebook_utils."""
import builtins

import tensorflow as tf
from tfx.orchestration.experimental.interactive import notebook_utils


class NotebookUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    builtins.__dict__['__IPYTHON__'] = True

  def testRequiresIPythonExecutes(self):
    self.foo_called = False
    def foo():
      self.foo_called = True

    notebook_utils.requires_ipython(foo)()
    self.assertTrue(self.foo_called)

  def testRequiresIPythonNoOp(self):
    del builtins.__dict__['__IPYTHON__']

    self.foo_called = False
    def foo():
      self.foo_called = True
    notebook_utils.requires_ipython(foo)()
    self.assertFalse(self.foo_called)


if __name__ == '__main__':
  tf.test.main()
