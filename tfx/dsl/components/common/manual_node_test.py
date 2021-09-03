# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Tests for tfx.dsl.components.common.manual_node."""
import tensorflow as tf
from tfx.dsl.components.common import manual_node


class ManualNodeTest(tf.test.TestCase):

  def testManualNodeConstruction(self):
    node = manual_node.ManualNode(description='test_description')
    self.assertDictEqual(
        node.exec_properties, {
            manual_node.DESCRIPTION_KEY: 'test_description',
        })
    self.assertEmpty(node.inputs)
    self.assertEmpty(node.outputs)

if __name__ == '__main__':
  tf.test.main()
