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
"""Tests for tfx.dsl.components.base.node_registry."""

import threading
from typing import Any, Dict, Text

import tensorflow as tf
from tfx.dsl.components.base import base_node
from tfx.dsl.components.base import node_registry


class _FakeNode(base_node.BaseNode):

  @property
  def inputs(self) -> Dict[Text, Any]:
    return {}

  @property
  def outputs(self) -> Dict[Text, Any]:
    return {}

  @property
  def exec_properties(self) -> Dict[str, Any]:
    return {}


class NodeRegistryTest(tf.test.TestCase):

  def testNodeRegistrySingleThread(self):
    thread_name = threading.current_thread().name
    self.assertSetEqual(node_registry.registered_nodes(), set())

    unused_node1 = _FakeNode().with_id(f'node1_{thread_name}')
    registered_node_names = {
        node.id for node in node_registry.registered_nodes()
    }
    self.assertSetEqual(registered_node_names, {f'node1_{thread_name}'})

    unused_node2 = _FakeNode().with_id(f'node2_{thread_name}')
    registered_node_names = {
        node.id for node in node_registry.registered_nodes()
    }
    self.assertSetEqual(registered_node_names,
                        {f'node1_{thread_name}', f'node2_{thread_name}'})

  def testNodeRegistryMultiThread(self):
    threads = [
        threading.Thread(target=self.testNodeRegistrySingleThread)
        for i in range(10)
    ]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()


if __name__ == '__main__':
  tf.test.main()
