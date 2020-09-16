# Lint as: python2, python3
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
"""Tests for tfx.utils.topsort."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attr
import tensorflow as tf
from tfx.utils import topsort


@attr.s
class Node(object):
  name = attr.ib()
  upstream_nodes = attr.ib()
  downstream_nodes = attr.ib()


class TopsortTest(tf.test.TestCase):

  def test_topsorted_layers_DAG(self):
    nodes = [
        Node('A', [], ['B', 'C', 'D']),
        Node('B', ['A'], []),
        Node('C', ['A'], ['D']),
        Node('D', ['A', 'C', 'F'], ['E']),
        Node('E', ['D'], []),
        Node('F', [], ['D'])
    ]
    node_map = {node.name: node for node in nodes}
    layers = topsort.topsorted_layers(
        nodes,
        get_node_id_fn=lambda n: n.name,
        get_parent_nodes=(
            lambda n: [node_map[name] for name in n.upstream_nodes]),
        get_child_nodes=(
            lambda n: [node_map[name] for name in n.downstream_nodes]))
    self.assertEqual([['A', 'F'], ['B', 'C'], ['D'], ['E']],
                     [[node.name for node in layer] for layer in layers])

  def test_topsorted_layers_error_if_cycle(self):
    nodes = [
        Node('A', [], ['B', 'E']),
        Node('B', ['A', 'D'], ['C']),
        Node('C', ['B'], ['D']),
        Node('D', ['C'], ['B']),
        Node('E', ['A'], [])
    ]
    node_map = {node.name: node for node in nodes}
    with self.assertRaisesRegexp(topsort.InvalidDAGError, 'Cycle detected.'):
      topsort.topsorted_layers(
          nodes,
          get_node_id_fn=lambda n: n.name,
          get_parent_nodes=(
              lambda n: [node_map[name] for name in n.upstream_nodes]),
          get_child_nodes=(
              lambda n: [node_map[name] for name in n.downstream_nodes]))

  def test_topsorted_layers_empty(self):
    layers = topsort.topsorted_layers(
        nodes=[],
        get_node_id_fn=lambda n: n.name,
        get_parent_nodes=lambda n: [],
        get_child_nodes=lambda n: [])
    self.assertEqual([], layers)


if __name__ == '__main__':
  tf.test.main()
