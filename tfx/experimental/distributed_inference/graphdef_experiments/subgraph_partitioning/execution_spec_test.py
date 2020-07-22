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
"""Test for execution_spec."""

import tensorflow as tf

from execution_spec import ExecutionSpec


class ExecutionSpecTest(tf.test.TestCase):
  """Test the execution_spec dataclass."""
  def test_spec(self):
    """Verify using an example spec."""
    subgraph = None
    input_names = {'a', 'b', 'c'}
    output_names = {'d'}
    is_remote_op = True
    body_node_names = {'d'}
    nodes_from_other_layers = set([])

    spec = ExecutionSpec(subgraph,
                         input_names,
                         output_names,
                         is_remote_op,
                         body_node_names,
                         nodes_from_other_layers)

    self.assertEqual(spec.subgraph, subgraph)
    self.assertEqual(spec.input_names, input_names)
    self.assertEqual(spec.output_names, output_names)
    self.assertEqual(spec.is_remote_op, is_remote_op)
    self.assertEqual(spec.body_node_names, body_node_names)
    self.assertEqual(spec.nodes_from_other_layers, nodes_from_other_layers)


if __name__ == '__main__':
  tf.test.main()
