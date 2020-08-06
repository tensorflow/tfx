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

from tfx.experimental.distributed_inference.graphdef_experiments.subgraph_partitioning import execution_spec


class ExecutionSpecTest(tf.test.TestCase):
  """A test for the dataclass ExecutionSpec."""

  def test_spec(self):
    """Verifies ExecutionSpec with an example."""
    subgraph = None
    input_names = {'a', 'b', 'c'}
    output_names = {'d'}
    is_remote_op = True

    spec = execution_spec.ExecutionSpec(subgraph, input_names, output_names,
                                        is_remote_op)

    self.assertEqual(spec.subgraph, subgraph)
    self.assertEqual(spec.input_names, input_names)
    self.assertEqual(spec.output_names, output_names)
    self.assertEqual(spec.is_remote_op, is_remote_op)


if __name__ == '__main__':
  tf.test.main()
