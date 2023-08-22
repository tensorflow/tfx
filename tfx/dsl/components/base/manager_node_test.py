# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Tests for manager_node creation."""

import tensorflow as tf
from tfx import v1 as tfx
from tfx.dsl.components.base import manager_node
from tfx.proto.orchestration import pipeline_pb2


@tfx.dsl.components.component
def MyCustomComponent():
  pass


class ManagerNodeTest(tf.test.TestCase):

  def testCreateManagerNodeFrom(self):
    component = MyCustomComponent()
    manager = manager_node.create_manager_node_from(component)
    self.assertTrue(manager.is_manager_node)
    self.assertEqual(
        manager.outputs[manager_node.MANAGER_ARTIFACT_OUTPUT_CHANNEL].type,
        tfx.types.standard_artifacts.ManagerArtifact,
    )
    self.assertTrue(manager.node_execution_options.lazily_execute)

    # Check that base component is not modified.
    self.assertFalse(
        hasattr(component.spec, manager_node.MANAGER_ARTIFACT_OUTPUT_CHANNEL)
    )
    self.assertNotEqual(manager.executor_spec, component.executor_spec)

  def testIsManagerNode(self):
    node = pipeline_pb2.PipelineNode()
    self.assertFalse(manager_node.is_manager_node(node))
    node.execution_options.manager_node_execution_options.SetInParent()
    self.assertTrue(manager_node.is_manager_node(node))


if __name__ == "__main__":
  tf.test.main()
