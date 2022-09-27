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
"""Tests for tfx.dsl.input_resolution.canned_resolver_functions."""

import tensorflow as tf

from tfx.dsl.compiler import compiler_context
from tfx.dsl.compiler import node_inputs_compiler
from tfx.dsl.components.base import base_node
from tfx.dsl.input_resolution import canned_resolver_functions
from tfx.dsl.input_resolution.ops import test_utils
from tfx.orchestration import pipeline
from tfx.orchestration.portable import inputs_utils
from tfx.proto.orchestration import pipeline_pb2
import tfx.types
from tfx.types import channel as channel_types
from tfx.utils import test_case_utils

from ml_metadata.proto import metadata_store_pb2


class DummyNode(base_node.BaseNode):

  def __init__(self, id: str, inputs=None, exec_properties=None):  # pylint: disable=redefined-builtin
    super().__init__()
    self.with_id(id)
    self._inputs = inputs or {}
    self._exec_properties = exec_properties or {}
    self._outputs = {}

  def output(self, key: str, artifact_type=test_utils.DummyArtifact):
    if key not in self._outputs:
      self._outputs[key] = channel_types.OutputChannel(artifact_type, self, key)
    return self._outputs[key]

  @property
  def inputs(self) ->...:
    return self._inputs

  @property
  def exec_properties(self) ->...:
    return self._exec_properties

  @property
  def outputs(self) ->...:
    return self._outputs


class CannedResolverFunctionsTest(tf.test.TestCase, test_case_utils.MlmdMixins):

  def assertArtifactEqual(self, mlmd_artifact: metadata_store_pb2.Artifact,
                          resolved_artifact: metadata_store_pb2.Artifact):
    """Checks that a mlmd Artifact and resolved artifact are equivalent."""
    self.assertEqual(mlmd_artifact.id, resolved_artifact.id)
    self.assertEqual(mlmd_artifact.type_id, resolved_artifact.type_id)
    self.assertEqual(mlmd_artifact.uri, resolved_artifact.uri)
    self.assertEqual(mlmd_artifact.state, resolved_artifact.state)

  def testLatestResolverFn_E2E(self):
    resolved_channel = canned_resolver_functions.latest_created(
        tfx.types.Channel(test_utils.DummyArtifact, output_key='x'), n=2)
    node = DummyNode('MyNode', inputs={'x': resolved_channel})

    p = pipeline.Pipeline(pipeline_name='pipeline', components=[node])
    ctx = compiler_context.PipelineContext(p)
    node_inputs = pipeline_pb2.NodeInputs()

    # Compile the node's inputs.
    node_inputs_compiler.compile_node_inputs(ctx, node, node_inputs)

    # Populate the MLMD database with DummyArtifacts to test the input
    # resolution end to end.
    pipeline_node = pipeline_pb2.PipelineNode(inputs=node_inputs)
    self.init_mlmd()
    self.enter_context(self.mlmd_handler)
    mlmd_context = self.put_context('pipeline', 'pipeline')
    mlmd_artifact_1 = self.put_artifact('DummyArtifact')
    mlmd_artifact_2 = self.put_artifact('DummyArtifact')
    mlmd_artifact_3 = self.put_artifact('DummyArtifact')

    for mlmd_artifact in [mlmd_artifact_1, mlmd_artifact_2, mlmd_artifact_3]:
      self.put_execution(
          'ProducerNode',
          inputs={},
          outputs={'x': [mlmd_artifact]},
          contexts=[mlmd_context])

    resolved = inputs_utils.resolve_input_artifacts(
        pipeline_node=pipeline_node, metadata_handler=self.mlmd_handler)

    actual_artifacts = [r.mlmd_artifact for r in resolved[0]['x']]
    self.assertIsInstance(resolved, inputs_utils.Trigger)
    self.assertLen(actual_artifacts, 2)

    # Check that actual_artifacts = [mlmd_artifact_2, mlmd_artifact_3] because
    # those two artifacts are the latest artifacts and n=2.
    self.assertArtifactEqual(actual_artifacts[0], mlmd_artifact_2)
    self.assertArtifactEqual(actual_artifacts[1], mlmd_artifact_3)


if __name__ == '__main__':
  tf.test.main()
