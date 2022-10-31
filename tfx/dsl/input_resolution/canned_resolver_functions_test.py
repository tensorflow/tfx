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

from typing import Dict, Sequence

import tensorflow as tf

from tfx import types
from tfx.dsl.compiler import compiler_context
from tfx.dsl.compiler import node_inputs_compiler
from tfx.dsl.components.base import base_node
from tfx.dsl.control_flow import for_each
from tfx.dsl.input_resolution import canned_resolver_functions
from tfx.dsl.input_resolution.ops import test_utils
from tfx.orchestration import pipeline
from tfx.orchestration.portable import inputs_utils
from tfx.proto.orchestration import pipeline_pb2
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


def _compile_inputs(
    inputs: Dict[str, channel_types.BaseChannel]) -> pipeline_pb2.PipelineNode:
  """Returns a compiled PipelineNode from the DummyNode inputs dictionary."""
  node = DummyNode('MyNode', inputs=inputs)
  p = pipeline.Pipeline(pipeline_name='pipeline', components=[node])
  ctx = compiler_context.PipelineContext(p)
  node_inputs = pipeline_pb2.NodeInputs()

  # Compile the NodeInputs and wrap in a PipelineNode.
  node_inputs_compiler.compile_node_inputs(ctx, node, node_inputs)
  return pipeline_pb2.PipelineNode(inputs=node_inputs)


class CannedResolverFunctionsTest(
    test_case_utils.TfxTest, test_case_utils.MlmdMixins):

  def setUp(self):
    super().setUp()
    self.init_mlmd()
    self.enter_context(self.mlmd_handle)

  def assertArtifactEqual(self,
                          resolved_artifact: metadata_store_pb2.Artifact,
                          mlmd_artifact: metadata_store_pb2.Artifact,
                          check_span_and_version: bool = False):
    """Checks that a MLMD artifacts and resolved artifact are equal."""
    self.assertEqual(mlmd_artifact.id, resolved_artifact.id)
    self.assertEqual(mlmd_artifact.type_id, resolved_artifact.type_id)
    self.assertEqual(mlmd_artifact.uri, resolved_artifact.uri)
    self.assertEqual(mlmd_artifact.state, resolved_artifact.state)

    if check_span_and_version:
      self.assertEqual(mlmd_artifact.properties['span'],
                       resolved_artifact.properties['span'])
      self.assertEqual(mlmd_artifact.properties['version'],
                       resolved_artifact.properties['version'])

  def _insert_artifacts_into_mlmd(
      self, spans: Sequence[int],
      versions: Sequence[int]) -> Sequence[metadata_store_pb2.Artifact]:
    """Inserts artifacts with the given spans and versions into MLMD."""
    mlmd_context = self.put_context('pipeline', 'pipeline')

    mlmd_artifacts = []
    for span, version in zip(spans, versions):
      mlmd_artifacts.append(
          self.put_artifact(
              artifact_type='DummyArtifact',
              properties={
                  'span': span,
                  'version': version
              }))

    for mlmd_artifact in mlmd_artifacts:
      self.put_execution(
          'ProducerNode',
          inputs={},
          outputs={'x': [mlmd_artifact]},
          contexts=[mlmd_context])

    return mlmd_artifacts

  def assertArtifactListEqual(self,
                              resolved_artifacts: metadata_store_pb2.Artifact,
                              mlmd_artifacts: metadata_store_pb2.Artifact,
                              check_span_and_version: bool = False):
    """Checks that a list of MLMD artifacts and resolved artifacts are equal."""
    self.assertEqual(len(mlmd_artifacts), len(resolved_artifacts))
    for mlmd_artifact, resolved_artifact in zip(mlmd_artifacts,
                                                resolved_artifacts):
      self.assertArtifactEqual(resolved_artifact, mlmd_artifact,
                               check_span_and_version)

  def testLatestCreatedResolverFn_E2E(self):
    channel = canned_resolver_functions.latest_created(
        types.Channel(test_utils.DummyArtifact, output_key='x'), n=2)
    pipeline_node = _compile_inputs({'x': channel})

    # Populate the MLMD database with DummyArtifacts to test the input
    # resolution end to end.
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
        pipeline_node=pipeline_node, metadata_handler=self.mlmd_handle)
    self.assertIsInstance(resolved, inputs_utils.Trigger)

    # Check that actual_artifacts = [mlmd_artifact_2, mlmd_artifact_3] because
    # those two artifacts are the latest artifacts and n=2.
    actual_artifacts = [r.mlmd_artifact for r in resolved[0]['x']]
    expected_artifacts = [mlmd_artifact_2, mlmd_artifact_3]
    self.assertArtifactListEqual(actual_artifacts, expected_artifacts)

  def testStaticRangeResolverFn_E2E(self):
    channel = canned_resolver_functions.static_range(
        types.Channel(test_utils.DummyArtifact, output_key='x'),
        end_span_number=5,
        keep_all_versions=True,
        exclude_span_numbers=[2])
    pipeline_node = _compile_inputs({'x': channel})

    spans = [0, 1, 2, 3, 3, 5, 7, 10]
    versions = [0, 0, 0, 0, 3, 0, 0, 0]
    mlmd_artifacts = self._insert_artifacts_into_mlmd(spans, versions)

    resolved = inputs_utils.resolve_input_artifacts(
        pipeline_node=pipeline_node, metadata_handler=self.mlmd_handle)
    self.assertIsInstance(resolved, inputs_utils.Trigger)

    # The resolved artifacts should have (span, version) tuples of:
    # [(0, 0), (1, 0), (3, 0), (3, 3), (5, 0)].
    actual_artifacts = [r.mlmd_artifact for r in resolved[0]['x']]
    expected_artifacts = [mlmd_artifacts[i] for i in [0, 1, 3, 4, 5]]
    self.assertArtifactListEqual(
        actual_artifacts, expected_artifacts, check_span_and_version=True)

  def testStaticRangeResolverFn_MinSpans_RaisesSkipSignal(self):
    channel = canned_resolver_functions.static_range(
        types.Channel(test_utils.DummyArtifact, output_key='x'),
        start_span_number=0,
        end_span_number=5)
    pipeline_node = _compile_inputs({'x': channel})

    spans = [0, 1, 2, 3, 3, 5, 7, 10]
    versions = [0, 0, 0, 0, 3, 0, 0, 0]
    self._insert_artifacts_into_mlmd(spans, versions)

    resolved = inputs_utils.resolve_input_artifacts(
        pipeline_node=pipeline_node, metadata_handler=self.mlmd_handle)
    self.assertIsInstance(resolved, inputs_utils.Skip)

  def testRollingRangeResolverFn_E2E(self):
    channel = canned_resolver_functions.rolling_range(
        types.Channel(test_utils.DummyArtifact, output_key='x'),
        start_span_number=3,
        num_spans=2,
        skip_num_recent_spans=1,
        keep_all_versions=True)
    pipeline_node = _compile_inputs({'x': channel})

    spans = [1, 2, 3, 3, 7, 8]
    versions = [0, 0, 1, 0, 1, 2]
    mlmd_artifacts = self._insert_artifacts_into_mlmd(spans, versions)

    resolved = inputs_utils.resolve_input_artifacts(
        pipeline_node=pipeline_node, metadata_handler=self.mlmd_handle)
    self.assertIsInstance(resolved, inputs_utils.Trigger)

    # The resolved artifacts should have (span, version) tuples of:
    # [(7, 1), (3, 0), (3, 1)].
    actual_artifacts = [r.mlmd_artifact for r in resolved[0]['x']]
    expected_artifacts = [mlmd_artifacts[i] for i in [4, 3, 2]]
    self.assertArtifactListEqual(
        actual_artifacts, expected_artifacts, check_span_and_version=True)

  def testAllSpansResolverFn_E2E(self):
    channel = canned_resolver_functions.all_spans(
        types.Channel(test_utils.DummyArtifact, output_key='x'),
        span_descending=False)
    pipeline_node = _compile_inputs({'x': channel})

    spans = [0, 1, 2, 3, 3, 5, 7, 10]
    versions = [0, 0, 0, 0, 3, 0, 0, 0]
    mlmd_artifacts = self._insert_artifacts_into_mlmd(spans, versions)

    resolved = inputs_utils.resolve_input_artifacts(
        pipeline_node=pipeline_node, metadata_handler=self.mlmd_handle)
    self.assertIsInstance(resolved, inputs_utils.Trigger)

    actual_artifacts = [r.mlmd_artifact for r in resolved[0]['x']]
    expected_artifacts = [mlmd_artifacts[i] for i in [0, 1, 2, 4, 5, 6, 7]]
    self.assertArtifactListEqual(
        actual_artifacts, expected_artifacts, check_span_and_version=True)

  def testLatestPipelineRunOutputsResolverFn(self):
    producer_pipeline = pipeline.Pipeline(
        outputs={'x': types.Channel(test_utils.DummyArtifact, output_key='x')},
        pipeline_name='producer-pipeline')
    return_value = canned_resolver_functions.latest_pipeline_run_outputs(
        pipeline=producer_pipeline)

    self.assertIsInstance(return_value['x'], types.BaseChannel)
    self.assertEqual('producer-pipeline',
                     return_value['x'].output_node.kwargs['pipeline_name'])

  def testRollingRangeResolverFn_MinSpans_RaisesSkipSignal(self):
    channel = canned_resolver_functions.rolling_range(
        types.Channel(test_utils.DummyArtifact, output_key='x'),
        start_span_number=3,
        num_spans=5,
        skip_num_recent_spans=1)
    pipeline_node = _compile_inputs({'x': channel})

    spans = [1, 2, 3, 3, 7, 8]
    versions = [0, 0, 1, 0, 1, 2]
    self._insert_artifacts_into_mlmd(spans, versions)

    resolved = inputs_utils.resolve_input_artifacts(
        pipeline_node=pipeline_node, metadata_handler=self.mlmd_handle)
    self.assertIsInstance(resolved, inputs_utils.Skip)

  def testSequentialRollingRangeResolverFn_E2E(self):
    xs = canned_resolver_functions.sequential_rolling_range(
        types.Channel(test_utils.DummyArtifact),
        start_span_number=1,
        num_spans=3,
        skip_num_recent_spans=1,
        keep_all_versions=False,
        exclude_span_numbers=[5])
    with for_each.ForEach(xs) as each_x:
      inputs = {'x': each_x}
    pipeline_node = _compile_inputs(inputs)

    spans = [1, 2, 3, 3, 4, 5, 7]
    versions = [0, 0, 1, 0, 0, 0]
    mlmd_artifacts = self._insert_artifacts_into_mlmd(spans, versions)

    resolved = inputs_utils.resolve_input_artifacts(
        pipeline_node=pipeline_node, metadata_handler=self.mlmd_handle)
    self.assertIsInstance(resolved, inputs_utils.Trigger)

    # The resolved artifacts should have (span, version) tuples of:
    # [(1, 0), (2, 0), (3, 1)], [(2, 0), (3, 1), (4,0)].
    expected_artifact_idxs = [[0, 1, 2], [1, 2, 4]]
    for i, artifacts in enumerate(resolved):
      actual_artifacts = [r.mlmd_artifact for r in artifacts['x']]
      expected_artifacts = [
          mlmd_artifacts[j] for j in expected_artifact_idxs[i]
      ]
      self.assertArtifactListEqual(
          actual_artifacts, expected_artifacts, check_span_and_version=True)

  def testSequentialRollingRangeResolverFn_E2E_SkipSignalRaised(self):
    # The artifacts will only have consecutive spans from [1, 5] but
    # num_spans=10 so no artifacts will be returnd by the resolver_fn and
    # a SkipSignal will be raised during input resolution.
    xs = canned_resolver_functions.sequential_rolling_range(
        types.Channel(test_utils.DummyArtifact),
        start_span_number=1,
        num_spans=10,
        skip_num_recent_spans=0,
        keep_all_versions=False)
    with for_each.ForEach(xs) as each_x:
      inputs = {'x': each_x}
    pipeline_node = _compile_inputs(inputs)

    spans = [1, 2, 3, 3, 4, 5]
    versions = [0, 0, 1, 0, 0, 0]
    _ = self._insert_artifacts_into_mlmd(spans, versions)

    resolved = inputs_utils.resolve_input_artifacts(
        pipeline_node=pipeline_node, metadata_handler=self.mlmd_handle)
    self.assertIsInstance(resolved, inputs_utils.Skip)

if __name__ == '__main__':
  tf.test.main()
