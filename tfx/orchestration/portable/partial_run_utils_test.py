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
"""Tests for tfx.orchestration.portable.partial_run_utils."""

from typing import Mapping

from absl.testing import absltest
from absl.testing import parameterized

from tfx.orchestration.portable import partial_run_utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import test_case_utils

from ml_metadata.proto import metadata_store_pb2


def _to_context_spec(type_name: str, name: str) -> pipeline_pb2.ContextSpec:
  return pipeline_pb2.ContextSpec(
      type=metadata_store_pb2.ContextType(name=type_name),
      name=pipeline_pb2.Value(
          field_value=metadata_store_pb2.Value(string_value=name)))


def _to_output_spec(artifact_name: str) -> pipeline_pb2.OutputSpec:
  return pipeline_pb2.OutputSpec(
      artifact_spec=pipeline_pb2.OutputSpec.ArtifactSpec(
          type=metadata_store_pb2.ArtifactType(name=artifact_name)))


def _to_input_channel(
    producer_output_key: str, producer_node_id: str, artifact_type: str,
    context_names: Mapping[str, str]) -> pipeline_pb2.InputSpec.Channel:
  # pylint: disable=g-complex-comprehension
  context_queries = [
      pipeline_pb2.InputSpec.Channel.ContextQuery(
          type=metadata_store_pb2.ContextType(name=context_type),
          name=pipeline_pb2.Value(
              field_value=metadata_store_pb2.Value(string_value=context_name)))
      for context_type, context_name in context_names.items()
  ]
  return pipeline_pb2.InputSpec.Channel(
      output_key=producer_output_key,
      producer_node_query=pipeline_pb2.InputSpec.Channel.ProducerNodeQuery(
          id=producer_node_id),
      context_queries=context_queries,
      artifact_query=pipeline_pb2.InputSpec.Channel.ArtifactQuery(
          type=metadata_store_pb2.ArtifactType(name=artifact_type)))


class PipelineFilteringTest(parameterized.TestCase, test_case_utils.TfxTest):

  def testAsyncPipeline_error(self):
    """If Pipeline has execution_mode ASYNC, raise ValueError."""
    input_pipeline = pipeline_pb2.Pipeline()
    input_pipeline.pipeline_info.id = 'my_pipeline'
    input_pipeline.execution_mode = pipeline_pb2.Pipeline.ASYNC

    with self.assertRaisesRegex(
        ValueError, 'Pipeline filtering is only supported for SYNC pipelines.'):
      _ = partial_run_utils.filter_pipeline(input_pipeline)

  def testSubpipeline_error(self):
    """If Pipeline contains sub-pipeline, raise ValueError."""
    input_pipeline = pipeline_pb2.Pipeline()
    input_pipeline.pipeline_info.id = 'my_pipeline'
    input_pipeline.execution_mode = pipeline_pb2.Pipeline.SYNC
    sub_pipeline_node = input_pipeline.nodes.add()
    sub_pipeline_node.sub_pipeline.pipeline_info.id = 'my_subpipeline'
    node_a = sub_pipeline_node.sub_pipeline.nodes.add()
    node_a.pipeline_node.node_info.id = 'a'

    with self.assertRaisesRegex(
        ValueError,
        'Pipeline filtering not supported for pipelines with sub-pipelines.'):
      _ = partial_run_utils.filter_pipeline(input_pipeline)

  def testNotTopologicallySorted_upstream_error(self):
    """If Pipeline is not topologically sorted, raise ValueError."""
    input_pipeline = pipeline_pb2.Pipeline()
    input_pipeline.pipeline_info.id = 'my_pipeline'
    input_pipeline.execution_mode = pipeline_pb2.Pipeline.SYNC
    node_b = input_pipeline.nodes.add()
    node_b.pipeline_node.node_info.id = 'b'
    node_b.pipeline_node.upstream_nodes.extend(['a'])
    node_a = input_pipeline.nodes.add()
    node_a.pipeline_node.node_info.id = 'a'
    node_a.pipeline_node.downstream_nodes.extend(['b'])

    with self.assertRaisesRegex(ValueError,
                                'Input pipeline is not topologically sorted.'):
      _ = partial_run_utils.filter_pipeline(input_pipeline)

  def testNotTopologicallySorted_downstream_error(self):
    """If Pipeline is not topologically sorted, raise ValueError."""
    input_pipeline = pipeline_pb2.Pipeline()
    input_pipeline.pipeline_info.id = 'my_pipeline'
    input_pipeline.execution_mode = pipeline_pb2.Pipeline.SYNC
    node_b = input_pipeline.nodes.add()
    node_b.pipeline_node.node_info.id = 'b'
    node_a = input_pipeline.nodes.add()
    node_a.pipeline_node.node_info.id = 'a'
    node_a.pipeline_node.downstream_nodes.extend(['b'])

    with self.assertRaisesRegex(ValueError,
                                'Input pipeline is not topologically sorted.'):
      _ = partial_run_utils.filter_pipeline(input_pipeline)

  def testNoFilter(self):
    """Basic case where there are no filters applied.

    input_pipeline: node_a -> node_b -> node_c
    from_node: all nodes
    to_node: all nodes
    expected output_pipeline: node_a -> node_b -> node_c
    """
    input_pipeline = pipeline_pb2.Pipeline()
    input_pipeline.pipeline_info.id = 'my_pipeline'
    input_pipeline.execution_mode = pipeline_pb2.Pipeline.SYNC
    node_a = input_pipeline.nodes.add()
    node_a.pipeline_node.node_info.id = 'a'
    node_a.pipeline_node.node_info.type.name = 'A'
    node_a.pipeline_node.contexts.contexts.extend([
        _to_context_spec('pipeline', 'my_pipeline'),
        _to_context_spec('component', 'a'),
    ])
    node_a.pipeline_node.outputs.outputs['out'].CopyFrom(_to_output_spec('AB'))
    node_a.pipeline_node.downstream_nodes.extend(['b'])
    node_b = input_pipeline.nodes.add()
    node_b.pipeline_node.node_info.id = 'b'
    node_b.pipeline_node.node_info.type.name = 'B'
    node_b.pipeline_node.contexts.contexts.extend([
        _to_context_spec('pipeline', 'my_pipeline'),
        _to_context_spec('component', 'b'),
    ])
    node_b.pipeline_node.inputs.inputs['in'].channels.extend([
        _to_input_channel(
            producer_node_id='a',
            producer_output_key='out',
            artifact_type='AB',
            context_names={
                'pipeline': 'my_pipeline',
                'component': 'a',
            }),
    ])
    node_b.pipeline_node.inputs.inputs['in'].min_count = 1
    node_b.pipeline_node.outputs.outputs['out'].CopyFrom(_to_output_spec('BC'))
    node_b.pipeline_node.upstream_nodes.extend(['a'])
    node_b.pipeline_node.downstream_nodes.extend(['c'])
    node_c = input_pipeline.nodes.add()
    node_c.pipeline_node.node_info.id = 'c'
    node_c.pipeline_node.node_info.type.name = 'C'
    node_c.pipeline_node.contexts.contexts.extend([
        _to_context_spec('pipeline', 'my_pipeline'),
        _to_context_spec('component', 'c')
    ])
    node_c.pipeline_node.inputs.inputs['in'].channels.extend([
        _to_input_channel(
            producer_node_id='b',
            producer_output_key='out',
            artifact_type='BC',
            context_names={
                'pipeline': 'my_pipeline',
                'component': 'b',
            }),
    ])
    node_c.pipeline_node.inputs.inputs['in'].min_count = 1
    node_c.pipeline_node.upstream_nodes.extend(['b'])

    expected_pipeline = pipeline_pb2.Pipeline()
    expected_pipeline.CopyFrom(input_pipeline)  # no changes expected
    expected_excluded = {}

    filtered_pipeline, excluded = partial_run_utils.filter_pipeline(
        input_pipeline)

    self.assertProtoEquals(expected_pipeline, filtered_pipeline)
    self.assertEqual(expected_excluded, excluded)

  def testFilterOutNothing(self):
    """Basic case where no nodes are filtered out.

    input_pipeline: node_a -> node_b -> node_c
    from_node: node_a
    to_node: node_c
    expected output_pipeline: node_a -> node_b -> node_c
    expected input_channel_ref: (empty)
    """
    input_pipeline = pipeline_pb2.Pipeline()
    input_pipeline.pipeline_info.id = 'my_pipeline'
    input_pipeline.execution_mode = pipeline_pb2.Pipeline.SYNC
    node_a = input_pipeline.nodes.add()
    node_a.pipeline_node.node_info.id = 'a'
    node_a.pipeline_node.node_info.type.name = 'A'
    node_a.pipeline_node.contexts.contexts.extend([
        _to_context_spec('pipeline', 'my_pipeline'),
        _to_context_spec('component', 'a'),
    ])
    node_a.pipeline_node.outputs.outputs['out'].CopyFrom(_to_output_spec('AB'))
    node_a.pipeline_node.downstream_nodes.extend(['b'])
    node_b = input_pipeline.nodes.add()
    node_b.pipeline_node.node_info.id = 'b'
    node_b.pipeline_node.node_info.type.name = 'B'
    node_b.pipeline_node.contexts.contexts.extend([
        _to_context_spec('pipeline', 'my_pipeline'),
        _to_context_spec('component', 'b'),
    ])
    node_b.pipeline_node.inputs.inputs['in'].channels.extend([
        _to_input_channel(
            producer_node_id='a',
            producer_output_key='out',
            artifact_type='AB',
            context_names={
                'pipeline': 'my_pipeline',
                'component': 'a',
            }),
    ])
    node_b.pipeline_node.inputs.inputs['in'].min_count = 1
    node_b.pipeline_node.outputs.outputs['out'].CopyFrom(_to_output_spec('BC'))
    node_b.pipeline_node.upstream_nodes.extend(['a'])
    node_b.pipeline_node.downstream_nodes.extend(['c'])
    node_c = input_pipeline.nodes.add()
    node_c.pipeline_node.node_info.id = 'c'
    node_c.pipeline_node.node_info.type.name = 'C'
    node_c.pipeline_node.contexts.contexts.extend([
        _to_context_spec('pipeline', 'my_pipeline'),
        _to_context_spec('component', 'c'),
    ])
    node_c.pipeline_node.inputs.inputs['in'].channels.extend([
        _to_input_channel(
            producer_node_id='b',
            producer_output_key='out',
            artifact_type='BC',
            context_names={
                'pipeline': 'my_pipeline',
                'component': 'b',
            }),
    ])
    node_c.pipeline_node.inputs.inputs['in'].min_count = 1
    node_c.pipeline_node.upstream_nodes.extend(['b'])

    expected_pipeline = pipeline_pb2.Pipeline()
    expected_pipeline.CopyFrom(input_pipeline)  # no changes expected
    expected_excluded = {}

    filtered_pipeline, excluded = partial_run_utils.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'a'),
        to_nodes=lambda node_id: (node_id == 'c'))

    self.assertProtoEquals(expected_pipeline, filtered_pipeline)
    self.assertEqual(expected_excluded, excluded)

  def testFilterOutSinkNode(self):
    """Filter out a node that has upstream nodes but no downstream nodes.

    input_pipeline: node_a -> node_b -> node_c
    to_node: node_b
    expected_output_pipeline: node_a -> node_b
    expected input_channel_ref: (empty)
    """
    input_pipeline = pipeline_pb2.Pipeline()
    input_pipeline.pipeline_info.id = 'my_pipeline'
    input_pipeline.execution_mode = pipeline_pb2.Pipeline.SYNC
    node_a = input_pipeline.nodes.add()
    node_a.pipeline_node.node_info.id = 'a'
    node_a.pipeline_node.node_info.type.name = 'A'
    node_a.pipeline_node.contexts.contexts.extend([
        _to_context_spec('pipeline', 'my_pipeline'),
        _to_context_spec('component', 'a'),
    ])
    node_a.pipeline_node.outputs.outputs['out'].CopyFrom(_to_output_spec('AB'))
    node_a.pipeline_node.downstream_nodes.extend(['b'])
    node_b = input_pipeline.nodes.add()
    node_b.pipeline_node.node_info.id = 'b'
    node_b.pipeline_node.node_info.type.name = 'B'
    node_b.pipeline_node.contexts.contexts.extend([
        _to_context_spec('pipeline', 'my_pipeline'),
        _to_context_spec('component', 'b'),
    ])
    node_b.pipeline_node.inputs.inputs['in'].channels.extend([
        _to_input_channel(
            producer_node_id='a',
            producer_output_key='out',
            artifact_type='AB',
            context_names={
                'pipeline': 'my_pipeline',
                'component': 'a',
            }),
    ])
    node_b.pipeline_node.inputs.inputs['in'].min_count = 1
    node_b.pipeline_node.outputs.outputs['out'].CopyFrom(_to_output_spec('BC'))
    node_b.pipeline_node.upstream_nodes.extend(['a'])
    node_b.pipeline_node.downstream_nodes.extend(['c'])
    node_c = input_pipeline.nodes.add()
    node_c.pipeline_node.node_info.id = 'c'
    node_c.pipeline_node.node_info.type.name = 'C'
    node_c.pipeline_node.contexts.contexts.extend([
        _to_context_spec('pipeline', 'my_pipeline'),
        _to_context_spec('component', 'c')
    ])
    node_c.pipeline_node.inputs.inputs['in'].channels.extend([
        _to_input_channel(
            producer_node_id='b',
            producer_output_key='out',
            artifact_type='BC',
            context_names={
                'pipeline': 'my_pipeline',
                'component': 'b',
            }),
    ])
    node_c.pipeline_node.inputs.inputs['in'].min_count = 1
    node_c.pipeline_node.upstream_nodes.extend(['b'])

    expected_pipeline = pipeline_pb2.Pipeline()
    expected_pipeline.pipeline_info.id = 'my_pipeline'
    expected_pipeline.execution_mode = pipeline_pb2.Pipeline.SYNC
    expected_pipeline.nodes.append(node_a)
    expected_pipeline.nodes.append(node_b)
    del expected_pipeline.nodes[-1].pipeline_node.downstream_nodes[:]
    expected_excluded = {}

    filtered_pipeline, excluded = partial_run_utils.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'a'),
        to_nodes=lambda node_id: (node_id == 'b'))

    self.assertProtoEquals(expected_pipeline, filtered_pipeline)
    self.assertEqual(expected_excluded, excluded)

  def testFilterOutSourceNode(self):
    """Filter out a node that has no upstream nodes but has downstream nodes.

    input_pipeline: node_a -> node_b -> node_c
    from_node: node_b
    to_node: node_c
    expected_output_pipeline: node_b -> node_c
    expected input_channel_ref: {node_a: AB}
    """
    input_pipeline = pipeline_pb2.Pipeline()
    input_pipeline.pipeline_info.id = 'my_pipeline'
    input_pipeline.execution_mode = pipeline_pb2.Pipeline.SYNC
    node_a = input_pipeline.nodes.add()
    node_a.pipeline_node.node_info.id = 'a'
    node_a.pipeline_node.node_info.type.name = 'A'
    node_a.pipeline_node.contexts.contexts.extend([
        _to_context_spec('pipeline', 'my_pipeline'),
        _to_context_spec('component', 'a'),
    ])
    node_a.pipeline_node.outputs.outputs['out'].CopyFrom(_to_output_spec('AB'))
    node_a.pipeline_node.downstream_nodes.extend(['b'])
    node_b = input_pipeline.nodes.add()
    node_b.pipeline_node.node_info.id = 'b'
    node_b.pipeline_node.node_info.type.name = 'B'
    node_b.pipeline_node.contexts.contexts.extend([
        _to_context_spec('pipeline', 'my_pipeline'),
        _to_context_spec('component', 'b'),
    ])
    node_b.pipeline_node.inputs.inputs['in'].channels.extend([
        _to_input_channel(
            producer_node_id='a',
            producer_output_key='out',
            artifact_type='AB',
            context_names={
                'pipeline': 'my_pipeline',
                'component': 'a',
            }),
    ])
    node_b.pipeline_node.inputs.inputs['in'].min_count = 1
    node_b.pipeline_node.outputs.outputs['out'].CopyFrom(_to_output_spec('BC'))
    node_b.pipeline_node.upstream_nodes.extend(['a'])
    node_b.pipeline_node.downstream_nodes.extend(['c'])
    node_c = input_pipeline.nodes.add()
    node_c.pipeline_node.node_info.id = 'c'
    node_c.pipeline_node.node_info.type.name = 'C'
    node_c.pipeline_node.contexts.contexts.extend([
        _to_context_spec('pipeline', 'my_pipeline'),
        _to_context_spec('component', 'c'),
    ])
    node_c.pipeline_node.inputs.inputs['in'].channels.extend([
        _to_input_channel(
            producer_node_id='b',
            producer_output_key='out',
            artifact_type='BC',
            context_names={
                'pipeline': 'my_pipeline',
                'component': 'b',
            }),
    ])
    node_c.pipeline_node.inputs.inputs['in'].min_count = 1
    node_c.pipeline_node.upstream_nodes.extend(['b'])

    expected_pipeline = pipeline_pb2.Pipeline()
    expected_pipeline.pipeline_info.id = 'my_pipeline'
    expected_pipeline.execution_mode = pipeline_pb2.Pipeline.SYNC
    expected_pipeline.nodes.append(node_b)
    del expected_pipeline.nodes[-1].pipeline_node.upstream_nodes[:]
    expected_pipeline.nodes.append(node_c)
    expected_excluded = {
        'a': node_b.pipeline_node.inputs.inputs['in'].channels[:],
    }

    filtered_pipeline, excluded = partial_run_utils.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'b'),
        to_nodes=lambda node_id: (node_id == 'c'))

    self.assertProtoEquals(expected_pipeline, filtered_pipeline)
    self.assertEqual(expected_excluded, excluded)

  def testFilterOutSourceNode_triangle(self):
    """Filter out a source node in a triangle.

    input_pipeline:
        node_a -> node_b -> node_c
             |--------------^
    from_node: node_b
    to_node: node_c
    expected_output_pipeline: node_b -> node_c
    expected input_channel_ref: {node_a: [AB, AC]}
    """
    input_pipeline = pipeline_pb2.Pipeline()
    input_pipeline.pipeline_info.id = 'my_pipeline'
    input_pipeline.execution_mode = pipeline_pb2.Pipeline.SYNC
    node_a = input_pipeline.nodes.add()
    node_a.pipeline_node.node_info.id = 'a'
    node_a.pipeline_node.node_info.type.name = 'A'
    node_a.pipeline_node.contexts.contexts.extend([
        _to_context_spec('pipeline', 'my_pipeline'),
        _to_context_spec('component', 'a'),
    ])
    node_a.pipeline_node.outputs.outputs['out_b'].CopyFrom(
        _to_output_spec('AB'))
    node_a.pipeline_node.outputs.outputs['out_c'].CopyFrom(
        _to_output_spec('AC'))
    node_a.pipeline_node.downstream_nodes.extend(['b', 'c'])
    node_b = input_pipeline.nodes.add()
    node_b.pipeline_node.node_info.id = 'b'
    node_b.pipeline_node.node_info.type.name = 'B'
    node_b.pipeline_node.contexts.contexts.extend([
        _to_context_spec('pipeline', 'my_pipeline'),
        _to_context_spec('component', 'b'),
    ])
    node_b.pipeline_node.inputs.inputs['in'].channels.extend([
        _to_input_channel(
            producer_node_id='a',
            producer_output_key='out_b',
            artifact_type='AB',
            context_names={
                'pipeline': 'my_pipeline',
                'component': 'a',
            }),
    ])
    node_b.pipeline_node.inputs.inputs['in'].min_count = 1
    node_b.pipeline_node.outputs.outputs['out'].CopyFrom(_to_output_spec('BC'))
    node_b.pipeline_node.upstream_nodes.extend(['a'])
    node_b.pipeline_node.downstream_nodes.extend(['c'])
    node_c = input_pipeline.nodes.add()
    node_c.pipeline_node.node_info.id = 'c'
    node_c.pipeline_node.node_info.type.name = 'C'
    node_c.pipeline_node.contexts.contexts.extend([
        _to_context_spec('pipeline', 'my_pipeline'),
        _to_context_spec('component', 'c'),
    ])
    node_c.pipeline_node.inputs.inputs['in_a'].channels.extend([
        _to_input_channel(
            producer_node_id='a',
            producer_output_key='out_c',
            artifact_type='AC',
            context_names={
                'pipeline': 'my_pipeline',
                'component': 'a',
            }),
    ])
    node_c.pipeline_node.inputs.inputs['in_a'].min_count = 1
    node_c.pipeline_node.inputs.inputs['in_b'].channels.extend([
        _to_input_channel(
            producer_node_id='b',
            producer_output_key='out',
            artifact_type='BC',
            context_names={
                'pipeline': 'my_pipeline',
                'component': 'b',
            }),
    ])
    node_c.pipeline_node.inputs.inputs['in_b'].min_count = 1
    node_c.pipeline_node.upstream_nodes.extend(['a', 'b'])

    expected_pipeline = pipeline_pb2.Pipeline()
    expected_pipeline.pipeline_info.id = 'my_pipeline'
    expected_pipeline.execution_mode = pipeline_pb2.Pipeline.SYNC
    expected_pipeline.nodes.append(node_b)
    del expected_pipeline.nodes[-1].pipeline_node.upstream_nodes[:]
    expected_pipeline.nodes.append(node_c)
    expected_pipeline.nodes[-1].pipeline_node.upstream_nodes[:] = ['b']
    expected_excluded = {
        'a': (node_b.pipeline_node.inputs.inputs['in'].channels[:] +
              node_c.pipeline_node.inputs.inputs['in_a'].channels[:]),
    }

    filtered_pipeline, excluded = partial_run_utils.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'b'),
        to_nodes=lambda node_id: (node_id == 'c'),
    )

    self.assertProtoEquals(expected_pipeline, filtered_pipeline)
    self.assertEqual(expected_excluded, excluded)


if __name__ == '__main__':
  absltest.main()
