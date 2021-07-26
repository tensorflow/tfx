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
from unittest import mock

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from tfx.dsl.compiler import compiler
from tfx.dsl.compiler import compiler_utils
from tfx.dsl.compiler import constants
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import component
from tfx.orchestration import metadata
from tfx.orchestration import pipeline as pipeline_lib
from tfx.orchestration.beam import beam_dag_runner
from tfx.orchestration.portable import execution_publish_utils
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable import partial_run_utils
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from ml_metadata.proto import metadata_store_pb2

_PIPELINE_RUN_CONTEXT_KEY = constants.PIPELINE_RUN_CONTEXT_TYPE_NAME


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
    expected_excluded_direct_deps = {}

    filtered_pipeline, excluded_direct_deps = partial_run_utils.filter_pipeline(
        input_pipeline)

    self.assertProtoEquals(expected_pipeline, filtered_pipeline)
    self.assertEqual(expected_excluded_direct_deps, excluded_direct_deps)

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
    expected_excluded_direct_deps = {}

    filtered_pipeline, excluded_direct_deps = partial_run_utils.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'a'),
        to_nodes=lambda node_id: (node_id == 'c'))

    self.assertProtoEquals(expected_pipeline, filtered_pipeline)
    self.assertEqual(expected_excluded_direct_deps, excluded_direct_deps)

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
    expected_excluded_direct_deps = {}

    filtered_pipeline, excluded_direct_deps = partial_run_utils.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'a'),
        to_nodes=lambda node_id: (node_id == 'b'))

    self.assertProtoEquals(expected_pipeline, filtered_pipeline)
    self.assertEqual(expected_excluded_direct_deps, excluded_direct_deps)

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
    expected_excluded_direct_deps = {
        'a': node_b.pipeline_node.inputs.inputs['in'].channels[:],
    }

    filtered_pipeline, excluded_direct_deps = partial_run_utils.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'b'),
        to_nodes=lambda node_id: (node_id == 'c'))

    self.assertProtoEquals(expected_pipeline, filtered_pipeline)
    self.assertEqual(expected_excluded_direct_deps, excluded_direct_deps)

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
    expected_excluded_direct_deps = {
        'a': (node_b.pipeline_node.inputs.inputs['in'].channels[:] +
              node_c.pipeline_node.inputs.inputs['in_a'].channels[:]),
    }

    filtered_pipeline, excluded_direct_deps = partial_run_utils.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'b'),
        to_nodes=lambda node_id: (node_id == 'c'),
    )

    self.assertProtoEquals(expected_pipeline, filtered_pipeline)
    self.assertEqual(expected_excluded_direct_deps, excluded_direct_deps)


# pylint: disable=invalid-name
# pytype: disable=wrong-arg-types
@component
def Load(start_num: Parameter[int],
         num: OutputArtifact[standard_artifacts.Integer]):
  num.value = start_num
  logging.info('Load with start_num=%s', start_num)


@component
def LoadFail(start_num: Parameter[int],
             num: OutputArtifact[standard_artifacts.Integer]):
  num.value = start_num
  logging.info('LoadFail about to raise an exception.')
  raise Exception('LoadFail fails, as expected.')


@component
def AddNum(to_add: Parameter[int],
           num: InputArtifact[standard_artifacts.Integer],
           added_num: OutputArtifact[standard_artifacts.Integer]):
  num_value = num.value
  added_num.value = num_value + to_add
  logging.info('AddNum with to_add=%s, num=%s', to_add, num_value)


@component
def Result(result: InputArtifact[standard_artifacts.Integer]):
  result_value = result.value
  logging.info('Result with result=%s', result_value)


# pytype: enable=wrong-arg-types
# pylint: enable=invalid-name


def _node_inputs_by_id(pipeline: pipeline_pb2.Pipeline,
                       node_id: str) -> pipeline_pb2.NodeInputs:
  """Doesn't make a copy."""

  node_ids_seen = []
  for node in pipeline.nodes:
    if node.pipeline_node.node_info.id == node_id:
      return node.pipeline_node.inputs
    node_ids_seen.append(node.pipeline_node.node_info.id)

  raise ValueError(
      f'node_id {node_id} not found in pipeline. Seen: {node_ids_seen}')


class PartialRunTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.pipeline_name = 'test_pipeline'
    pipeline_root_dir = self.create_tempdir(name=self.pipeline_name)
    self.pipeline_root = pipeline_root_dir.full_path
    self.metadata_config = metadata.sqlite_metadata_connection_config(
        pipeline_root_dir.create_file('mlmd.sqlite').full_path)
    self.result_node_id = Result.__name__

  def make_pipeline(self, components, run_id: str) -> pipeline_pb2.Pipeline:
    pipeline = pipeline_lib.Pipeline(self.pipeline_name, self.pipeline_root,
                                     self.metadata_config, components)
    result = compiler.Compiler().compile(pipeline)
    runtime_parameter_utils.substitute_runtime_parameter(
        result, {constants.PIPELINE_RUN_ID_PARAMETER_NAME: run_id})
    return result

  def assertResultEquals(self, pipeline_pb: pipeline_pb2.Pipeline,
                         expected_result: int):
    result_node_inputs = _node_inputs_by_id(
        pipeline_pb, node_id=self.result_node_id)
    with metadata.Metadata(self.metadata_config) as m:
      result_artifact = inputs_utils.resolve_input_artifacts(
          m, result_node_inputs)['result'][0]
    result_artifact.read()
    self.assertEqual(result_artifact.value, expected_result)

  def testRemoveFirstNode(self):
    """Test that partial run with the first node removed works."""
    ############################################################################
    # PART 1a: Full run -- Author pipeline.
    #
    # `Load` takes an exec_param at init time and outputs it at runtime.
    # `AddNum` takes an exec_param at init time and adds it to its input.
    # `Result` takes an input and does nothing.
    #
    #             (1)             (1)
    #              v               v
    #           -------        ---------        ---------
    #  run_1:  | Load | ----> | AddNum | ----> | Result |
    #          -------        ---------        ---------
    #
    ############################################################################
    load = Load(start_num=1)  # pylint: disable=no-value-for-parameter
    add_num = AddNum(to_add=1, num=load.outputs['num'])  # pylint: disable=no-value-for-parameter
    result = Result(result=add_num.outputs['added_num'])
    pipeline_pb_run_1 = self.make_pipeline(
        components=[load, add_num, result], run_id='run_1')
    ############################################################################
    # PART 1b: Full run -- Run pipeline.
    #
    #             (1)             (1)
    #              v               v
    #           -------   1    ---------   ?    ---------
    #  run_1:  | Load | ----> | AddNum | ----> | Result |
    #          -------        ---------        ---------
    #
    ############################################################################
    beam_dag_runner.BeamDagRunner().run(pipeline_pb_run_1)
    ############################################################################
    # PART 1c: Full run -- Verify result (1 + 1 == 2).
    #
    #             (1)             (1)
    #              v               v
    #           -------   1    ---------   2    ---------
    #  run_1:  | Load | ----> | AddNum | ----> | Result |
    #          -------        ---------        ---------
    #
    ############################################################################
    self.assertResultEquals(pipeline_pb_run_1, 2)

    ############################################################################
    # PART 2: Modify the `AddNum` component.
    #
    # We reuse the same `Load` component.
    # However, `AddNum` now takes 5 as its exec_param.
    # `Result` takes its input from the modified `AddNum` component.
    #
    #             (1)             (5)
    #              v               v
    #           -------        ---------        ---------
    #  run_2:  | Load | ----> | AddNum | ----> | Result |
    #          -------        ---------        ---------
    #
    ############################################################################
    add_num_v2 = AddNum(to_add=5, num=load.outputs['num'])  # pylint: disable=no-value-for-parameter
    load.remove_downstream_node(add_num)  # This line is important.
    result_v2 = Result(result=add_num_v2.outputs['added_num'])
    pipeline_pb_run_2 = self.make_pipeline(
        components=[load, add_num_v2, result_v2], run_id='run_2')

    ############################################################################
    # PART 3a: Partial run -- DAG filtering.
    #
    # We specify that we only wish to run the pipeline from `AddNum` onwards.
    #
    #                             (5)
    #                              v
    #                          ---------        ---------
    #  run_2:          --/--> | AddNum | ----> | Result |
    #                         ---------        ---------
    #
    ############################################################################
    pipeline_pb_run_2, _ = partial_run_utils.filter_pipeline(
        pipeline_pb_run_2,
        from_nodes=lambda node_id: (node_id == add_num_v2.id))
    ############################################################################
    # PART 3b: Partial run -- Reuse node outputs.
    #
    # Reuses output artifacts from the previous run of `Load`, so that
    # `AddNum` can resolve its inputs.
    #
    #                             (5)
    #                              v
    #                     ?    ---------        ---------
    #  run_2:           ----> | AddNum | ----> | Result |
    #                         ---------        ---------
    #
    ############################################################################
    with metadata.Metadata(self.metadata_config) as m:
      partial_run_utils.reuse_node_outputs(
          m,
          pipeline_name=self.pipeline_name,
          node_id=load.id,
          old_run_id='run_1',
          new_run_id='run_2')
    ############################################################################
    # PART 3c: Partial run -- Run pipeline.
    #
    #                             (5)
    #                              v
    #                     ?    ---------   ?    ---------
    #  run_2:           ----> | AddNum | ----> | Result |
    #                         ---------        ---------
    #
    ############################################################################
    beam_dag_runner.BeamDagRunner().run(pipeline_pb_run_2)
    ############################################################################
    # PART 3d: Partial run -- Verify result (1 + 5 == 6).
    #
    #                             (5)
    #                              v
    #                     1    ---------   6    ---------
    #  run_2:           ----> | AddNum | ----> | Result |
    #                         ---------        ---------
    #
    ############################################################################
    self.assertResultEquals(pipeline_pb_run_2, 6)

  def testNonExistentContext_lookupError(self):
    """Raise error if user provides non-existent pipeline_run_id or node_id."""
    load = Load(start_num=1)  # pylint: disable=no-value-for-parameter
    add_num = AddNum(to_add=1, num=load.outputs['num'])  # pylint: disable=no-value-for-parameter
    result = Result(result=add_num.outputs['added_num'])
    pipeline_pb_run_1 = self.make_pipeline(
        components=[load, add_num, result], run_id='run_1')
    beam_dag_runner.BeamDagRunner().run(pipeline_pb_run_1)
    with metadata.Metadata(self.metadata_config) as m:
      with self.assertRaisesRegex(LookupError,
                                  'pipeline_run_id .* not found in MLMD.'):
        partial_run_utils.reuse_node_outputs(
            m,
            pipeline_name=self.pipeline_name,
            node_id=load.id,
            old_run_id='non_existent_run_id',
            new_run_id='run_2')
      with self.assertRaisesRegex(LookupError,
                                  'node context .* not found in MLMD.'):
        partial_run_utils.reuse_node_outputs(
            m,
            pipeline_name=self.pipeline_name,
            node_id='non_existent_node_id',
            old_run_id='run_1',
            new_run_id='run_2')

  def testNoPreviousSuccessfulExecution_lookupError(self):
    """Raise error if user tries to reuse node w/o any successful Executions."""
    load_fail = LoadFail(start_num=1)  # pylint: disable=no-value-for-parameter
    add_num = AddNum(to_add=1, num=load_fail.outputs['num'])  # pylint: disable=no-value-for-parameter
    result = Result(result=add_num.outputs['added_num'])
    pipeline_pb_run_1 = self.make_pipeline(
        components=[load_fail, add_num, result], run_id='run_1')
    try:
      # Suppress exception here, since we expect this pipeline run to fail.
      beam_dag_runner.BeamDagRunner().run(pipeline_pb_run_1)
    except Exception:  # pylint: disable=broad-except
      pass

    with metadata.Metadata(self.metadata_config) as m:
      with self.assertRaisesRegex(LookupError,
                                  'No previous successful executions found'):
        partial_run_utils.reuse_node_outputs(
            m,
            pipeline_name=self.pipeline_name,
            node_id=load_fail.id,
            old_run_id='run_1',
            new_run_id='run_2')

  def testIdempotence_retryReusesRegisteredCacheExecution(self):
    """Ensures that there is only one registered cache execution.

    If an execution is registered but orchestrator gets preempted right after,
    a naive implementation will cause this execution to be abandoned (in state
    RUNNING) and a new cached execution to be created with the same
    pipeline_run_id context upon restart.

    TFX orchestrator requires that a node has at most one active execution.
    It ensures this for executions it creates directly. TFX frontend GUI would
    also show abandoned active executions as "running" which is confusing to
    users since they will never complete.

    Thus, we need to make sure that even if `reuse_node_outputs` was retried
    after registration succeeds, there will only be one cache execution.
    """
    load = Load(start_num=1)  # pylint: disable=no-value-for-parameter
    add_num = AddNum(to_add=1, num=load.outputs['num'])  # pylint: disable=no-value-for-parameter
    result = Result(result=add_num.outputs['added_num'])
    pipeline_pb_run_1 = self.make_pipeline(
        components=[load, add_num, result], run_id='run_1')
    beam_dag_runner.BeamDagRunner().run(pipeline_pb_run_1)
    self.assertResultEquals(pipeline_pb_run_1, 2)

    add_num_v2 = AddNum(to_add=5, num=load.outputs['num'])  # pylint: disable=no-value-for-parameter
    load.remove_downstream_node(add_num)  # This line is important.
    result_v2 = Result(result=add_num_v2.outputs['added_num'])
    pipeline_pb_run_2 = self.make_pipeline(
        components=[load, add_num_v2, result_v2], run_id='run_2')

    pipeline_pb_run_2, _ = partial_run_utils.filter_pipeline(
        pipeline_pb_run_2,
        from_nodes=lambda node_id: (node_id == add_num_v2.id))

    with metadata.Metadata(self.metadata_config) as m:
      # Simulate success in registering cache execution, but failure when
      # publishing -- e.g., job was pre-empted.
      with mock.patch.object(
          execution_publish_utils, 'publish_cached_execution',
          autospec=True) as mock_publish_cached_execution:
        mock_publish_cached_execution.side_effect = ConnectionResetError()
        try:
          partial_run_utils.reuse_node_outputs(
              m,
              pipeline_name=self.pipeline_name,
              node_id=load.id,
              old_run_id='run_1',
              new_run_id='run_2')
        except ConnectionResetError:
          pass
      # Simulate a retry attempt that succeeds.
      partial_run_utils.reuse_node_outputs(
          m,
          pipeline_name=self.pipeline_name,
          node_id=load.id,
          old_run_id='run_1',
          new_run_id='run_2')

      # Make sure that only one new cache execution is created,
      # despite the retry.
      new_cache_executions = (
          execution_lib.get_executions_associated_with_all_contexts(
              m,
              contexts=[
                  # Same node context
                  m.store.get_context_by_type_and_name(
                      type_name=constants.NODE_CONTEXT_TYPE_NAME,
                      context_name=compiler_utils.node_context_name(
                          self.pipeline_name, load.id)),
                  # New pipeline run context
                  m.store.get_context_by_type_and_name(
                      type_name=constants.PIPELINE_RUN_CONTEXT_TYPE_NAME,
                      context_name='run_2'),
              ]))
      self.assertLen(new_cache_executions, 1)

    # Make sure it still works!
    beam_dag_runner.BeamDagRunner().run(pipeline_pb_run_2)
    self.assertResultEquals(pipeline_pb_run_2, 6)

  def testIdempotence_retryReusesPreviousSuccessfulCacheExecution(self):
    """Ensures idempotence.

    This test checks that after the first successful cache execution, following
    reuse_node_execution attempts will be no-op.
    """
    load = Load(start_num=1)  # pylint: disable=no-value-for-parameter
    add_num = AddNum(to_add=1, num=load.outputs['num'])  # pylint: disable=no-value-for-parameter
    result = Result(result=add_num.outputs['added_num'])
    pipeline_pb_run_1 = self.make_pipeline(
        components=[load, add_num, result], run_id='run_1')
    beam_dag_runner.BeamDagRunner().run(pipeline_pb_run_1)
    self.assertResultEquals(pipeline_pb_run_1, 2)

    add_num_v2 = AddNum(to_add=5, num=load.outputs['num'])  # pylint: disable=no-value-for-parameter
    load.remove_downstream_node(add_num)  # This line is important.
    result_v2 = Result(result=add_num_v2.outputs['added_num'])
    pipeline_pb_run_2 = self.make_pipeline(
        components=[load, add_num_v2, result_v2], run_id='run_2')

    pipeline_pb_run_2, _ = partial_run_utils.filter_pipeline(
        pipeline_pb_run_2,
        from_nodes=lambda node_id: (node_id == add_num_v2.id))

    with metadata.Metadata(self.metadata_config) as m:
      # Simulate two successful attempts.
      for _ in [1, 2]:
        partial_run_utils.reuse_node_outputs(
            m,
            pipeline_name=self.pipeline_name,
            node_id=load.id,
            old_run_id='run_1',
            new_run_id='run_2')

      # Make sure that only one new cache execution is created,
      # despite the multiple attempts.
      new_cache_executions = (
          execution_lib.get_executions_associated_with_all_contexts(
              m,
              contexts=[
                  # Same node context
                  m.store.get_context_by_type_and_name(
                      type_name=constants.NODE_CONTEXT_TYPE_NAME,
                      context_name=compiler_utils.node_context_name(
                          self.pipeline_name, load.id)),
                  # New pipeline run context
                  m.store.get_context_by_type_and_name(
                      type_name=constants.PIPELINE_RUN_CONTEXT_TYPE_NAME,
                      context_name='run_2'),
              ]))
      self.assertLen(new_cache_executions, 1)

    # Make sure it still works!
    beam_dag_runner.BeamDagRunner().run(pipeline_pb_run_2)
    self.assertResultEquals(pipeline_pb_run_2, 6)


if __name__ == '__main__':
  absltest.main()
