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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from tfx.dsl.compiler import compiler
from tfx.dsl.compiler import constants
from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import component
from tfx.orchestration import metadata
from tfx.orchestration import pipeline as pipeline_lib
from tfx.orchestration.beam import beam_dag_runner
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable import partial_run_utils
from tfx.orchestration.portable import runtime_parameter_utils
from tfx.proto.orchestration import pipeline_pb2 as p_pb2
from tfx.types import standard_artifacts
from tfx.utils import test_case_utils

from ml_metadata.proto import metadata_store_pb2 as m_pb2

_PIPELINE_RUN_CONTEXT_KEY = constants.PIPELINE_RUN_CONTEXT_TYPE_NAME


def _to_context_spec(type_name: str, name: str) -> p_pb2.ContextSpec:
  return p_pb2.ContextSpec(
      type=m_pb2.ContextType(name=type_name),
      name=p_pb2.Value(field_value=m_pb2.Value(string_value=name)))


def _to_output_spec(artifact_name: str) -> p_pb2.OutputSpec:
  return p_pb2.OutputSpec(
      artifact_spec=p_pb2.OutputSpec.ArtifactSpec(
          type=m_pb2.ArtifactType(name=artifact_name)))


def _to_input_channel(
    producer_output_key: str, producer_node_id: str, artifact_type: str,
    context_names: Mapping[str, str]) -> p_pb2.InputSpec.Channel:
  # pylint: disable=g-complex-comprehension
  context_queries = [
      p_pb2.InputSpec.Channel.ContextQuery(
          type=m_pb2.ContextType(name=context_type),
          name=p_pb2.Value(field_value=m_pb2.Value(string_value=context_name)))
      for context_type, context_name in context_names.items()
  ]
  return p_pb2.InputSpec.Channel(
      output_key=producer_output_key,
      producer_node_query=p_pb2.InputSpec.Channel.ProducerNodeQuery(
          id=producer_node_id),
      context_queries=context_queries,
      artifact_query=p_pb2.InputSpec.Channel.ArtifactQuery(
          type=m_pb2.ArtifactType(name=artifact_type)))


class PipelineFilteringTest(parameterized.TestCase, test_case_utils.TfxTest):

  def testSubpipeline_error(self):
    """If Pipeline contains sub-pipeline, raise NotImplementedError."""
    node_a = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='A'), id='a'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'a')
            ])))
    sub_pipeline_node = p_pb2.Pipeline.PipelineOrNode(
        sub_pipeline=p_pb2.Pipeline(
            pipeline_info=p_pb2.PipelineInfo(id='my_subpipeline'),
            execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
            nodes=[node_a]))
    input_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[sub_pipeline_node])

    with self.assertRaisesRegex(
        ValueError, 'Pipeline filtering not supported for '
        'pipelines with sub-pipelines.'):
      _ = partial_run_utils.filter_pipeline(
          input_pipeline,
          from_nodes=lambda _: True,
          to_nodes=lambda _: True,
      )

  def testNoFilter(self):
    """Basic case where there are no filters applied.

    input_pipeline: node_a -> node_b -> node_c
    from_node: all nodes
    to_node: all nodes
    expected output_pipeline: node_a -> node_b -> node_c
    """

    node_a = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='A'), id='a'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'a')
            ]),
            outputs=p_pb2.NodeOutputs(outputs={'out': _to_output_spec('AB')}),
            downstream_nodes=['b']))
    node_b = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='B'), id='b'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'b')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                _to_input_channel(
                                    producer_node_id='a',
                                    producer_output_key='out',
                                    artifact_type='AB',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'a'
                                    })
                            ],
                            min_count=1)
                }),
            outputs=p_pb2.NodeOutputs(outputs={'out': _to_output_spec('BC')}),
            upstream_nodes=['a'],
            downstream_nodes=['c']))
    node_c = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='C'), id='c'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'c')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                _to_input_channel(
                                    producer_node_id='b',
                                    producer_output_key='out',
                                    artifact_type='BC',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'b'
                                    })
                            ],
                            min_count=1)
                }),
            upstream_nodes=['b']))
    input_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_a, node_b, node_c])

    filtered_pipeline, _ = partial_run_utils.filter_pipeline(input_pipeline)

    self.assertProtoEquals(input_pipeline, filtered_pipeline)

  def testFilterOutNothing(self):
    """Basic case where no nodes are filtered out.

    input_pipeline: node_a -> node_b -> node_c
    from_node: node_a
    to_node: node_c
    expected output_pipeline: node_a -> node_b -> node_c
    expected input_channel_ref: (empty)
    """
    node_a = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='A'), id='a'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'a')
            ]),
            outputs=p_pb2.NodeOutputs(outputs={'out': _to_output_spec('AB')}),
            downstream_nodes=['b']))
    node_b = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='B'), id='b'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'b')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                _to_input_channel(
                                    producer_node_id='a',
                                    producer_output_key='out',
                                    artifact_type='AB',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'a'
                                    })
                            ],
                            min_count=1)
                }),
            outputs=p_pb2.NodeOutputs(outputs={'out': _to_output_spec('BC')}),
            upstream_nodes=['a'],
            downstream_nodes=['c']))
    node_c = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='C'), id='c'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'c')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                _to_input_channel(
                                    producer_node_id='b',
                                    producer_output_key='out',
                                    artifact_type='BC',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'b'
                                    })
                            ],
                            min_count=1)
                }),
            upstream_nodes=['b']))
    input_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_a, node_b, node_c])

    filtered_pipeline, input_channel_dict = partial_run_utils.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'a'),
        to_nodes=lambda node_id: (node_id == 'c'))

    self.assertProtoEquals(input_pipeline, filtered_pipeline)
    self.assertEqual(input_channel_dict, {})

  def testFilterOutSinkNode(self):
    """Filter out a node that has upstream nodes but no downstream nodes.

    input_pipeline: node_a -> node_b -> node_c
    to_node: node_b
    expected_output_pipeline: node_a -> node_b
    expected input_channel_ref: (empty)
    """
    node_a = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='A'), id='a'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'a')
            ]),
            outputs=p_pb2.NodeOutputs(outputs={'out': _to_output_spec('AB')}),
            downstream_nodes=['b']))
    node_b = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='B'), id='b'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'b')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                _to_input_channel(
                                    producer_node_id='a',
                                    producer_output_key='out',
                                    artifact_type='AB',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'a'
                                    })
                            ],
                            min_count=1)
                }),
            outputs=p_pb2.NodeOutputs(outputs={'out': _to_output_spec('BC')}),
            upstream_nodes=['a'],
            downstream_nodes=['c']))
    node_c = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='C'), id='c'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'c')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                _to_input_channel(
                                    producer_node_id='b',
                                    producer_output_key='out',
                                    artifact_type='BC',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'b'
                                    })
                            ],
                            min_count=1)
                }),
            upstream_nodes=['b']))
    input_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_a, node_b, node_c])

    filtered_pipeline, input_channel_dict = partial_run_utils.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'a'),
        to_nodes=lambda node_id: (node_id == 'b'),
    )

    node_b_no_downstream = p_pb2.Pipeline.PipelineOrNode()
    node_b_no_downstream.CopyFrom(node_b)
    del node_b_no_downstream.pipeline_node.downstream_nodes[:]
    expected_output_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_a, node_b_no_downstream])
    self.assertProtoEquals(expected_output_pipeline, filtered_pipeline)
    self.assertEqual(input_channel_dict, {})

  def testFilterOutSourceNode(self):
    """Filter out a node that has no upstream nodes but has downstream nodes.

    input_pipeline: node_a -> node_b -> node_c
    from_node: node_b
    to_node: node_c
    expected_output_pipeline: node_b -> node_c
    expected input_channel_ref: {node_a: AB}
    """
    node_a = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='A'), id='a'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'a')
            ]),
            outputs=p_pb2.NodeOutputs(outputs={'out': _to_output_spec('AB')}),
            downstream_nodes=['b']))
    node_b = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='B'), id='b'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'b')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                _to_input_channel(
                                    producer_node_id='a',
                                    producer_output_key='out',
                                    artifact_type='AB',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'a'
                                    })
                            ],
                            min_count=1)
                }),
            outputs=p_pb2.NodeOutputs(outputs={'out': _to_output_spec('BC')}),
            upstream_nodes=['a'],
            downstream_nodes=['c']))
    node_c = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='C'), id='c'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'c')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                _to_input_channel(
                                    producer_node_id='b',
                                    producer_output_key='out',
                                    artifact_type='BC',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'b'
                                    })
                            ],
                            min_count=1)
                }),
            upstream_nodes=['b']))
    input_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_a, node_b, node_c])

    filtered_pipeline, input_channel_dict = partial_run_utils.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'b'),
        to_nodes=lambda node_id: (node_id == 'c'),
    )

    node_b_fixed = p_pb2.Pipeline.PipelineOrNode()
    node_b_fixed.CopyFrom(node_b)
    del node_b_fixed.pipeline_node.upstream_nodes[:]
    expected_output_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_b_fixed, node_c])
    self.assertProtoEquals(expected_output_pipeline, filtered_pipeline)
    self.assertEqual(
        input_channel_dict, {
            'a': [
                _to_input_channel(
                    producer_node_id='a',
                    producer_output_key='out',
                    artifact_type='AB',
                    context_names={
                        'pipeline': 'my_pipeline',
                        'component': 'a'
                    }),
            ],
        })

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
    node_a = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='A'), id='a'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'a')
            ]),
            outputs=p_pb2.NodeOutputs(outputs={
                'out_b': _to_output_spec('AB'),
                'out_c': _to_output_spec('AC')
            }),
            downstream_nodes=['b', 'c']))
    node_b = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='B'), id='b'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'b')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in':
                        p_pb2.InputSpec(
                            channels=[
                                _to_input_channel(
                                    producer_node_id='a',
                                    producer_output_key='out_b',
                                    artifact_type='AB',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'a'
                                    })
                            ],
                            min_count=1)
                }),
            outputs=p_pb2.NodeOutputs(outputs={'out': _to_output_spec('BC')}),
            upstream_nodes=['a'],
            downstream_nodes=['c']))
    node_c = p_pb2.Pipeline.PipelineOrNode(
        pipeline_node=p_pb2.PipelineNode(
            node_info=p_pb2.NodeInfo(
                type=m_pb2.ExecutionType(name='C'), id='c'),
            contexts=p_pb2.NodeContexts(contexts=[
                _to_context_spec('pipeline', 'my_pipeline'),
                _to_context_spec('component', 'c')
            ]),
            inputs=p_pb2.NodeInputs(
                inputs={
                    'in_a':
                        p_pb2.InputSpec(
                            channels=[
                                _to_input_channel(
                                    producer_node_id='a',
                                    producer_output_key='out_c',
                                    artifact_type='AC',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'a'
                                    })
                            ],
                            min_count=1),
                    'in_b':
                        p_pb2.InputSpec(
                            channels=[
                                _to_input_channel(
                                    producer_node_id='b',
                                    producer_output_key='out',
                                    artifact_type='BC',
                                    context_names={
                                        'pipeline': 'my_pipeline',
                                        'component': 'b'
                                    })
                            ],
                            min_count=1)
                }),
            upstream_nodes=['a', 'b']))
    input_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_a, node_b, node_c])

    filtered_pipeline, input_channel_dict = partial_run_utils.filter_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'b'),
        to_nodes=lambda node_id: (node_id == 'c'),
    )

    node_b_fixed = p_pb2.Pipeline.PipelineOrNode()
    node_b_fixed.CopyFrom(node_b)
    del node_b_fixed.pipeline_node.upstream_nodes[:]
    node_c_fixed = p_pb2.Pipeline.PipelineOrNode()
    node_c_fixed.CopyFrom(node_c)
    node_c_fixed.pipeline_node.upstream_nodes[:] = 'b'
    expected_output_pipeline = p_pb2.Pipeline(
        pipeline_info=p_pb2.PipelineInfo(id='my_pipeline'),
        execution_mode=p_pb2.Pipeline.ExecutionMode.SYNC,
        nodes=[node_b_fixed, node_c_fixed])
    self.assertProtoEquals(expected_output_pipeline, filtered_pipeline)
    self.assertEqual(
        input_channel_dict, {
            'a': [
                _to_input_channel(
                    producer_node_id='a',
                    producer_output_key='out_b',
                    artifact_type='AB',
                    context_names={
                        'pipeline': 'my_pipeline',
                        'component': 'a'
                    }),
                _to_input_channel(
                    producer_node_id='a',
                    producer_output_key='out_c',
                    artifact_type='AC',
                    context_names={
                        'pipeline': 'my_pipeline',
                        'component': 'a'
                    }),
            ],
        })


# pylint: disable=invalid-name
# pytype: disable=wrong-arg-types
@component
def Load(start_num: Parameter[int],
         num: OutputArtifact[standard_artifacts.Integer]):
  num.value = start_num
  logging.info('Load with start_num=%s', start_num)


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


def _node_inputs_by_id(pipeline: p_pb2.Pipeline,
                       node_id: str) -> p_pb2.NodeInputs:
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

  def make_pipeline(self, components, run_id: str) -> p_pb2.Pipeline:
    pipeline = pipeline_lib.Pipeline(self.pipeline_name, self.pipeline_root,
                                     self.metadata_config, components)
    result = compiler.Compiler().compile(pipeline)
    runtime_parameter_utils.substitute_runtime_parameter(
        result, {constants.PIPELINE_RUN_ID_PARAMETER_NAME: run_id})
    return result

  def assertResultEquals(self, pipeline_pb: p_pb2.Pipeline,
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


if __name__ == '__main__':
  absltest.main()
