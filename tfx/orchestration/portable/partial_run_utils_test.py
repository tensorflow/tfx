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

import collections
from typing import Dict, List, Mapping, Optional, Set, Tuple, Union
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


class MarkPipelineFnTest(parameterized.TestCase, test_case_utils.TfxTest):

  def _checkNodeExecutionOptions(self, pipeline: pipeline_pb2.Pipeline,
                                 snapshot_node: Optional[str],
                                 nodes_to_run: Set[str],
                                 nodes_requiring_snapshot: Set[str],
                                 nodes_to_skip: Set[str],
                                 nodes_to_reuse: Set[str]):
    for node in pipeline.nodes:
      try:
        node_id = node.pipeline_node.node_info.id
        run_or_skip = node.pipeline_node.execution_options.WhichOneof(
            'partial_run_option')
        self.assertIsNotNone(run_or_skip)
        # assert partial_run_option == 'run' iff node_id in nodes_to_run
        if node_id in nodes_to_run:
          if node_id in nodes_to_skip:
            raise ValueError(f'node_id {node_id} appears in both nodes_to_run '
                             'and nodes_to_skip.')
          self.assertEqual(run_or_skip, 'run')
          # assert snapshot_settings is set iff node_id == snapshot_node
          if node_id == snapshot_node:
            self.assertTrue(
                node.pipeline_node.execution_options.run.perform_snapshot)
          else:
            self.assertFalse(
                node.pipeline_node.execution_options.run.perform_snapshot)
          # assert depends_on_snapshot == (node_id in nodes_requiring_snapshot)
          if node.pipeline_node.execution_options.run.depends_on_snapshot:
            self.assertIn(node_id, nodes_requiring_snapshot)
          else:
            self.assertNotIn(node_id, nodes_requiring_snapshot)
        elif node_id in nodes_to_skip:
          self.assertEqual(run_or_skip, 'skip')
          # assert reuse_artifacts iff node_id in nodes_to_reuse
          if node_id in nodes_to_reuse:
            self.assertTrue(
                node.pipeline_node.execution_options.skip.reuse_artifacts)
          else:
            self.assertFalse(
                node.pipeline_node.execution_options.skip.reuse_artifacts)
        else:
          raise ValueError(f'node_id {node_id} appears in neither nodes_to_run '
                           'nor nodes_to_skip.')
      except AssertionError:
        logging.exception('_checkNodeExecutionOptions failed in node: %s',
                          node.pipeline_node)
        raise

  def _createInputPipeline(self, node_to_downstream_nodes: Dict[str,
                                                                List[str]]):
    pipeline_name = 'test_pipeline'
    input_pipeline = pipeline_pb2.Pipeline()
    input_pipeline.pipeline_info.id = pipeline_name
    input_pipeline.execution_mode = pipeline_pb2.Pipeline.SYNC

    node_to_upstream_nodes = collections.defaultdict(list)
    for node, downstream_nodes in node_to_downstream_nodes.items():
      for downstream_node in downstream_nodes:
        node_to_upstream_nodes[downstream_node].append(node)

    for node, downstream_nodes in node_to_downstream_nodes.items():
      # Use upper case of node name as its type.
      node_type = node.upper()
      n = input_pipeline.nodes.add()
      n.pipeline_node.node_info.id = node
      n.pipeline_node.node_info.type.name = node_type
      n.pipeline_node.contexts.contexts.extend([
          _to_context_spec('pipeline', pipeline_name),
          _to_context_spec('component', node),
      ])

      if node in node_to_upstream_nodes:
        for upstream_node in node_to_upstream_nodes[node]:
          upstream_node_type = upstream_node.upper()
          n.pipeline_node.inputs.inputs['in'].channels.extend([
              _to_input_channel(
                  producer_node_id=upstream_node,
                  producer_output_key='out',
                  artifact_type=upstream_node_type + node_type,
                  context_names={
                      'pipeline': pipeline_name,
                      'component': node,
                  }),
          ])
          n.pipeline_node.inputs.inputs['in'].min_count = 1
          n.pipeline_node.upstream_nodes.extend([upstream_node])

      for downstream_node in downstream_nodes:
        downstream_node_type = downstream_node.upper()
        n.pipeline_node.outputs.outputs['out'].CopyFrom(
            _to_output_spec(node_type + downstream_node_type))
        n.pipeline_node.downstream_nodes.extend([downstream_node])

    return input_pipeline

  def testAsyncPipeline_error(self):
    """If Pipeline has execution_mode ASYNC, raise ValueError."""
    input_pipeline = pipeline_pb2.Pipeline()
    input_pipeline.pipeline_info.id = 'my_pipeline'
    input_pipeline.execution_mode = pipeline_pb2.Pipeline.ASYNC

    with self.assertRaisesRegex(
        ValueError, 'Pipeline filtering is only supported for SYNC pipelines.'):
      partial_run_utils.mark_pipeline(input_pipeline)

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
      partial_run_utils.mark_pipeline(input_pipeline)

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
      partial_run_utils.mark_pipeline(input_pipeline)

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
      partial_run_utils.mark_pipeline(input_pipeline)

  def testNoFilter(self):
    """Basic case where there are no filters applied.

    pipeline: node_a -> node_b -> node_c

    expected snapshot_node: None
    expected nodes_to_run: [node_a, node_b, node_c]
    expected nodes_requiring_snapshot: []
    expected nodes_to_skip: []
    expected nodes_to_reuse: []
    """
    input_pipeline = self._createInputPipeline({
        'a': ['b'],
        'b': ['c'],
        'c': []
    })
    partial_run_utils.mark_pipeline(input_pipeline)

    self._checkNodeExecutionOptions(
        input_pipeline,
        snapshot_node=None,
        nodes_to_run=set(['a', 'b', 'c']),
        nodes_requiring_snapshot=set(),
        nodes_to_skip=set(),
        nodes_to_reuse=set())

  def testFilterOutNothing(self):
    """Basic case where no nodes are filtered out.

    input_pipeline: node_a -> node_b -> node_c
    from_node: node_a
    to_node: node_c

    expected snapshot_node: None
    expected nodes_to_run: [node_a, node_b, node_c]
    expected nodes_requiring_snapshot: []
    expected nodes_to_skip: []
    expected nodes_to_reuse: []
    """
    input_pipeline = self._createInputPipeline({
        'a': ['b'],
        'b': ['c'],
        'c': []
    })
    partial_run_utils.mark_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'a'),
        to_nodes=lambda node_id: (node_id == 'c'))

    self._checkNodeExecutionOptions(
        input_pipeline,
        snapshot_node=None,
        nodes_to_run=set(['a', 'b', 'c']),
        nodes_requiring_snapshot=set(),
        nodes_to_skip=set(),
        nodes_to_reuse=set())

  def testFilterOutSinkNode(self):
    """Filter out a node that has upstream nodes but no downstream nodes.

    input_pipeline: node_a -> node_b -> node_c
    to_node: node_b

    expected snapshot_node: None
    expected nodes_to_run: [node_a, node_b]
    expected nodes_requiring_snapshot: []
    expected nodes_to_skip: [node_c]
    expected nodes_to_reuse: []
    """
    input_pipeline = self._createInputPipeline({
        'a': ['b'],
        'b': ['c'],
        'c': []
    })
    partial_run_utils.mark_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'a'),
        to_nodes=lambda node_id: (node_id == 'b'))

    self._checkNodeExecutionOptions(
        input_pipeline,
        snapshot_node=None,
        nodes_to_run=set(['a', 'b']),
        nodes_requiring_snapshot=set(),
        nodes_to_skip=set(['c']),
        nodes_to_reuse=set())

  def testFilterOutSourceNode(self):
    """Filter out a node that has no upstream nodes but has downstream nodes.

    input_pipeline: node_a -> node_b -> node_c
    from_node: node_b
    to_node: node_c

    expected snapshot_node: node_b
    expected nodes_to_run: [node_b, node_c]
    expected nodes_requiring_snapshot: [node_b]
    expected nodes_to_skip: [node_a]
    expected nodes_to_reuse: [node_a]
    """
    input_pipeline = self._createInputPipeline({
        'a': ['b'],
        'b': ['c'],
        'c': []
    })
    partial_run_utils.mark_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'b'),
        to_nodes=lambda node_id: (node_id == 'c'))

    self._checkNodeExecutionOptions(
        input_pipeline,
        snapshot_node='b',
        nodes_to_run=set(['b', 'c']),
        nodes_requiring_snapshot=set(['b']),
        nodes_to_skip=set(['a']),
        nodes_to_reuse=set(['a']))

  def testFilterOutSourceNode_triangle(self):
    """Filter out a source node in a triangle.

    input_pipeline:
        node_a -> node_b -> node_c
             |--------------^
    from_node: node_b
    to_node: node_c

    expected snapshot_node: node_b
    expected nodes_to_run: [node_b, node_c]
    expected nodes_requiring_snapshot: [node_b, node_c]
    expected nodes_to_skip: [node_a]
    expected nodes_to_reuse: [node_a]
    """
    input_pipeline = self._createInputPipeline({
        'a': ['b', 'c'],
        'b': ['c'],
        'c': []
    })
    partial_run_utils.mark_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'b'),
        to_nodes=lambda node_id: (node_id == 'c'),
    )

    self._checkNodeExecutionOptions(
        input_pipeline,
        snapshot_node='b',
        nodes_to_run=set(['b', 'c']),
        nodes_requiring_snapshot=set(['b', 'c']),
        nodes_to_skip=set(['a']),
        nodes_to_reuse=set(['a']))

  def testRunMiddleNode(self):
    """Run only the middle node.

    input_pipeline:
        node_a -> node_b -> node_c -> node_d
    from_node: node_c
    to_node: node_c

    expected snapshot_node: node_c
    expected nodes_to_run: [node_c]
    expected nodes_requiring_snapshot: [node_c]
    expected nodes_to_skip: [node_a, node_b, node_d]
    expected nodes_to_reuse: [node_a, node_b]
    """
    input_pipeline = self._createInputPipeline({
        'a': ['b'],
        'b': ['c'],
        'c': ['d'],
        'd': []
    })
    partial_run_utils.mark_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'c'),
        to_nodes=lambda node_id: (node_id == 'c'),
    )

    self._checkNodeExecutionOptions(
        input_pipeline,
        snapshot_node='c',
        nodes_to_run=set(['c']),
        nodes_requiring_snapshot=set(['c']),
        nodes_to_skip=set(['a', 'b', 'd']),
        nodes_to_reuse=set(['a', 'b']))

  def testRunSinkNode(self):
    """Run only a sink node.

    Note that even though node_a does not need to be reused in order to run
    node_c, we deliberately reuse node_a anyway to make sure that after the
    snapshot, node_b's dependencies are also associated with the same pipeline
    run. It makes it easier to use BASE_PIPELINE_RUN strategy.

    input_pipeline:
        node_a -> node_b -> node_c
    from_node: node_c

    expected snapshot_node: node_c
    expected nodes_to_run: [node_c]
    expected nodes_requiring_snapshot: [node_c]
    expected nodes_to_skip: [node_a, node_b]
    expected nodes_to_reuse: [node_a, node_b]
    """
    input_pipeline = self._createInputPipeline({
        'a': ['b'],
        'b': ['c'],
        'c': []
    })
    partial_run_utils.mark_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'c'),
    )

    self._checkNodeExecutionOptions(
        input_pipeline,
        snapshot_node='c',
        nodes_to_run=set(['c']),
        nodes_requiring_snapshot=set(['c']),
        nodes_to_skip=set(['a', 'b']),
        nodes_to_reuse=set(['a', 'b']))

  def testRunSinkNode_triangle(self):
    """Filter out a source node in a triangle.

    input_pipeline:
        node_a -> node_b -> node_c
             |--------------^
    from_node: node_c

    expected snapshot_node: node_c
    expected nodes_to_run: [node_c]
    expected nodes_requiring_snapshot: [node_c]
    expected nodes_to_skip: [node_a, node_b]
    expected nodes_to_reuse: [node_a, node_b]
    """
    input_pipeline = self._createInputPipeline({
        'a': ['b', 'c'],
        'b': ['c'],
        'c': []
    })
    partial_run_utils.mark_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'c'),
    )

    self._checkNodeExecutionOptions(
        input_pipeline,
        snapshot_node='c',
        nodes_to_run=set(['c']),
        nodes_requiring_snapshot=set(['c']),
        nodes_to_skip=set(['a', 'b']),
        nodes_to_reuse=set(['a', 'b']))

  def testRunMiddleNode_twoIndependentDAGs(self):
    """Run only a middle node in a pipeline with two independent DAGs.

    Note that even though node_b2 only needs to reuse outputs from node_a2, we
    reuse outputs from the other independent DAG anyway. This makes it easier to
    use BASE_PIPELINE_RUN strategy.

    Also note that we do not reuse node_c2, since reusing it would associate
    the previous pipeline run artifacts with the current partial run, and will
    eventually result in an inconsistent MLMD state.

    input_pipeline:
        node_a1 -> node_b1 -> node_c1
        node_a2 -> node_b2 -> node_c2
    from_node: node_b2
    to_node: node_b2

    expected snapshot_node: node_b2
    expected nodes_to_run: [node_b2]
    expected nodes_requiring_snapshot: [node_b2]
    expected nodes_to_skip: [node_a1, node_b1, node_c1, node_a2, node_c2]
    expected nodes_to_reuse: [node_a1, node_b1, node_c1, node_a2]
    """
    input_pipeline = self._createInputPipeline({
        'a1': ['b1'],
        'b1': ['c1'],
        'c1': [],
        'a2': ['b2'],
        'b2': ['c2'],
        'c2': []
    })
    partial_run_utils.mark_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'b2'),
        to_nodes=lambda node_id: (node_id == 'b2'),
    )

    self._checkNodeExecutionOptions(
        input_pipeline,
        snapshot_node='b2',
        nodes_to_run=set(['b2']),
        nodes_requiring_snapshot=set(['b2']),
        nodes_to_skip=set(['a1', 'b1', 'c1', 'a2', 'c2']),
        nodes_to_reuse=set(['a1', 'b1', 'c1', 'a2']))

  def testReuseableNodes(self):
    """Node excluded will not be run.

    input_pipeline:
        node_a -> node_b -> node_c -> node_d
    from_node: node_b
    to_node: node_d

    expected snapshot_node: node_c
    expected nodes_to_run: [node_c]
    expected nodes_requiring_snapshot: [node_c]
    expected nodes_to_skip: [node_a, node_b, node_d]
    expected nodes_to_reuse: [node_a, node_b]
    """
    input_pipeline = self._createInputPipeline({
        'a': ['b'],
        'b': ['c'],
        'c': ['d'],
        'd': []
    })
    partial_run_utils.mark_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'b'),
        to_nodes=lambda node_id: (node_id == 'd'),
        skip_nodes=lambda node_id: (node_id == 'c'))
    self._checkNodeExecutionOptions(
        input_pipeline,
        snapshot_node='b',
        nodes_to_run=set(['b', 'c', 'd']),
        nodes_requiring_snapshot=set(['b']),
        nodes_to_skip=set(['a']),
        nodes_to_reuse=set(['a']))

    input_pipeline = self._createInputPipeline({
        'a': ['b'],
        'b': ['c'],
        'c': ['d'],
        'd': []
    })
    partial_run_utils.mark_pipeline(
        input_pipeline,
        from_nodes=lambda node_id: (node_id == 'b'),
        to_nodes=lambda node_id: (node_id == 'd'),
        skip_nodes=lambda node_id: (node_id == 'a' or node_id == 'b'))
    self._checkNodeExecutionOptions(
        input_pipeline,
        snapshot_node='c',
        nodes_to_run=set(['c', 'd']),
        nodes_requiring_snapshot=set(['c']),
        nodes_to_skip=set(['a', 'b']),
        nodes_to_reuse=set(['a', 'b']))


# pylint: disable=invalid-name
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
def SubtractNums(num_1: InputArtifact[standard_artifacts.Integer],
                 num_2: InputArtifact[standard_artifacts.Integer],
                 diff: OutputArtifact[standard_artifacts.Integer]):
  num_1, num_2 = num_1.value, num_2.value
  diff.value = num_1 - num_2
  logging.info('SubtractNums with num_1=%s, num_2=%s', num_1, num_2)


@component
def Result(result: InputArtifact[standard_artifacts.Integer]):
  result_value = result.value
  logging.info('Result with result=%s', result_value)


# pytype: enable=wrong-arg-types
# pylint: enable=invalid-name


def _node_inputs_by_id(pipeline: pipeline_pb2.Pipeline,
                       node_id: str) -> pipeline_pb2.PipelineNode:
  """Doesn't make a copy."""

  node_ids_seen = []
  for node in pipeline.nodes:
    if node.pipeline_node.node_info.id == node_id:
      return node.pipeline_node
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

  def make_pipeline(self,
                    components,
                    run_id: Optional[str] = None) -> pipeline_pb2.Pipeline:
    """Make compiled pipeline from components.

    Args:
      components: List of components.
      run_id: Optional.If provided, will be used to substitute the
        pipeline_run_id RuntimeParameter.

    Returns:
      The compiled Pipeline IR.
    """
    pipeline = pipeline_lib.Pipeline(self.pipeline_name, self.pipeline_root,
                                     self.metadata_config, components)
    result = compiler.Compiler().compile(pipeline)
    if run_id:
      runtime_parameter_utils.substitute_runtime_parameter(
          result, {constants.PIPELINE_RUN_ID_PARAMETER_NAME: run_id})
    return result

  def assertResultEqual(self, pipeline_pb: pipeline_pb2.Pipeline,
                        expected_result: Union[int, List[Tuple[str, int]]]):
    node_id_exp_result_tups = []
    if isinstance(expected_result, int):
      node_id_exp_result_tups.append((self.result_node_id, expected_result))
    else:
      node_id_exp_result_tups = expected_result

    with metadata.Metadata(self.metadata_config) as m:
      for node_id, exp_result in node_id_exp_result_tups:
        result_node_inputs = _node_inputs_by_id(pipeline_pb, node_id=node_id)
        input_resolution_result = inputs_utils.resolve_input_artifacts_v2(
            metadata_handler=m, pipeline_node=result_node_inputs)
        self.assertIsInstance(input_resolution_result, inputs_utils.Trigger)
        result_artifact = input_resolution_result[0]['result'][0]
        result_artifact.read()
        self.assertEqual(result_artifact.value, exp_result)

  def testSnapshot_removeFirstNode(self):
    """Tests that partial run with the first node removed works."""
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
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_1)
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
    self.assertResultEqual(pipeline_pb_run_1, 2)

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
    partial_run_utils.mark_pipeline(
        pipeline_pb_run_2,
        from_nodes=lambda node_id: (node_id == add_num_v2.id))
    ############################################################################
    # PART 3b: Partial run -- Run pipeline.
    #
    # Reuses output artifacts from the previous run of `Load`, so that
    # `AddNum` can resolve its inputs.
    #
    #                             (5)
    #                              v
    #                     1    ---------   ?    ---------
    #  run_2:           ----> | AddNum | ----> | Result |
    #                         ---------        ---------
    #
    ############################################################################
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_2)
    ############################################################################
    # PART 3c: Partial run -- Verify result (1 + 5 == 6).
    #
    #                             (5)
    #                              v
    #                     1    ---------   6    ---------
    #  run_2:           ----> | AddNum | ----> | Result |
    #                         ---------        ---------
    #
    ############################################################################
    self.assertResultEqual(pipeline_pb_run_2, 6)

  def testReusePipelineArtifacts_twoIndependentSubgraphs(self):
    """Tests a sequence of partial runs with independent sub-graphs."""
    ############################################################################
    # PART 1a: Full run -- Author pipeline.
    #
    # This pipeline consists of *two* independent sub-graphs.
    #
    #              (1)               (1)
    #               v                 v
    #           ---------        -----------        -----------
    #          | Load 1 | ----> | AddNum 1 | ----> | Result 1 |
    #          ---------        -----------        -----------
    #  run_1:
    #           ---------        -----------        -----------
    #          | Load 2 | ----> | AddNum 2 | ----> | Result 2 |
    #          ---------        -----------        -----------
    #              ^                 ^
    #            (10)              (10)
    #
    ############################################################################
    # pylint: disable=no-value-for-parameter
    load_1 = Load(start_num=1).with_id('load_1')
    add_num_1 = AddNum(to_add=1, num=load_1.outputs['num']).with_id('add_num_1')
    result_1 = Result(result=add_num_1.outputs['added_num']).with_id('result_1')
    load_2 = Load(start_num=10).with_id('load_2')
    add_num_2 = AddNum(
        to_add=10, num=load_2.outputs['num']).with_id('add_num_2')
    result_2 = Result(result=add_num_2.outputs['added_num']).with_id('result_2')
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_1 = self.make_pipeline(
        components=[load_1, add_num_1, result_1, load_2, add_num_2, result_2],
        run_id='run_1')
    ############################################################################
    # PART 1b: Full run -- Run and verify (1 + 1 == 2 and 10 + 10 == 20).
    #
    #              (1)               (1)
    #               v                 v
    #           ---------   1    -----------   2    -----------
    #          | Load 1 | ----> | AddNum 1 | ----> | Result 1 |
    #          ---------        -----------        -----------
    #  run_1:
    #           ---------  10    -----------  20    -----------
    #          | Load 2 | ----> | AddNum 2 | ----> | Result 2 |
    #          ---------        -----------        -----------
    #              ^                 ^
    #            (10)              (10)
    #
    ############################################################################
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_1)
    self.assertResultEqual(pipeline_pb_run_1, [(result_1.id, 2),
                                               (result_2.id, 20)])

    ############################################################################
    # PART 2: Modify the `AddNum` component in the first "sub-pipeline".
    #
    #              (1)               (5)
    #               v                 v
    #           ---------        -----------        -----------
    #          | Load 1 | ----> | AddNum 1 | ----> | Result 1 |
    #          ---------        -----------        -----------
    #
    ############################################################################
    # pylint: disable=no-value-for-parameter
    add_num_1_v2 = AddNum(
        to_add=5, num=load_1.outputs['num']).with_id('add_num_1')
    load_1.remove_downstream_node(add_num_1)  # This line is important.
    result_1_v2 = Result(
        result=add_num_1_v2.outputs['added_num']).with_id('result_1')
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_2 = self.make_pipeline(
        components=[
            load_1,
            add_num_1_v2,
            result_1_v2,
            load_2,
            add_num_2,
            result_2,
        ],
        run_id='run_2')

    ############################################################################
    # PART 3a: Partial run -- DAG filtering.
    #
    # Only run the first sub-graph from `AddNum 1` onwards.
    #
    #                              (5)
    #                               v
    #                          -----------        -----------
    #  run_2:          --/--> | AddNum 1 | ----> | Result 1 |
    #                         -----------        -----------
    #
    ############################################################################
    partial_run_utils.mark_pipeline(
        pipeline_pb_run_2,
        from_nodes=lambda node_id: (node_id == add_num_1_v2.id))
    ############################################################################
    # PART 3b: Partial run -- Reuse pipeline run artifacts.
    #
    # Reuses artifacts from `run_1` to resolve inputs in `run_2`.
    # Run and verify result (1 + 5 == 6).
    #
    #                              (5)
    #                               v
    #                     1    -----------   6    -----------
    #  run_2:           ----> | AddNum 1 | ----> | Result 1 |
    #                         -----------        -----------
    #
    ############################################################################
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_2)
    self.assertResultEqual(pipeline_pb_run_2, [(result_1_v2.id, 6)])

    ############################################################################
    # PART 4: Now in the second sub-graph, modify `AddNum 2`.
    #
    #              (10)              (50)
    #               v                 v
    #           ---------        -----------        -----------
    #          | Load 2 | ----> | AddNum 2 | ----> | Result 2 |
    #          ---------        -----------        -----------
    #
    ############################################################################
    # pylint: disable=no-value-for-parameter
    add_num_2_v2 = AddNum(
        to_add=50, num=load_2.outputs['num']).with_id('add_num_2')
    load_2.remove_downstream_node(add_num_2)  # This line is important.
    result_2_v2 = Result(
        result=add_num_2_v2.outputs['added_num']).with_id('result_2')
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_3 = self.make_pipeline(
        components=[
            load_1,
            add_num_1_v2,
            result_1_v2,
            load_2,
            add_num_2_v2,
            result_2_v2,
        ],
        run_id='run_3')

    ############################################################################
    # PART 5a: Partial run -- DAG filtering.
    #
    # Only run the second sub-graph from `AddNum 2` onwards.
    #
    #                              (50)
    #                               v
    #                          -----------        -----------
    #  run_3:          --/--> | AddNum 2 | ----> | Result 2 |
    #                         -----------        -----------
    #
    ############################################################################
    partial_run_utils.mark_pipeline(
        pipeline_pb_run_3,
        from_nodes=lambda node_id: (node_id == add_num_2_v2.id))
    ############################################################################
    # PART 5b: Partial run -- Reuse pipeline run artifacts.
    #
    # Reuses artifacts from `run_2` to resolve inputs in `run_3`.
    #
    # Observe that even though `run_2` does not need to be associated `Load 2`,
    # `run_3` is able to obtain the outputs of `Load 2` from `run_2`.
    # This is done on purpose to make partial runs more user-friendly.
    # Run and verify result (10 + 50 == 60).
    #
    #                              (50)
    #                               v
    #                     10   -----------   60   -----------
    #  run_3:           ----> | AddNum 2 | ----> | Result 2 |
    #                         -----------        -----------
    #
    ############################################################################
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_3)
    self.assertResultEqual(pipeline_pb_run_3, [(result_2_v2.id, 60)])

    ############################################################################
    # PART 6a: Partial run -- Two sub-graphs at the same time.
    #
    #                              (5)
    #                               v
    #                          -----------        -----------
    #                  --/--> | AddNum 1 | ----> | Result 1 |
    #                         -----------        -----------
    #  run_4:
    #                          -----------        -----------
    #                  --/--> | AddNum 2 | ----> | Result 2 |
    #                         -----------        -----------
    #                              ^
    #                            (50)
    #
    ############################################################################
    pipeline_pb_run_4 = self.make_pipeline(
        components=[
            load_1,
            add_num_1_v2,
            result_1_v2,
            load_2,
            add_num_2_v2,
            result_2_v2,
        ],
        run_id='run_4')
    partial_run_utils.mark_pipeline(
        pipeline_pb_run_4, from_nodes=lambda node_id: ('add_num' in node_id))

    ############################################################################
    # PART 6b: Partial run -- Reuse pipeline run artifacts.
    #
    # Again, observe that even though `run_3` does not need to be associated
    # `Load 1`, `run_3` is able to obtain the outputs of `Load 2` from `run_3`.
    #
    # Run and verify.
    #
    #                              (5)
    #                               v
    #                     1    -----------   6    -----------
    #                   ----> | AddNum 1 | ----> | Result 1 |
    #                         -----------        -----------
    #  run_4:
    #                    10    -----------  60    -----------
    #                   ----> | AddNum 2 | ----> | Result 2 |
    #                         -----------        -----------
    #                              ^
    #                            (50)
    #
    ############################################################################
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_4)
    self.assertResultEqual(pipeline_pb_run_4, [(result_1_v2.id, 6),
                                               (result_2_v2.id, 60)])
    # Also verify that parent contexts are added.
    with metadata.Metadata(self.metadata_config) as m:
      pipeline_run_contexts = {
          run_context.name: run_context for run_context in
          m.store.get_contexts_by_type(constants.PIPELINE_RUN_CONTEXT_TYPE_NAME)
      }
      self.assertEqual(
          m.store.get_parent_contexts_by_context(
              pipeline_run_contexts['run_4'].id),
          [pipeline_run_contexts['run_3']])
      self.assertEqual(
          m.store.get_parent_contexts_by_context(
              pipeline_run_contexts['run_3'].id),
          [pipeline_run_contexts['run_2']])
      self.assertEqual(
          m.store.get_parent_contexts_by_context(
              pipeline_run_contexts['run_2'].id),
          [pipeline_run_contexts['run_1']])

  def testReusePipelineArtifacts_preventInconsistency(self):
    """Tests that a tricky sequence of partial runs raises an error."""
    ############################################################################
    # PART 1a: Full run -- Author pipeline.
    #
    # This pipeline contains `SubtractNums`, which has two input channels.
    # It substract the output of `Load` from the value of `AddNum` -- i.e.,
    # it should recover `AddNum`'s exec_param.
    #
    #             (1)            (1)
    #              v              v
    #           -------       ---------        ---------------        ---------
    #  run_1:  | Load |---.> | AddNum | ----> | SubtractNums | ----> | Result |
    #          -------   |   ---------        ---------------        ---------
    #                    \                        ^
    #                     \______________________/
    #
    ############################################################################
    # pylint: disable=no-value-for-parameter
    load = Load(start_num=1)
    add_num = AddNum(to_add=1, num=load.outputs['num'])
    subtract_nums = SubtractNums(
        num_1=add_num.outputs['added_num'], num_2=load.outputs['num'])
    result = Result(result=subtract_nums.outputs['diff'])
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_1 = self.make_pipeline(
        components=[load, add_num, subtract_nums, result], run_id='run_1')
    ############################################################################
    # PART 1b: Full run -- Run and verify ((1 + 1) - 1 == 1).
    #
    #             (1)            (1)
    #              v              v
    #           -------  1    ---------   2    ---------------   1    ---------
    #  run_1:  | Load |---.> | AddNum | ----> | SubtractNums | ----> | Result |
    #          -------   |   ---------        ---------------        ---------
    #                    \                        ^
    #                     \______________________/
    #
    ############################################################################
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_1)
    self.assertResultEqual(pipeline_pb_run_1, 1)

    ############################################################################
    # PART 2: Modify the `AddNum` component in the first "sub-pipeline".
    #
    #             (1)            (5)
    #              v              v
    #           -------       ---------        ---------------        ---------
    #          | Load |---.> | AddNum | ----> | SubtractNums | ----> | Result |
    #          -------   |   ---------        ---------------        ---------
    #                    \                        ^
    #                     \______________________/
    #
    ############################################################################
    # pylint: disable=no-value-for-parameter
    add_num_v2 = AddNum(to_add=5, num=load.outputs['num'])
    load.remove_downstream_node(add_num)  # This line is important.
    subtract_nums_v2 = SubtractNums(
        num_1=add_num_v2.outputs['added_num'], num_2=load.outputs['num'])
    load.remove_downstream_node(subtract_nums)  # This line is important.
    result_v2 = Result(result=subtract_nums_v2.outputs['diff'])
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_2 = self.make_pipeline(
        components=[load, add_num_v2, subtract_nums_v2, result_v2],
        run_id='run_2')

    ############################################################################
    # PART 3a: Partial run -- DAG filtering.
    #
    # Only run from `AddNum` onwards.
    #
    #                            (5)
    #                             v
    #                         ---------        ---------------        ---------
    #                 -/--.> | AddNum | ----> | SubtractNums | ----> | Result |
    #                    |   ---------        ---------------        ---------
    #                    \                        ^
    #                     \______________________/
    #
    ############################################################################
    partial_run_utils.mark_pipeline(
        pipeline_pb_run_2,
        from_nodes=lambda node_id: (node_id == add_num_v2.id))
    ############################################################################
    # PART 3b: Partial run -- Reuse pipeline run artifacts.
    #
    # Reuses artifacts from `run_1` to resolve inputs in `run_2`.
    # Run and verify result ((1 + 5) - 1 == 5).
    #
    #                            (5)
    #                             v
    #                    1    ---------   6    ---------------   5    ---------
    #  run_2:          ---.> | AddNum | ----> | SubtractNums | ----> | Result |
    #                    |   ---------        ---------------        ---------
    #                    \                        ^
    #                     \______________________/
    #
    ############################################################################
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_2)
    self.assertResultEqual(pipeline_pb_run_2, 5)

    ############################################################################
    # PART 4: Now modify `Load`.
    #
    #             (5)            (5)
    #              v              v
    #           -------       ---------        ---------------        ---------
    #          | Load |---.> | AddNum | ----> | SubtractNums | ----> | Result |
    #          -------   |   ---------        ---------------        ---------
    #                    \                        ^
    #                     \______________________/
    #
    ############################################################################
    # pylint: disable=no-value-for-parameter
    load_v2 = Load(start_num=5)
    add_num_v3 = AddNum(to_add=5, num=load_v2.outputs['num'])
    subtract_nums_v3 = SubtractNums(
        num_1=add_num_v3.outputs['added_num'], num_2=load_v2.outputs['num'])
    result_v3 = Result(result=subtract_nums_v3.outputs['diff'])
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_3 = self.make_pipeline(
        components=[load_v2, add_num_v3, subtract_nums_v3, result_v3],
        run_id='run_3')

    ############################################################################
    # PART 5: Partial run. Only run `Load`.
    #
    #             (5)
    #              v
    #           -------
    #  run_3:  | Load |---.>
    #          -------   |
    #                    \                        ^
    #                     \______________________/
    #
    ############################################################################
    partial_run_utils.mark_pipeline(
        pipeline_pb_run_3,
        from_nodes=lambda node_id: (node_id == load_v2.id),
        to_nodes=lambda node_id: (node_id == load_v2.id))
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_3)

    ############################################################################
    # PART 6a: Partial run -- Only `SubtractNum` onwards, using `run_3` as base.
    #
    #                                          ---------------        ---------
    #  run_4:                          --/--> | SubtractNums | ----> | Result |
    #                                         ---------------        ---------
    #                                             ^
    #                                             x
    #
    ############################################################################
    pipeline_pb_run_4 = self.make_pipeline(
        components=[load_v2, add_num_v3, subtract_nums_v3, result_v3],
        run_id='run_4')
    partial_run_utils.mark_pipeline(
        pipeline_pb_run_4,
        from_nodes=lambda node_id: (node_id == subtract_nums_v3.id))

    ############################################################################
    # PART 6b: Partial run -- Reuse pipeline run artifacts.
    #
    # Reusing artifacts with run_3 as base fails. This is by design, because
    # in run_3, we only execute `Load` but not `AddNum`. We do not want to reuse
    # the output of `AddNum` in run_2, because it was made using an old version
    # of `Load`. The reason we want to prevent this is that if you allowed this,
    # you could associate a ModelEvaluation with the wrong Model, and that would
    # be very bad, because you might deploy a bad Model while thinking it was a
    # good Model.
    #
    #                                          ---------------        ---------
    #                          (!!)    --/--> | SubtractNums | ----> | Result |
    #                                         ---------------        ---------
    #                                             ^
    #                                             x
    #
    ############################################################################
    with self.assertRaisesRegex(
        LookupError,
        'No previous successful executions found for node_id AddNum in '
        'pipeline_run run_3'):
      beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_4)
    ############################################################################
    # PART 6b: Partial run -- Reuse pipeline run artifacts.
    #
    # But you can retry with base_run_id='run_2', and it should work fine.
    #
    #                                     6    ---------------   5    ---------
    #  run_5:                   OK      ----> | SubtractNums | ----> | Result |
    #                                         ---------------        ---------
    #                                             ^
    #                                        1 --/
    #
    ############################################################################
    pipeline_pb_run_5 = self.make_pipeline(
        components=[load_v2, add_num_v3, subtract_nums_v3, result_v3],
        run_id='run_5')
    snapshot_settings = pipeline_pb2.SnapshotSettings()
    snapshot_settings.base_pipeline_run_strategy.base_run_id = 'run_2'
    partial_run_utils.mark_pipeline(
        pipeline_pb_run_5,
        from_nodes=lambda node_id: (node_id == subtract_nums_v3.id),
        snapshot_settings=snapshot_settings)
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_5)
    self.assertResultEqual(pipeline_pb_run_5, 5)

  def testNonExistentBaseRunId_lookupError(self):
    """Raise error if user provides non-existent base_run_id."""
    # pylint: disable=no-value-for-parameter
    load = Load(start_num=1)
    add_num = AddNum(to_add=1, num=load.outputs['num'])
    result = Result(result=add_num.outputs['added_num'])
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_1 = self.make_pipeline(
        components=[load, add_num, result], run_id='run_1')
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_1)

    pipeline_pb_run_2 = self.make_pipeline(
        components=[load, add_num, result], run_id='run_2')
    snapshot_settings = pipeline_pb2.SnapshotSettings()
    snapshot_settings.base_pipeline_run_strategy.base_run_id = 'non_existent_id'
    partial_run_utils.mark_pipeline(
        pipeline_pb_run_2,
        from_nodes=lambda node_id: (node_id == add_num.id),
        snapshot_settings=snapshot_settings)
    with self.assertRaisesRegex(LookupError,
                                'pipeline_run_id .* not found in MLMD.'):
      beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_2)

  def testNonExistentNodeId_lookupError(self):
    """Raise error if user provides non-existent pipeline_run_id or node_id."""
    # pylint: disable=no-value-for-parameter
    load = Load(start_num=1)
    add_num = AddNum(to_add=1, num=load.outputs['num'])
    result = Result(result=add_num.outputs['added_num'])
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_1 = self.make_pipeline(
        components=[load, add_num, result], run_id='run_1')
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_1)

    # pylint: disable=no-value-for-parameter
    load_v2 = Load(start_num=2).with_id('non_existent_id')
    add_num_v2 = AddNum(to_add=1, num=load_v2.outputs['num'])
    result_v2 = Result(result=add_num_v2.outputs['added_num'])
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_2 = self.make_pipeline(
        components=[load_v2, add_num_v2, result_v2], run_id='run_2')
    snapshot_settings = pipeline_pb2.SnapshotSettings()
    snapshot_settings.base_pipeline_run_strategy.base_run_id = 'non_existent_id'
    partial_run_utils.mark_pipeline(
        pipeline_pb_run_2,
        from_nodes=lambda node_id: (node_id == add_num_v2.id),
        snapshot_settings=snapshot_settings)
    with self.assertRaisesRegex(LookupError,
                                'node context .* not found in MLMD.'):
      beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_2)

  def testNoPreviousSuccessfulExecution_lookupError(self):
    """Raise error if user tries to reuse node w/o any successful Executions."""
    # pylint: disable=no-value-for-parameter
    load_fail = LoadFail(start_num=1)
    add_num = AddNum(to_add=1, num=load_fail.outputs['num'])
    result = Result(result=add_num.outputs['added_num'])
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_1 = self.make_pipeline(
        components=[load_fail, add_num, result], run_id='run_1')
    try:
      # Suppress exception here, since we expect this pipeline run to fail.
      beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_1)
    except Exception:  # pylint: disable=broad-except
      pass

    # pylint: disable=no-value-for-parameter
    add_num_v2 = AddNum(to_add=1, num=load_fail.outputs['num'])
    load_fail.remove_downstream_node(add_num)  # this line is important
    result_v2 = Result(result=add_num_v2.outputs['added_num'])
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_2 = self.make_pipeline(
        components=[load_fail, add_num_v2, result_v2], run_id='run_2')
    partial_run_utils.mark_pipeline(
        pipeline_pb_run_2,
        from_nodes=lambda node_id: (node_id == add_num_v2.id))
    with self.assertRaisesRegex(LookupError,
                                'No previous successful executions found'):
      beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_2)

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
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_1)
    self.assertResultEqual(pipeline_pb_run_1, 2)

    add_num_v2 = AddNum(to_add=5, num=load.outputs['num'])  # pylint: disable=no-value-for-parameter
    load.remove_downstream_node(add_num)  # This line is important.
    result_v2 = Result(result=add_num_v2.outputs['added_num'])
    pipeline_pb_run_2 = self.make_pipeline(
        components=[load, add_num_v2, result_v2], run_id='run_2')

    partial_run_utils.mark_pipeline(
        pipeline_pb_run_2,
        from_nodes=lambda node_id: (node_id == add_num_v2.id))

    # Simulate success in registering cache execution, but failure when
    # publishing -- e.g., job was pre-empted.
    with mock.patch.object(
        execution_publish_utils, 'publish_cached_execution',
        autospec=True) as mock_publish_cached_execution:
      mock_publish_cached_execution.side_effect = ConnectionResetError()
      try:
        with metadata.Metadata(self.metadata_config) as m:
          partial_run_utils.snapshot(m, pipeline_pb_run_2)
      except ConnectionResetError:
        pass
    # A retry attempt that succeeds.
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_2)
    self.assertResultEqual(pipeline_pb_run_2, 6)

    # Make sure that only one new cache execution is created,
    # despite the retry.
    with metadata.Metadata(self.metadata_config) as m:
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

  def testIdempotence_retryReusesPreviousSuccessfulCacheExecution(self):
    """Ensures idempotence.

    This test checks that after the first successful cache execution, following
    reuse_node_execution attempts will be no-op.
    """
    # pylint: disable=no-value-for-parameter
    load = Load(start_num=1)
    add_num = AddNum(to_add=1, num=load.outputs['num'])
    result = Result(result=add_num.outputs['added_num'])
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_1 = self.make_pipeline(
        components=[load, add_num, result], run_id='run_1')
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_1)
    self.assertResultEqual(pipeline_pb_run_1, 2)

    # pylint: disable=no-value-for-parameter
    add_num_v2 = AddNum(to_add=5, num=load.outputs['num'])
    load.remove_downstream_node(add_num)  # This line is important.
    result_v2 = Result(result=add_num_v2.outputs['added_num'])
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_2 = self.make_pipeline(
        components=[load, add_num_v2, result_v2], run_id='run_2')

    partial_run_utils.mark_pipeline(
        pipeline_pb_run_2,
        from_nodes=lambda node_id: (node_id == add_num_v2.id))

    # Simulate two successful attempts.
    for _ in [1, 2]:
      with metadata.Metadata(self.metadata_config) as m:
        partial_run_utils.snapshot(m, pipeline_pb_run_2)
    # Make sure that only one new cache execution is created,
    # despite the multiple attempts.
    with metadata.Metadata(self.metadata_config) as m:
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
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_2)
    self.assertResultEqual(pipeline_pb_run_2, 6)

  def testReusePipelineArtifacts_missingNewRunId_error(self):
    """If pipeline IR has no run id, and user does not provide it, fail."""
    ############################################################################
    # PART 1: Full run.
    #
    #             (1)             (1)
    #              v               v
    #           -------   1    ---------   2    ---------
    #  run_1:  | Load | ----> | AddNum | ----> | Result |
    #          -------        ---------        ---------
    #
    ############################################################################
    # pylint: disable=no-value-for-parameter
    load = Load(start_num=1)
    add_num = AddNum(to_add=1, num=load.outputs['num'])
    result = Result(result=add_num.outputs['added_num'])
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_1 = self.make_pipeline(
        components=[load, add_num, result], run_id='run_1')
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_1)
    self.assertResultEqual(pipeline_pb_run_1, 2)

    ############################################################################
    # PART 2: Modify the `AddNum` component in the first "sub-pipeline".
    #
    #             (1)             (5)
    #              v               v
    #           -------        ---------        ---------
    #          | Load | ----> | AddNum | ----> | Result |
    #          -------        ---------        ---------
    #
    ############################################################################
    # pylint: disable=no-value-for-parameter
    add_num_v2 = AddNum(to_add=5, num=load.outputs['num'])
    load.remove_downstream_node(add_num)  # This line is important.
    result_v2 = Result(result=add_num_v2.outputs['added_num'])
    # pylint: enable=no-value-for-parameter
    # NOTE: pipeline_pb_run_2's pipeline_run_id RuntimeParameter is still
    # unresolved.
    pipeline_pb_run_2 = self.make_pipeline(
        components=[load, add_num_v2, result_v2])

    ############################################################################
    # PART 3a: Partial run -- DAG filtering.
    #
    # Only run the first sub-graph from `AddNum` onwards.
    #
    #                             (5)
    #                              v
    #                          ---------        ---------
    #  ???             --/--> | AddNum | ----> | Result |
    #                         ---------        ---------
    #
    ############################################################################
    partial_run_utils.mark_pipeline(
        pipeline_pb_run_2,
        from_nodes=lambda node_id: (node_id == add_num_v2.id))

    # We simulate a *user-error* here: The full_pipeline's pipeline_run_id was
    # not resolved, and the user does not provide a value for new_run_id.
    with metadata.Metadata(self.metadata_config) as m:
      with self.assertRaisesRegex(ValueError,
                                  'Unable to infer new pipeline run id.'):
        partial_run_utils._reuse_pipeline_run_artifacts(
            m, pipeline_pb_run_2, base_run_id='run_1')

    # Check that once the user provides the new_run_id, it still works.
    with metadata.Metadata(self.metadata_config) as m:
      partial_run_utils._reuse_pipeline_run_artifacts(
          m, pipeline_pb_run_2, base_run_id='run_1', new_run_id='run_2')
    runtime_parameter_utils.substitute_runtime_parameter(
        pipeline_pb_run_2, {constants.PIPELINE_RUN_ID_PARAMETER_NAME: 'run_2'})
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_2)
    self.assertResultEqual(pipeline_pb_run_2, 6)

  def testReusePipelineArtifacts_inconsistentNewRunId_error(self):
    """If pipeline IR's run_id differs from user-provided run_id, fail."""
    ############################################################################
    # PART 1: Full run.
    #
    #             (1)             (1)
    #              v               v
    #           -------   1    ---------   2    ---------
    #  run_1:  | Load | ----> | AddNum | ----> | Result |
    #          -------        ---------        ---------
    #
    ############################################################################
    # pylint: disable=no-value-for-parameter
    load = Load(start_num=1)
    add_num = AddNum(to_add=1, num=load.outputs['num'])
    result = Result(result=add_num.outputs['added_num'])
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_1 = self.make_pipeline(
        components=[load, add_num, result], run_id='run_1')
    beam_dag_runner.BeamDagRunner().run_with_ir(pipeline_pb_run_1)
    self.assertResultEqual(pipeline_pb_run_1, 2)

    ############################################################################
    # PART 2: Modify the `AddNum` component in the first "sub-pipeline".
    #
    #             (1)             (5)
    #              v               v
    #           -------        ---------        ---------
    #          | Load | ----> | AddNum | ----> | Result |
    #          -------        ---------        ---------
    #
    ############################################################################
    # pylint: disable=no-value-for-parameter
    add_num_v2 = AddNum(to_add=5, num=load.outputs['num'])
    load.remove_downstream_node(add_num)  # This line is important.
    result_v2 = Result(result=add_num_v2.outputs['added_num'])
    # pylint: enable=no-value-for-parameter
    pipeline_pb_run_2 = self.make_pipeline(
        components=[load, add_num_v2, result_v2], run_id='run_2')

    ############################################################################
    # PART 3a: Partial run -- DAG filtering.
    #
    # Only run the first sub-graph from `AddNum` onwards.
    #
    #                             (5)
    #                              v
    #                          ---------        ---------
    #  run_2           --/--> | AddNum | ----> | Result |
    #  (or run_3?!)           ---------        ---------
    #
    ############################################################################
    partial_run_utils.mark_pipeline(
        pipeline_pb_run_2,
        from_nodes=lambda node_id: (node_id == add_num_v2.id))

    # We simulate a *user-error* here: The full_pipeline was resolved with
    # pipeline_run_id='run_2', but the user provides 'run_3'.
    with metadata.Metadata(self.metadata_config) as m:
      with self.assertRaisesRegex(ValueError,
                                  'Conflicting new pipeline run ids found.'):
        partial_run_utils._reuse_pipeline_run_artifacts(
            m, pipeline_pb_run_2, base_run_id='run_1',
            new_run_id='run_3')  # <-- user error here


if __name__ == '__main__':
  absltest.main()
