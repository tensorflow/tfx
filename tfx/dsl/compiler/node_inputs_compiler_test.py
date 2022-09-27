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
"""Tests for tfx.dsl.compiler.node_inputs_compiler."""

from typing import List, Type

import tensorflow as tf
from tfx import types
from tfx.dsl.compiler import compiler_context
from tfx.dsl.compiler import node_inputs_compiler
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_node
from tfx.dsl.components.base import executor_spec
from tfx.dsl.control_flow import for_each
from tfx.dsl.experimental.conditionals import conditional
from tfx.dsl.input_resolution import resolver_function
from tfx.dsl.input_resolution import resolver_op
from tfx.orchestration import pipeline
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import channel as channel_types
from tfx.types import component_spec
from tfx.types import standard_artifacts

from google.protobuf import text_format


class DummyArtifact(types.Artifact):
  TYPE_NAME = 'Dummy'
  PROPERTIES = {}


class DummyNode(base_node.BaseNode):

  def __init__(self, id: str, inputs=None, exec_properties=None):  # pylint: disable=redefined-builtin
    super().__init__()
    self.with_id(id)
    self._inputs = inputs or {}
    self._exec_properties = exec_properties or {}
    self._outputs = {}

  def output(self, key: str, artifact_type=DummyArtifact):
    if key not in self._outputs:
      self._outputs[key] = channel_types.OutputChannel(artifact_type, self, key)
    return self._outputs[key]

  @property
  def inputs(self) -> ...:
    return self._inputs

  @property
  def exec_properties(self) -> ...:
    return self._exec_properties

  @property
  def outputs(self) -> ...:
    return self._outputs


class DummyArtifactList(
    resolver_op.ResolverOp,
    canonical_name='testing.DummyArtifactList',
    arg_data_types=(),
    return_data_type=resolver_op.DataType.ARTIFACT_LIST):

  def apply(self):
    return []


class DummyDict(
    resolver_op.ResolverOp,
    canonical_name='testing.DummyDict',
    arg_data_types=(),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP):

  def apply(self):
    return {'x': []}


class DummyDictList(
    resolver_op.ResolverOp,
    canonical_name='testing.DummyDictList',
    arg_data_types=(),
    return_data_type=resolver_op.DataType.ARTIFACT_MULTIMAP_LIST):

  def apply(self):
    return []


@resolver_function.resolver_function
def dummy_artifact_list():
  return DummyArtifactList()


@resolver_function.resolver_function
def dummy_dict():
  return DummyDict()


@resolver_function.resolver_function
def dummy_dict_list():
  return DummyDictList()


class NodeInputsCompilerTest(tf.test.TestCase):
  pipeline_name = 'dummy-pipeline'

  def _prepare_pipeline(
      self, components: List[base_node.BaseNode]) -> pipeline.Pipeline:
    return pipeline.Pipeline(
        pipeline_name=self.pipeline_name,
        components=components)

  def _compile_node_inputs(
      self, node, components=None) -> pipeline_pb2.NodeInputs:
    if not components:
      components = [node]
    p = self._prepare_pipeline(components)
    ctx = compiler_context.PipelineContext(p)
    result = pipeline_pb2.NodeInputs()
    node_inputs_compiler.compile_node_inputs(ctx, node, result)
    return result

  def _get_channel_pb(
      self, artifact_type: Type[types.Artifact] = DummyArtifact,
      pipeline_name: str = '', node_id: str = '',
      output_key: str = '') -> pipeline_pb2.InputSpec.Channel:
    result = pipeline_pb2.InputSpec.Channel()
    node_inputs_compiler._compile_channel_pb(
        artifact_type=artifact_type,
        pipeline_name=pipeline_name or self.pipeline_name,
        node_id=node_id,
        output_key=output_key,
        result=result)
    return result

  def testCompileAlreadyCompiledInputs(self):
    producer = DummyNode('MyProducer')
    consumer = DummyNode('MyConsumer', inputs={'x': producer.output('x')})

    p = self._prepare_pipeline([producer, consumer])
    ctx = compiler_context.PipelineContext(p)

    fake_channel_pb = text_format.Parse("""
      context_queries {
        type {
          name: "Foo"
        }
        name {
          field_value {
            string_value: "foo"
          }
        }
      }
      artifact_query {
        type {
          name: "Dummy"
        }
      }
      output_key: "x"
    """, pipeline_pb2.InputSpec.Channel())

    ctx.channels[producer.output('x')] = fake_channel_pb
    result = pipeline_pb2.NodeInputs()
    node_inputs_compiler.compile_node_inputs(ctx, consumer, result)

    self.assertProtoEquals(
        result.inputs['x'].channels[0],
        fake_channel_pb)

  def testCompileChannel(self):
    channel = channel_types.Channel(
        type=DummyArtifact,
        producer_component_id='MyProducer',
        output_key='my_output_key')
    node = DummyNode('MyNode', inputs={'x': channel})

    result = self._compile_node_inputs(node)

    self.assertLen(result.inputs['x'].channels, 1)
    self.assertProtoEquals(
        result.inputs['x'].channels[0],
        self._get_channel_pb(
            node_id='MyProducer', output_key='my_output_key'))

  def testCompileExternalPipelineOutputChannel(self):
    a = DummyNode('A')
    p1 = pipeline.Pipeline(
        pipeline_name='p1',
        components=[a],
        outputs={'x': a.output('x')})
    p2_inputs = pipeline.PipelineInputs({'x': p1.outputs['x']})
    p2 = pipeline.Pipeline(pipeline_name='p2', inputs=p2_inputs)

    ctx = compiler_context.PipelineContext(p2)
    result = pipeline_pb2.NodeInputs()
    node_inputs_compiler.compile_node_inputs(ctx, p2, result)

    self.assertLen(result.inputs, 1)
    self.assertLen(result.inputs['x'].channels, 1)
    self.assertProtoEquals(
        result.inputs['x'].channels[0],
        self._get_channel_pb(pipeline_name='p1', node_id='A', output_key='x'))

  def testCompileUnionChannel(self):
    producer = DummyNode('MyProducer')
    consumer = DummyNode('MyConsumer', inputs={
        'x': channel_types.union([
            producer.output('x'),
            producer.output('y'),
            channel_types.Channel(
                type=DummyArtifact,
                producer_component_id='Z',
                output_key='z'),
        ])
    })

    result = self._compile_node_inputs(
        consumer, components=[producer, consumer])

    self.assertLen(result.inputs, 4)
    self.assertEqual(
        result.inputs['x'].mixed_inputs.method,
        pipeline_pb2.InputSpec.Mixed.Method.UNION)
    dep_input_keys = list(result.inputs['x'].mixed_inputs.input_keys)
    self.assertLen(dep_input_keys, 3)

  def testCompileLoopVarChannel(self):
    producer = DummyNode('MyProducer')
    with for_each.ForEach(producer.output('xs')) as x:
      consumer = DummyNode('MyConsumer', inputs={'x': x})

    result = self._compile_node_inputs(
        consumer, components=[producer, consumer])

    self.assertLen(result.inputs, 2)
    other_input_key = [
        input_key for input_key in result.inputs
        if input_key != 'x'][0]
    graph_id = result.inputs['x'].input_graph_ref.graph_id
    self.assertNotEmpty(graph_id)
    self.assertLen(result.input_graphs, 1)
    for graph_node in result.input_graphs[graph_id].nodes.values():
      if graph_node.WhichOneof('kind') == 'op_node':
        self.assertEqual(graph_node.op_node.op_type, 'tfx.internal.Unnest')
      elif graph_node.WhichOneof('kind') == 'input_node':
        self.assertEqual(
            graph_node.input_node.input_key, other_input_key)

  def testCompileInputGraph(self):
    channel = dummy_artifact_list.with_output_type(DummyArtifact)()
    node = DummyNode('MyNode', inputs={'x': channel})
    p = self._prepare_pipeline([node])
    ctx = compiler_context.PipelineContext(p)
    result = pipeline_pb2.NodeInputs()

    with self.subTest('First compilation'):
      input_graph_id = node_inputs_compiler._compile_input_graph(
          ctx, node, channel, result)
      self.assertLen(result.input_graphs, 1)
      self.assertProtoEquals("""
        nodes {
          key: "op_1"
          value: {
            output_data_type: ARTIFACT_LIST
            op_node {
              op_type: "testing.DummyArtifactList"
            }
          }
        }
        result_node: "op_1"
      """, result.input_graphs[input_graph_id])

    with self.subTest('Second compilation'):
      second_input_graph_id = node_inputs_compiler._compile_input_graph(
          ctx, node, channel, result)
      self.assertEqual(input_graph_id, second_input_graph_id)

  def testCompileInputGraphRef(self):
    x1 = dummy_artifact_list.with_output_type(DummyArtifact)()
    x2 = dummy_dict.with_output_type({'x': DummyArtifact})()['x']
    dict_list = dummy_dict_list.with_output_type({'x': DummyArtifact})()
    with for_each.ForEach(dict_list) as each_dict:
      x3 = each_dict['x']
      node = DummyNode('MyNode', inputs={'x1': x1, 'x2': x2, 'x3': x3})

    result = self._compile_node_inputs(node)

    self.assertNotEmpty(result.inputs['x1'].input_graph_ref.graph_id)
    self.assertEmpty(result.inputs['x1'].input_graph_ref.key)
    self.assertNotEmpty(result.inputs['x2'].input_graph_ref.graph_id)
    self.assertEqual(result.inputs['x2'].input_graph_ref.key, 'x')
    self.assertNotEmpty(result.inputs['x3'].input_graph_ref.graph_id)
    self.assertEqual(result.inputs['x3'].input_graph_ref.key, 'x')

  def testCompileConditionals(self):
    cond_node = DummyNode('CondNode')
    with conditional.Cond(
        cond_node.output('x').future().value == 42):
      inner_node = DummyNode('InnerNode')

    result = self._compile_node_inputs(
        inner_node, components=[cond_node, inner_node])

    self.assertLen(result.inputs, 1)
    cond_input_key = list(result.inputs)[0]
    self.assertFalse(result.inputs[cond_input_key].hidden)
    self.assertEqual(result.inputs[cond_input_key].min_count, 1)
    self.assertLen(result.conditionals, 1)
    cond = list(result.conditionals.values())[0]
    self.assertProtoEquals("""
      operator {
        compare_op {
          op: EQUAL
          rhs {
            value {
              int_value: 42
            }
          }
          lhs {
            operator {
              artifact_value_op {
                expression {
                  operator {
                    index_op {
                      expression {
                        placeholder {
                          key: "%s"
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    """ % cond_input_key, cond.placeholder_expression)

  def testCompileInputsForDynamicProperties(self):
    producer = DummyNode('Producer')
    consumer = DummyNode('Consumer', exec_properties={
        'x': producer.output('x', standard_artifacts.Integer).future().value
    })

    result = self._compile_node_inputs(
        consumer, components=[producer, consumer])

    self.assertLen(result.inputs, 1)
    dynamic_prop_input_key = list(result.inputs)[0]
    self.assertFalse(result.inputs[dynamic_prop_input_key].hidden)
    self.assertEqual(result.inputs[dynamic_prop_input_key].min_count, 1)

  def testCompileMinCount(self):

    class DummyComponentSpec(component_spec.ComponentSpec):
      INPUTS = {
          'required': component_spec.ChannelParameter(
              DummyArtifact, optional=False),
          'optional_but_not_allow_empty': component_spec.ChannelParameter(
              DummyArtifact, optional=True, allow_empty=False),
          'optional_and_allow_empty': component_spec.ChannelParameter(
              DummyArtifact, optional=True, allow_empty=True),
      }
      OUTPUTS = {}
      PARAMETERS = {}

    class DummyComponent(base_component.BaseComponent):
      SPEC_CLASS = DummyComponentSpec
      EXECUTOR_SPEC = executor_spec.ExecutorSpec()

      def __init__(self, **inputs):
        super().__init__(DummyComponentSpec(**inputs))

    producer = DummyNode('Producer')
    c1 = DummyComponent(
        required=producer.output('x'),
    ).with_id('Consumer1')
    c2 = DummyComponent(
        required=producer.output('x'),
        optional_but_not_allow_empty=producer.output('x'),
    ).with_id('Consumer2')
    c3 = DummyComponent(
        required=producer.output('x'),
        optional_and_allow_empty=producer.output('x'),
    ).with_id('Consumer3')

    p = self._prepare_pipeline([producer, c1, c2, c3])
    ctx = compiler_context.PipelineContext(p)

    r1 = pipeline_pb2.NodeInputs()
    node_inputs_compiler.compile_node_inputs(ctx, c1, r1)
    r2 = pipeline_pb2.NodeInputs()
    node_inputs_compiler.compile_node_inputs(ctx, c2, r2)
    r3 = pipeline_pb2.NodeInputs()
    node_inputs_compiler.compile_node_inputs(ctx, c3, r3)

    self.assertEqual(r1.inputs['required'].min_count, 1)
    self.assertEqual(r2.inputs['optional_but_not_allow_empty'].min_count, 1)
    self.assertEqual(r2.inputs['optional_and_allow_empty'].min_count, 0)


if __name__ == '__main__':
  tf.test.main()
