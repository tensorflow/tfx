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

import json
import os
from typing import Any, Dict, List, Optional, Text

import tensorflow as tf
from tfx import types
from tfx.dsl.compiler import compiler
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.components.common import resolver
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.proto.orchestration import execution_result_pb2
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec


class Resolver(resolver.Resolver):

  def override_output_dict(self, **output_dict):
    self._output_dict.clear()
    self._output_dict.update(output_dict)


class DummyArtifacts:

  class X(types.Artifact):
    TYPE_NAME = 'X'

  class Y(types.Artifact):
    TYPE_NAME = 'Y'


class DummyExecutor(base_executor.BaseExecutor):
  """Dummy Executor that does nothing."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]
         ) -> Optional[execution_result_pb2.ExecutorOutput]:
    pass


def create_dummy_component_type(name, inputs, outputs):
  spec_cls = type(f'{name}Spec', (ComponentSpec,), {
      'INPUTS': inputs,
      'OUTPUTS': outputs,
      'PARAMETERS': {},
  })
  component_cls_holder = []

  def component_init(self, **kwargs):
    spec_kwargs = {}
    for input_key in inputs:
      spec_kwargs[input_key] = kwargs.get(input_key)
    for output_key, channel_param in outputs.items():
      spec_kwargs[output_key] = types.Channel(type=channel_param.type)
    spec = spec_cls(**spec_kwargs)
    component_cls = component_cls_holder[0]
    super(component_cls, self).__init__(spec=spec)
    self._id = kwargs.get('instance_name')

  component_cls = type(name, (base_component.BaseComponent,), {
      'SPEC_CLASS': spec_cls,
      'EXECUTOR_SPEC': executor_spec.ExecutorClassSpec(DummyExecutor),
      '__init__': component_init,
  })
  component_cls_holder.append(component_cls)

  return component_cls


class DummyComponents:

  A = create_dummy_component_type(
      'A', inputs={}, outputs={
          'x': ChannelParameter(type=DummyArtifacts.X),
          'y': ChannelParameter(type=DummyArtifacts.Y),
      })

  B = create_dummy_component_type(
      'B', inputs={
          'x': ChannelParameter(type=DummyArtifacts.X, optional=True),
          'y': ChannelParameter(type=DummyArtifacts.Y, optional=True),
      }, outputs={
          'xx': ChannelParameter(type=DummyArtifacts.X),
          'yy': ChannelParameter(type=DummyArtifacts.Y),
      })


class DummyResolverStrategy(resolver.ResolverStrategy):
  """Dummy resolver class that does nothing."""

  def __init__(self, **unused_kwargs):
    super().__init__()

  def resolve_artifacts(self, metadata_handler, input_dict):
    return input_dict


class TestCase(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    temp_dir = self.get_temp_dir()
    self.pipeline_root = os.path.join(temp_dir, 'pipeline')
    self.metadata_conn_config = metadata.sqlite_metadata_connection_config(
        os.path.join(temp_dir, 'metadata', 'metadata.db'))
    self.compiler = compiler.Compiler()

  def compile_async_pipeline(self, components):
    p = pipeline.Pipeline(
        'TestPipeline',
        pipeline_root=self.pipeline_root,
        metadata_connection_config=self.metadata_conn_config,
        execution_mode=pipeline.ExecutionMode.ASYNC,
        components=components,
    )
    return self.compiler.compile(p)

  def extract_pipeline_node(self, pipeline_ir, node_id):
    for node in pipeline_ir.nodes:
      if node.WhichOneof('node') == 'pipeline_node':
        pipeline_node = node.pipeline_node
        if pipeline_node.node_info.id == node_id:
          return pipeline_node
    return None


class CompilerResolverTest(TestCase):

  def test_resolver_node_is_not_in_ir(self):
    a = DummyComponents.A()
    r = Resolver(strategy_class=DummyResolverStrategy, x=a.outputs['x'])
    b = DummyComponents.B(x=r.outputs['x'])
    pipeline_ir = self.compile_async_pipeline([a, r, b])

    node_ids = {node.pipeline_node.node_info.id for node in pipeline_ir.nodes}
    self.assertCountEqual(node_ids, {'A', 'B'})

  def test_input_channel_skips_resolver_node(self):
    a = DummyComponents.A()
    r = Resolver(strategy_class=DummyResolverStrategy, x=a.outputs['x'])
    b = DummyComponents.B(x=r.outputs['x'])
    pipeline_ir = self.compile_async_pipeline([a, r, b])

    b_ir = self.extract_pipeline_node(pipeline_ir, 'B')
    self.assertIn('x', b_ir.inputs.inputs)
    self.assertEqual(
        b_ir.inputs.inputs['x'].channels[0].producer_node_query.id, 'A')

  def test_resolver_config_is_added(self):
    a = DummyComponents.A()
    r = Resolver(strategy_class=DummyResolverStrategy, x=a.outputs['x'])
    b = DummyComponents.B(x=r.outputs['x'])
    pipeline_ir = self.compile_async_pipeline([a, r, b])

    b_ir = self.extract_pipeline_node(pipeline_ir, 'B')
    resolver_steps = b_ir.inputs.resolver_config.resolver_steps
    self.assertLen(resolver_steps, 1)
    self.assertEndsWith(resolver_steps[0].class_path, '.DummyResolverStrategy')
    self.assertEqual(resolver_steps[0].input_keys, ['x'])

  def test_resolver_input_key_and_downstream_input_key_should_be_same(self):
    a = DummyComponents.A()
    r = Resolver(strategy_class=DummyResolverStrategy, alt_x=a.outputs['x'])
    b = DummyComponents.B(x=r.outputs['alt_x'])

    with self.assertRaisesRegex(ValueError, r'Downstream node input key \(x\) '
                                'should be the same as the output key '
                                r'\(alt_x\) of the resolver node.'):
      self.compile_async_pipeline([a, r, b])

  def test_multichannel_resolver(self):
    a = DummyComponents.A()
    r = Resolver(strategy_class=DummyResolverStrategy,
                 x=a.outputs['x'],
                 y=a.outputs['y'])
    b = DummyComponents.B(x=r.outputs['x'],
                          y=r.outputs['y'])
    pipeline_ir = self.compile_async_pipeline([a, r, b])

    b_ir = self.extract_pipeline_node(pipeline_ir, 'B')
    self.assertIn('x', b_ir.inputs.inputs)
    self.assertIn('y', b_ir.inputs.inputs)
    resolver_steps = b_ir.inputs.resolver_config.resolver_steps
    self.assertLen(resolver_steps, 1)
    self.assertEndsWith(resolver_steps[0].class_path, '.DummyResolverStrategy')
    self.assertCountEqual(resolver_steps[0].input_keys, ['x', 'y'])

  def test_skip_connection(self):
    a = DummyComponents.A()
    r = Resolver(strategy_class=DummyResolverStrategy, x=a.outputs['x'])
    b = DummyComponents.B(x=r.outputs['x'], y=a.outputs['y'])
    pipeline_ir = self.compile_async_pipeline([a, r, b])

    b_ir = self.extract_pipeline_node(pipeline_ir, 'B')
    self.assertIn('x', b_ir.inputs.inputs)
    self.assertIn('y', b_ir.inputs.inputs)
    resolver_steps = b_ir.inputs.resolver_config.resolver_steps
    self.assertLen(resolver_steps, 1)
    self.assertEndsWith(resolver_steps[0].class_path, '.DummyResolverStrategy')
    self.assertCountEqual(resolver_steps[0].input_keys, ['x'])

  def test_duplicated_key_error_if_different_channel(self):
    a1 = DummyComponents.A().with_id('A1')
    a2 = DummyComponents.A().with_id('A2')
    r = Resolver(strategy_class=DummyResolverStrategy,
                 x=a1.outputs['x'],
                 y=a1.outputs['y'])
    b = DummyComponents.B(x=r.outputs['x'],
                          y=a2.outputs['y'])

    # Same input key "y" is used with different input channels
    # a1.outputs['y'] and a2.outputs['y'].
    with self.assertRaisesRegex(ValueError, 'Duplicated input key y'):
      self.compile_async_pipeline([a1, a2, r, b])

  def test_multiple_upstream_nodes(self):
    a1 = DummyComponents.A().with_id('A1')
    a2 = DummyComponents.A().with_id('A2')
    r = Resolver(strategy_class=DummyResolverStrategy,
                 x=a1.outputs['x'],
                 y=a2.outputs['y'])
    b = DummyComponents.B(x=r.outputs['x'],
                          y=r.outputs['y'])
    pipeline_ir = self.compile_async_pipeline([a1, a2, r, b])

    b_ir = self.extract_pipeline_node(pipeline_ir, 'B')
    self.assertEqual(
        b_ir.inputs.inputs['x'].channels[0].producer_node_query.id, 'A1')
    self.assertEqual(
        b_ir.inputs.inputs['y'].channels[0].producer_node_query.id, 'A2')

  def test_multiple_downstream_nodes(self):
    # resolver node can be shared across multiple downstream nodes with their
    # own resolver config.
    a = DummyComponents.A()
    r = Resolver(strategy_class=DummyResolverStrategy,
                 x=a.outputs['x'],
                 y=a.outputs['y'])
    b1 = DummyComponents.B(x=r.outputs['x']).with_id('B1')
    b2 = DummyComponents.B(y=r.outputs['y']).with_id('B2')
    pipeline_ir = self.compile_async_pipeline([a, r, b1, b2])

    b1_ir = self.extract_pipeline_node(pipeline_ir, 'B1')
    b2_ir = self.extract_pipeline_node(pipeline_ir, 'B2')
    b1_steps = b1_ir.inputs.resolver_config.resolver_steps
    self.assertLen(b1_steps, 1)
    self.assertEndsWith(b1_steps[0].class_path, '.DummyResolverStrategy')
    self.assertEqual(b1_steps[0].input_keys, ['x', 'y'])
    b2_steps = b2_ir.inputs.resolver_config.resolver_steps
    self.assertLen(b2_steps, 1)
    self.assertEndsWith(b2_steps[0].class_path, '.DummyResolverStrategy')
    self.assertEqual(b2_steps[0].input_keys, ['x', 'y'])

  def test_sequential_resolver_nodes(self):
    a = DummyComponents.A()
    r1 = Resolver(strategy_class=DummyResolverStrategy,
                  config={'iam': 'r1'},
                  x=a.outputs['x']).with_id('R1')
    r2 = Resolver(strategy_class=DummyResolverStrategy,
                  config={'iam': 'r2'},
                  x=r1.outputs['x']).with_id('R2')
    b = DummyComponents.B(x=r2.outputs['x'])
    pipeline_ir = self.compile_async_pipeline([a, r1, r2, b])

    b_ir = self.extract_pipeline_node(pipeline_ir, 'B')
    self.assertLen(b_ir.inputs.inputs, 1)
    self.assertIn('x', b_ir.inputs.inputs)
    self.assertEqual(
        b_ir.inputs.inputs['x'].channels[0].producer_node_query.id, 'A')

    resolver_steps = b_ir.inputs.resolver_config.resolver_steps
    self.assertLen(resolver_steps, 2)
    # R1 resolver.
    self.assertEndsWith(resolver_steps[0].class_path, '.DummyResolverStrategy')
    self.assertEqual(json.loads(resolver_steps[0].config_json), {'iam': 'r1'})
    self.assertEqual(resolver_steps[0].input_keys, ['x'])
    # R2 resolver.
    self.assertEndsWith(resolver_steps[1].class_path, '.DummyResolverStrategy')
    self.assertEqual(json.loads(resolver_steps[1].config_json), {'iam': 'r2'})
    self.assertEqual(resolver_steps[1].input_keys, ['x'])

  def test_sequential_resolver_nodes_with_skip_connection(self):
    a = DummyComponents.A()
    r1 = Resolver(strategy_class=DummyResolverStrategy,
                  config={'iam': 'r1'},
                  x=a.outputs['x']).with_id('R1')
    r2 = Resolver(strategy_class=DummyResolverStrategy,
                  config={'iam': 'r2'},
                  x=r1.outputs['x'],
                  y=a.outputs['y']).with_id('R2')
    b = DummyComponents.B(x=r1.outputs['x'],
                          y=r2.outputs['y'])
    pipeline_ir = self.compile_async_pipeline([a, r1, r2, b])

    b_ir = self.extract_pipeline_node(pipeline_ir, 'B')
    self.assertLen(b_ir.inputs.inputs, 2)
    self.assertIn('x', b_ir.inputs.inputs)
    self.assertEqual(
        b_ir.inputs.inputs['x'].channels[0].producer_node_query.id, 'A')
    self.assertIn('y', b_ir.inputs.inputs)
    self.assertEqual(
        b_ir.inputs.inputs['y'].channels[0].producer_node_query.id, 'A')

    resolver_steps = b_ir.inputs.resolver_config.resolver_steps
    self.assertLen(resolver_steps, 2)
    # R1 resolver.
    self.assertEndsWith(resolver_steps[0].class_path, '.DummyResolverStrategy')
    self.assertEqual(json.loads(resolver_steps[0].config_json), {'iam': 'r1'})
    self.assertEqual(resolver_steps[0].input_keys, ['x'])
    # R2 resolver.
    self.assertEndsWith(resolver_steps[1].class_path, '.DummyResolverStrategy')
    self.assertEqual(json.loads(resolver_steps[1].config_json), {'iam': 'r2'})
    self.assertEqual(resolver_steps[1].input_keys, ['x', 'y'])

  def test_parallel_resolver_nodes(self):
    a = DummyComponents.A()
    r1 = Resolver(strategy_class=DummyResolverStrategy,
                  x=a.outputs['x']).with_id('R1')
    r2 = Resolver(strategy_class=DummyResolverStrategy,
                  y=a.outputs['y']).with_id('R2')
    b = DummyComponents.B(x=r1.outputs['x'], y=r2.outputs['y'])
    pipeline_ir = self.compile_async_pipeline([a, r1, r2, b])

    b_ir = self.extract_pipeline_node(pipeline_ir, 'B')
    self.assertLen(b_ir.inputs.inputs, 2)
    self.assertIn('x', b_ir.inputs.inputs)
    self.assertIn('y', b_ir.inputs.inputs)
    self.assertEqual(
        b_ir.inputs.inputs['x'].channels[0].producer_node_query.id, 'A')
    self.assertEqual(
        b_ir.inputs.inputs['y'].channels[0].producer_node_query.id, 'A')


if __name__ == '__main__':
  tf.test.main()
