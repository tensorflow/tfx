# Lint as: python2, python3
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
"""Tests for tfx.orchestration.experminental.optimizations.beam_fusion"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import os
import tempfile
from typing import Any, Dict, Text, Type

import tensorflow as tf

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base import base_node
from tfx.components.base import executor_spec
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.types import node_common
from tfx.types.component_spec import ChannelParameter
from tfx.orchestration.experimental.optimizations import beam_fusion


class _OutputArtifact(types.Artifact):
  TYPE_NAME = 'OutputArtifact'


def _make_fake_node_instance(name: Text):

  class _FakeNode(base_node.BaseNode):

    @property
    def inputs(self) -> node_common._PropertyDictWrapper:  # pylint: disable=protected-access
      return node_common._PropertyDictWrapper({})  # pylint: disable=protected-access

    @property
    def outputs(self) -> node_common._PropertyDictWrapper:  # pylint: disable=protected-access
      return node_common._PropertyDictWrapper({})  # pylint: disable=protected-access

    @property
    def exec_properties(self) -> Dict[Text, Any]:
      return {}

  return _FakeNode(instance_name=name)

def _make_fake_component_instance(name: Text, output_type: Type[types.Artifact],
                                  inputs: Dict[Text, types.Channel],
                                  outputs: Dict[Text, types.Channel],
                                  use_fuseable_beam_executor: bool):

  class _FakeComponentSpec(types.ComponentSpec):
    PARAMETERS = {}
    INPUTS = dict([(arg, ChannelParameter(type=channel.type)) # pylint: disable=consider-using-dict-comprehension
                   for arg, channel in inputs.items()])
    OUTPUTS = dict([(arg, ChannelParameter(type=channel.type))
                    for arg, channel in outputs.items()] +
                   [('output', ChannelParameter(type=output_type))])

  class _FakeComponent(base_component.BaseComponent):

    SPEC_CLASS = _FakeComponentSpec

    if use_fuseable_beam_executor:
      EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
          base_executor.FuseableBeamExecutor)
    else:
      EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
          base_executor.BaseExecutor)

    def __init__(
        self,
        type: Type[types.Artifact],  # pylint: disable=redefined-builtin
        spec_kwargs: Dict[Text, Any]):
      spec = _FakeComponentSpec(output=types.Channel(type=type), **spec_kwargs)
      super(_FakeComponent, self).__init__(spec=spec, instance_name=name)

  spec_kwargs = dict(itertools.chain(inputs.items(), outputs.items()))
  return _FakeComponent(output_type, spec_kwargs)

class _OutputTypeA(types.Artifact):
  TYPE_NAME = 'OutputTypeA'


class _OutputTypeB(types.Artifact):
  TYPE_NAME = 'OutputTypeB'


class _OutputTypeC(types.Artifact):
  TYPE_NAME = 'OutputTypeC'


class _OutputTypeD(types.Artifact):
  TYPE_NAME = 'OutputTypeD'


class _OutputTypeE(types.Artifact):
  TYPE_NAME = 'OutputTypeE'


class _OutputTypeF(types.Artifact):
  TYPE_NAME = 'OutputTypeE'


class _OutputTypeG(types.Artifact):
  TYPE_NAME = 'OutputTypeE'


class PipelineTest(tf.test.TestCase):

  def setUp(self):
    super(PipelineTest, self).setUp()
    tmp_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir())
    self._tmp_file = os.path.join(tmp_dir, self._testMethodName,
                                  tempfile.mkstemp(prefix='cli_tmp_')[1])
    self._tmp_dir = os.path.join(tmp_dir, self._testMethodName,
                                 tempfile.mkdtemp(prefix='cli_tmp_')[1])
    self._original_tmp_value = os.environ.get(
        'TFX_JSON_EXPORT_PIPELINE_ARGS_PATH', '')
    self._metadata_connection_config = (
        metadata.sqlite_metadata_connection_config(
            os.path.join(self._tmp_dir, 'metadata')))

  def tearDown(self):
    super(PipelineTest, self).tearDown()
    os.environ['TFX_TMP_DIR'] = self._original_tmp_value

  def testPipelineGetFuseableSubgraphs(self):
    component_a = _make_fake_component_instance(
        'component_a', _OutputTypeA, {}, {}, use_fuseable_beam_executor=False)
    component_b = _make_fake_component_instance(
        'component_b', _OutputTypeB, {'a': component_a.outputs['output']}, {},
        use_fuseable_beam_executor=True)
    component_c = _make_fake_component_instance(
        'component_c', _OutputTypeC, {'a': component_a.outputs['output']}, {},
        use_fuseable_beam_executor=True)
    component_d = _make_fake_component_instance(
        'component_d', _OutputTypeD, {'b': component_b.outputs['output']}, {},
        use_fuseable_beam_executor=True)
    component_e = _make_fake_component_instance(
        'component_e', _OutputTypeE, {
            'a': component_a.outputs['output'],
            'd': component_d.outputs['output']
        }, {}, use_fuseable_beam_executor=True)
    component_f = _make_fake_component_instance(
        'component_f', _OutputTypeF, {'c': component_c.outputs['output']}, {},
        use_fuseable_beam_executor=True)
    component_g = _make_fake_component_instance(
        'component_g', _OutputTypeG, {'c': component_c.outputs['output']}, {},
        use_fuseable_beam_executor=True)

    my_pipeline = pipeline.Pipeline(
        pipeline_name='my_pipeline',
        pipeline_root='root',
        components=[
            component_d, component_c, component_g, component_a, component_b,
            component_e, component_f
        ],
        enable_cache=True,
        metadata_connection_config=self._metadata_connection_config,
        beam_pipeline_args=['--runner=PortableRunner'],
        additional_pipeline_args={})

    optimizer = beam_fusion.BeamFusionOptimizer(my_pipeline)

    actual_subgraphs = optimizer.get_fuseable_subgraphs()
    actual_sources = []
    actual_sinks = []
    for subgraph in actual_subgraphs:
      actual_sources.append(optimizer.get_subgraph_sources(subgraph))
      actual_sinks.append(optimizer.get_subgraph_sinks(subgraph))

    expected_subgraphs = [[component_b, component_d, component_e],
                          [component_c, component_f, component_g]]
    expected_sources = [{component_b}, {component_c}]
    expected_sinks = [{component_e}, {component_f, component_g}]

    self.assertEqual(actual_subgraphs, expected_subgraphs)
    self.assertEqual(expected_sources, actual_sources)
    self.assertEqual(expected_sinks, actual_sinks)

  def testPipelineGetFuseableSubgraphsNoBeamComponents(self):
    component_a = _make_fake_component_instance(
        'component_a', _OutputTypeA, {}, {}, use_fuseable_beam_executor=False)
    component_b = _make_fake_component_instance(
        'component_b', _OutputTypeB, {'a': component_a.outputs['output']}, {},
        use_fuseable_beam_executor=False)
    component_c = _make_fake_component_instance(
        'component_c', _OutputTypeC, {'a': component_a.outputs['output']}, {},
        use_fuseable_beam_executor=False)

    my_pipeline = pipeline.Pipeline(
        pipeline_name='my_pipeline',
        pipeline_root='root',
        components=[component_b, component_a, component_c],
        enable_cache=True,
        metadata_connection_config=self._metadata_connection_config,
        beam_pipeline_args=['--runner=PortableRunner'],
        additional_pipeline_args={})

    optimizer = beam_fusion.BeamFusionOptimizer(my_pipeline)

    actual_subgraphs = optimizer.get_fuseable_subgraphs()
    actual_sources = []
    actual_sinks = []
    for subgraph in actual_subgraphs:
      actual_sources.append(optimizer.get_subgraph_sources(subgraph))
      actual_sinks.append(optimizer.get_subgraph_sinks(subgraph))

    expected_subgraphs = []
    expected_sources = []
    expected_sinks = []

    self.assertEqual(actual_subgraphs, expected_subgraphs)
    self.assertEqual(actual_sources, expected_sources)
    self.assertEqual(actual_sinks, expected_sinks)

  def testPipelineGetFuseableSubgraphsNotAllParentsBeamComponent(self):
    component_a = _make_fake_component_instance(
        'component_a', _OutputTypeA, {}, {}, use_fuseable_beam_executor=False)
    component_b = _make_fake_component_instance(
        'component_b', _OutputTypeB, {}, {}, use_fuseable_beam_executor=True)
    component_c = _make_fake_component_instance(
        'component_c', _OutputTypeC, {}, {}, use_fuseable_beam_executor=True)
    component_d = _make_fake_component_instance(
        'component_d', _OutputTypeD, {
            'a': component_a.outputs['output'],
            'b': component_b.outputs['output'],
            'c': component_c.outputs['output']
        }, {}, True)

    my_pipeline = pipeline.Pipeline(
        pipeline_name='my_pipeline',
        pipeline_root='root',
        components=[component_b, component_a, component_c, component_d],
        enable_cache=True,
        metadata_connection_config=self._metadata_connection_config,
        beam_pipeline_args=['--runner=PortableRunner'],
        additional_pipeline_args={})

    optimizer = beam_fusion.BeamFusionOptimizer(my_pipeline)

    actual_subgraphs = optimizer.get_fuseable_subgraphs()
    actual_sources = []
    actual_sinks = []
    for subgraph in actual_subgraphs:
      actual_sources.append(optimizer.get_subgraph_sources(subgraph))
      actual_sinks.append(optimizer.get_subgraph_sinks(subgraph))

    expected_subgraphs = [[component_b, component_c, component_d]]
    expected_sources = [{component_b, component_c}]
    expected_sinks = [{component_d}]

    self.assertEqual(actual_subgraphs, expected_subgraphs)
    self.assertEqual(actual_sources, expected_sources)
    self.assertEqual(actual_sinks, expected_sinks)

  def testPipelineGetFuseableSubgraphsFusion(self):
    component_a = _make_fake_component_instance(
        'component_a', _OutputTypeA, {}, {}, use_fuseable_beam_executor=True)
    component_b = _make_fake_component_instance(
        'component_b', _OutputTypeB, {}, {}, use_fuseable_beam_executor=True)
    component_c = _make_fake_component_instance(
        'component_c', _OutputTypeC, {
            'a': component_a.outputs['output']
        }, {}, True)
    component_d = _make_fake_component_instance(
        'component_d', _OutputTypeD, {
            'a': component_a.outputs['output']
        }, {}, True)
    component_e = _make_fake_component_instance(
        'component_e', _OutputTypeE, {
            'b': component_b.outputs['output']
        }, {}, True)
    component_f = _make_fake_component_instance(
        'component_f', _OutputTypeF, {
            'e': component_e.outputs['output']
        }, {}, True)
    component_g = _make_fake_component_instance(
        'component_g', _OutputTypeG, {
            'c': component_c.outputs['output'],
            'd': component_d.outputs['output'],
            'f': component_f.outputs['output']
        }, {}, True)
    my_pipeline = pipeline.Pipeline(
        pipeline_name='my_pipeline',
        pipeline_root='root',
        components=[component_a, component_c, component_f, component_e,
                    component_d, component_b, component_g],
        enable_cache=True,
        metadata_connection_config=self._metadata_connection_config,
        beam_pipeline_args=['--runner=PortableRunner'],
        additional_pipeline_args={})


    optimizer = beam_fusion.BeamFusionOptimizer(my_pipeline)

    actual_subgraphs = optimizer.get_fuseable_subgraphs()
    actual_sources = []
    actual_sinks = []
    for subgraph in actual_subgraphs:
      actual_sources.append(optimizer.get_subgraph_sources(subgraph))
      actual_sinks.append(optimizer.get_subgraph_sinks(subgraph))

    expected_subgraphs = [[component_a, component_b, component_c, component_d,
                           component_e, component_f, component_g]]
    expected_sources = [{component_a, component_b}]
    expected_sinks = [{component_g}]

    self.assertEqual(actual_subgraphs, expected_subgraphs)
    self.assertEqual(actual_sources, expected_sources)
    self.assertEqual(actual_sinks, expected_sinks)

  def testPipelineGetFuseableSubgraphsDiamond(self):
    component_a = _make_fake_component_instance(
        'component_a', _OutputTypeA, {}, {}, use_fuseable_beam_executor=True)
    component_b = _make_fake_component_instance(
        'component_b', _OutputTypeB, {
            'a': component_a.outputs['output']
        }, {}, use_fuseable_beam_executor=True)
    component_c = _make_fake_component_instance(
        'component_c', _OutputTypeC, {
            'a': component_a.outputs['output']
        }, {}, use_fuseable_beam_executor=False)
    component_d = _make_fake_component_instance(
        'component_d', _OutputTypeD, {
            'b': component_b.outputs['output'],
            'c': component_c.outputs['output']
        }, {}, use_fuseable_beam_executor=True)

    my_pipeline = pipeline.Pipeline(
        pipeline_name='my_pipeline',
        pipeline_root='root',
        components=[component_c, component_b, component_d, component_a],
        enable_cache=True,
        metadata_connection_config=self._metadata_connection_config,
        beam_pipeline_args=['--runner=PortableRunner'],
        additional_pipeline_args={})
    optimizer = beam_fusion.BeamFusionOptimizer(my_pipeline)

    actual_subgraphs = optimizer.get_fuseable_subgraphs()
    actual_sources = []
    actual_sinks = []
    for subgraph in actual_subgraphs:
      actual_sources.append(optimizer.get_subgraph_sources(subgraph))
      actual_sinks.append(optimizer.get_subgraph_sinks(subgraph))

    expected_subgraphs = [[component_a, component_b]]
    expected_sources = [{component_a}]
    expected_sinks = [{component_b}]

    self.assertEqual(actual_subgraphs, expected_subgraphs)
    self.assertEqual(actual_sources, expected_sources)
    self.assertEqual(actual_sinks, expected_sinks)

  def testPipelineGetFuseableSubgraphsUpstreamDependencies(self):
    component_a = _make_fake_component_instance('component_a', _OutputTypeA, {},
                                                {}, True)
    component_b = _make_fake_component_instance(
        'component_b', _OutputTypeB, {
            'a': component_a.outputs['output']
        }, {}, use_fuseable_beam_executor=True)
    component_c = _make_fake_component_instance(
        'component_c', _OutputTypeC, {
            'a': component_a.outputs['output']
        }, {}, use_fuseable_beam_executor=False)
    component_d = _make_fake_component_instance(
        'component_d', _OutputTypeD, {
            'c': component_c.outputs['output']
        }, {}, use_fuseable_beam_executor=True)
    component_e = _make_fake_component_instance(
        'component_e', _OutputTypeE, {
            'b': component_b.outputs['output'],
            'd': component_d.outputs['output']
        }, {}, use_fuseable_beam_executor=True)

    my_pipeline = pipeline.Pipeline(
        pipeline_name='my_pipeline',
        pipeline_root='root',
        components=[component_e, component_c, component_b, component_d,
                    component_a],
        enable_cache=True,
        metadata_connection_config=self._metadata_connection_config,
        beam_pipeline_args=['--runner=PortableRunner'],
        additional_pipeline_args={})
    optimizer = beam_fusion.BeamFusionOptimizer(my_pipeline)

    actual_subgraphs = optimizer.get_fuseable_subgraphs()
    actual_sources = []
    actual_sinks = []
    for subgraph in actual_subgraphs:
      actual_sources.append(optimizer.get_subgraph_sources(subgraph))
      actual_sinks.append(optimizer.get_subgraph_sinks(subgraph))

    expected_subgraphs = [[component_a, component_b],
                          [component_d, component_e]]
    expected_sources = [{component_a}, {component_d}]
    expected_sinks = [{component_b}, {component_e}]

    self.assertEqual(actual_subgraphs, expected_subgraphs)
    self.assertEqual(actual_sources, expected_sources)
    self.assertEqual(actual_sinks, expected_sinks)

  def testPipelineGetFuseableSubgraphsTwoDisjointSubgraphs(self):
    component_a = _make_fake_component_instance(
        'component_a', _OutputTypeA, {}, {}, use_fuseable_beam_executor=True)
    component_b = _make_fake_component_instance(
        'component_b', _OutputTypeB, {
            'a': component_a.outputs['output']
        }, {}, use_fuseable_beam_executor=False)
    component_c = _make_fake_component_instance(
        'component_c', _OutputTypeC, {
            'b': component_b.outputs['output']
        }, {}, use_fuseable_beam_executor=True)
    component_d = _make_fake_component_instance(
        'component_d', _OutputTypeD, {
            'c': component_c.outputs['output']
        }, {}, use_fuseable_beam_executor=True)
    component_f = _make_fake_component_instance(
        'component_f', _OutputTypeF, {
            'a': component_a.outputs['output'],
        }, {}, use_fuseable_beam_executor=True)
    component_e = _make_fake_component_instance(
        'component_e', _OutputTypeE, {
            'c': component_c.outputs['output'],
            'f': component_f.outputs['output']
        }, {}, use_fuseable_beam_executor=True)

    my_pipeline = pipeline.Pipeline(
        pipeline_name='my_pipeline',
        pipeline_root='root',
        components=[component_e, component_c, component_b, component_d,
                    component_a, component_f],
        enable_cache=True,
        metadata_connection_config=self._metadata_connection_config,
        beam_pipeline_args=['--runner=PortableRunner'],
        additional_pipeline_args={})
    optimizer = beam_fusion.BeamFusionOptimizer(my_pipeline)

    actual_subgraphs = optimizer.get_fuseable_subgraphs()
    actual_sources = []
    actual_sinks = []
    for subgraph in actual_subgraphs:
      actual_sources.append(optimizer.get_subgraph_sources(subgraph))
      actual_sinks.append(optimizer.get_subgraph_sinks(subgraph))

    expected_subgraphs = [[component_a, component_f],
                          [component_c, component_d, component_e]]
    expected_sources = [{component_a}, {component_c}]
    expected_sinks = [{component_f}, {component_d, component_e}]

    self.assertEqual(actual_subgraphs, expected_subgraphs)
    self.assertEqual(actual_sources, expected_sources)
    self.assertEqual(actual_sinks, expected_sinks)


if __name__ == '__main__':
  tf.test.main()
