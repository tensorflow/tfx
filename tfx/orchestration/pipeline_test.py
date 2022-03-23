# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Tests for tfx.orchestration.pipeline."""

import itertools
import os
from typing import Any, Dict, Optional, Type

import tensorflow as tf
from tfx import types
from tfx.dsl.components.base import base_beam_component
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import base_node
from tfx.dsl.components.base import executor_spec
from tfx.dsl.placeholder import placeholder as ph
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter
from tfx.utils import test_case_utils


class _OutputArtifact(types.Artifact):
  TYPE_NAME = 'OutputArtifact'


def _make_fake_node_instance(name: str):

  class _FakeNode(base_node.BaseNode):

    @property
    def inputs(self) -> Dict[str, Any]:
      return {}

    @property
    def outputs(self) -> Dict[str, Any]:
      return {}

    @property
    def exec_properties(self) -> Dict[str, Any]:
      return {}

  return _FakeNode().with_id(name)


def _make_fake_component_instance(
    name: str,
    output_type: Type[types.Artifact],
    inputs: Dict[str, types.Channel],
    outputs: Dict[str, types.Channel],
    with_beam: bool = False,
    dynamic_exec_property: Optional[ph.Placeholder] = None):

  class _FakeComponentSpec(types.ComponentSpec):
    PARAMETERS = {
        'exec_prop': ExecutionParameter(type=int)
    } if dynamic_exec_property is not None else {}
    INPUTS = dict([(arg, ChannelParameter(type=channel.type))
                   for arg, channel in inputs.items()])
    OUTPUTS = dict([(arg, ChannelParameter(type=channel.type))
                    for arg, channel in outputs.items()] +
                   [('output', ChannelParameter(type=output_type))])

  class _FakeComponent(base_component.BaseComponent):

    SPEC_CLASS = _FakeComponentSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(base_executor.BaseExecutor)

    def __init__(
        self,
        type: Type[types.Artifact],  # pylint: disable=redefined-builtin
        spec_kwargs: Dict[str, Any]):
      spec = _FakeComponentSpec(output=types.Channel(type=type), **spec_kwargs)
      super().__init__(spec=spec)
      self._id = name

  class _FakeBeamComponent(base_beam_component.BaseBeamComponent):

    SPEC_CLASS = _FakeComponentSpec
    EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(base_executor.BaseExecutor)

    def __init__(
        self,
        type: Type[types.Artifact],  # pylint: disable=redefined-builtin
        spec_kwargs: Dict[str, Any]):
      spec = _FakeComponentSpec(output=types.Channel(type=type), **spec_kwargs)
      super().__init__(spec=spec)
      self._id = name
      if dynamic_exec_property is not None:
        self.exec_properties['exec_prop'] = dynamic_exec_property

  spec_kwargs = dict(itertools.chain(inputs.items(), outputs.items()))
  if dynamic_exec_property is not None:
    spec_kwargs['exec_prop'] = dynamic_exec_property
  return _FakeBeamComponent(output_type,
                            spec_kwargs) if with_beam else _FakeComponent(
                                output_type, spec_kwargs)


class _ArtifactTypeOne(types.Artifact):
  TYPE_NAME = 'ArtifactTypeOne'


class _ArtifactTypeTwo(types.Artifact):
  TYPE_NAME = 'ArtifactTypeTwo'


class _ArtifactTypeThree(types.Artifact):
  TYPE_NAME = 'ArtifactTypeThree'


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


class PipelineTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._metadata_connection_config = metadata.sqlite_metadata_connection_config(
        os.path.join(self.tmp_dir, 'metadata'))

  def testPipelineWithDynamicExecProperties(self):
    component_a = _make_fake_component_instance('component_a', _OutputTypeA, {},
                                                {})
    dynamic_exec_prop = component_a.outputs['output'].future()[0].value
    component_b = _make_fake_component_instance(
        name='component_b',
        output_type=_OutputTypeB,
        inputs={},
        outputs={},
        with_beam=True,
        dynamic_exec_property=dynamic_exec_prop)

    my_pipeline = pipeline.Pipeline(
        pipeline_name='a',
        pipeline_root='b',
        components=[component_a, component_b],
        enable_cache=True,
        metadata_connection_config=self._metadata_connection_config,
        beam_pipeline_args=['--runner=PortableRunner'],
        additional_pipeline_args={})

    self.assertCountEqual(my_pipeline.components[0].downstream_nodes,
                          [component_b])
    self.assertCountEqual(my_pipeline.components[1].upstream_nodes,
                          [component_a])

  def testPipeline(self):
    component_a = _make_fake_component_instance('component_a', _OutputTypeA, {},
                                                {})
    component_b = _make_fake_component_instance(
        'component_b', _OutputTypeB, {'a': component_a.outputs['output']}, {})
    component_c = _make_fake_component_instance(
        'component_c', _OutputTypeC, {'a': component_a.outputs['output']}, {})
    component_d = _make_fake_component_instance('component_d', _OutputTypeD, {
        'b': component_b.outputs['output'],
        'c': component_c.outputs['output']
    }, {})
    component_e = _make_fake_component_instance(
        'component_e',
        _OutputTypeE, {
            'a': component_a.outputs['output'],
            'b': component_b.outputs['output'],
            'd': component_d.outputs['output']
        }, {},
        with_beam=True)

    my_pipeline = pipeline.Pipeline(
        pipeline_name='a',
        pipeline_root='b',
        components=[
            component_d, component_c, component_a, component_b, component_e,
            component_a
        ],
        enable_cache=True,
        metadata_connection_config=self._metadata_connection_config,
        beam_pipeline_args=['--runner=PortableRunner'],
        additional_pipeline_args={})
    self.assertCountEqual(
        my_pipeline.components,
        [component_a, component_b, component_c, component_d, component_e])
    self.assertCountEqual(my_pipeline.components[0].downstream_nodes,
                          [component_b, component_c, component_e])
    self.assertEqual(my_pipeline.components[-1], component_e)
    self.assertEqual(my_pipeline.pipeline_info.pipeline_name, 'a')
    self.assertEqual(my_pipeline.pipeline_info.pipeline_root, 'b')
    self.assertEqual(my_pipeline.metadata_connection_config,
                     self._metadata_connection_config)
    self.assertTrue(my_pipeline.enable_cache)
    self.assertEqual(component_e.executor_spec.beam_pipeline_args,
                     ['--runner=PortableRunner'])
    self.assertCountEqual(my_pipeline.beam_pipeline_args,
                          ['--runner=PortableRunner'])
    self.assertDictEqual(my_pipeline.additional_pipeline_args, {})

  def testPipelineWithLongname(self):
    with self.assertRaises(ValueError):
      pipeline.Pipeline(
          pipeline_name='a' * (1 + pipeline._MAX_PIPELINE_NAME_LENGTH),
          pipeline_root='root',
          components=[],
          metadata_connection_config=self._metadata_connection_config)

  def testPipelineWithNode(self):
    my_pipeline = pipeline.Pipeline(
        pipeline_name='my_pipeline',
        pipeline_root='root',
        components=[_make_fake_node_instance('my_node')],
        metadata_connection_config=self._metadata_connection_config)
    self.assertEqual(1, len(my_pipeline.components))

  def testPipelineWarnMissingNode(self):
    channel_one = types.Channel(type=_ArtifactTypeOne)
    channel_two = types.Channel(type=_ArtifactTypeTwo)
    component_a = _make_fake_component_instance('component_a', _OutputTypeA,
                                                {'a': channel_one}, {})
    component_b = _make_fake_component_instance(
        name='component_b',
        output_type=_OutputTypeB,
        inputs={'a': component_a.outputs['output']},
        outputs={'b': channel_two})

    warn_text = (
        'Node component_b depends on the output of node component_a, '
        'but component_a is not included in the components of pipeline. '
        'Did you forget to add it?')
    with self.assertWarnsRegex(UserWarning, warn_text):
      pipeline.Pipeline(
          pipeline_name='name',
          pipeline_root='root',
          components=[
              component_b,
          ],
          metadata_connection_config=self._metadata_connection_config)

  def testPipelineWithLoop(self):
    channel_one = types.Channel(type=_ArtifactTypeOne)
    channel_two = types.Channel(type=_ArtifactTypeTwo)
    channel_three = types.Channel(type=_ArtifactTypeThree)
    component_a = _make_fake_component_instance('component_a', _OutputTypeA, {},
                                                {})
    component_b = _make_fake_component_instance(
        name='component_b',
        output_type=_OutputTypeB,
        inputs={
            'a': component_a.outputs['output'],
            'one': channel_one
        },
        outputs={'two': channel_two})
    component_c = _make_fake_component_instance(
        name='component_b',
        output_type=_OutputTypeB,
        inputs={
            'a': component_a.outputs['output'],
            'two': channel_two
        },
        outputs={'three': channel_three})
    component_d = _make_fake_component_instance(
        name='component_b',
        output_type=_OutputTypeB,
        inputs={
            'a': component_a.outputs['output'],
            'three': channel_three
        },
        outputs={'one': channel_one})

    with self.assertRaises(RuntimeError):
      pipeline.Pipeline(
          pipeline_name='a',
          pipeline_root='b',
          components=[component_c, component_d, component_b, component_a],
          metadata_connection_config=self._metadata_connection_config)

  def testPipelineWithOldReferences(self):
    component_a = _make_fake_component_instance(
        name='component_a', output_type=_OutputTypeA, inputs={}, outputs={})
    component_b_v1 = _make_fake_component_instance(
        name='component_b',
        output_type=_OutputTypeB,
        inputs={
            'a': component_a.outputs['output'],
        },
        outputs={})
    component_c_v1 = _make_fake_component_instance(
        name='component_c',
        output_type=_OutputTypeC,
        inputs={
            'b': component_b_v1.outputs['output'],
        },
        outputs={})
    my_pipeline_v1 = pipeline.Pipeline(
        pipeline_name='a',
        pipeline_root='b',
        components=[component_a, component_b_v1, component_c_v1],
        metadata_connection_config=self._metadata_connection_config)
    self.assertEqual(3, len(my_pipeline_v1.components))

    component_b_v2 = _make_fake_component_instance(
        name='component_b',
        output_type=_OutputTypeB,
        inputs={
            'a': component_a.outputs['output'],  # reuses component_a
        },
        outputs={})
    component_c_v2 = _make_fake_component_instance(
        name='component_c',
        output_type=_OutputTypeC,
        inputs={
            # no dependency on component_b_v1, only depends on component_b_v2
            'b': component_b_v2.outputs['output'],
        },
        outputs={})

    my_pipeline_v2 = pipeline.Pipeline(
        pipeline_name='a',
        pipeline_root='b',
        components=[component_a, component_b_v2, component_c_v2],
        metadata_connection_config=self._metadata_connection_config)
    self.assertEqual(3, len(my_pipeline_v2.components))

  def testPipelineWithDuplicatedNodeId(self):
    component_a = _make_fake_node_instance('').with_id('component_a')
    component_b = _make_fake_component_instance('', _OutputTypeA, {},
                                                {}).with_id('component_a')
    component_c = _make_fake_component_instance('', _OutputTypeA, {},
                                                {}).with_id('component_a')

    with self.assertRaises(RuntimeError):
      pipeline.Pipeline(
          pipeline_name='a',
          pipeline_root='b',
          components=[component_c, component_b, component_a],
          metadata_connection_config=self._metadata_connection_config)

  def testPipelineWithBeamPipelineArgs(self):
    expected_args = [
        '--my_first_beam_pipeline_args=foo',
        '--my_second_beam_pipeline_args=bar'
    ]
    p = pipeline.Pipeline(
        pipeline_name='a',
        pipeline_root='b',
        log_root='c',
        components=[
            _make_fake_component_instance(
                'component_a', _OutputTypeA, {}, {},
                with_beam=True).with_beam_pipeline_args([expected_args[1]])
        ],
        beam_pipeline_args=[expected_args[0]],
        metadata_connection_config=self._metadata_connection_config)
    self.assertEqual(expected_args,
                     p.components[0].executor_spec.beam_pipeline_args)

  def testComponentsSetAfterCreationWithBeamPipelineArgs(self):
    expected_args = [
        '--my_first_beam_pipeline_args=foo',
        '--my_second_beam_pipeline_args=bar'
    ]
    p = pipeline.Pipeline(
        pipeline_name='a',
        pipeline_root='b',
        log_root='c',
        beam_pipeline_args=[expected_args[0]],
        metadata_connection_config=self._metadata_connection_config)
    p.components = [
        _make_fake_component_instance(
            'component_a', _OutputTypeA, {}, {},
            with_beam=True).with_beam_pipeline_args([expected_args[1]])
    ]
    self.assertEqual(expected_args,
                     p.components[0].executor_spec.beam_pipeline_args)


if __name__ == '__main__':
  tf.test.main()
