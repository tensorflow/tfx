# Lint as: python2, python3
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
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import base_node
from tfx.dsl.components.base import executor_spec
from tfx.dsl.io import fileio
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.types import node_common
from tfx.types.component_spec import ChannelParameter


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
                                  outputs: Dict[Text, types.Channel]):

  class _FakeComponentSpec(types.ComponentSpec):
    PARAMETERS = {}
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
        spec_kwargs: Dict[Text, Any]):
      spec = _FakeComponentSpec(output=types.Channel(type=type), **spec_kwargs)
      super(_FakeComponent, self).__init__(spec=spec, instance_name=name)

  spec_kwargs = dict(itertools.chain(inputs.items(), outputs.items()))
  return _FakeComponent(output_type, spec_kwargs)


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


class PipelineTest(tf.test.TestCase):

  def setUp(self):
    super(PipelineTest, self).setUp()
    tmp_dir = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir())
    self._tmp_file = os.path.join(tmp_dir, self._testMethodName,
                                  tempfile.mkstemp(prefix='cli_tmp_')[1])
    self._tmp_dir = os.path.join(tmp_dir, self._testMethodName,
                                 tempfile.mkdtemp(prefix='cli_tmp_')[1])
    # Back up the environmental variable.
    self._original_tmp_value = os.environ.get(
        'TFX_JSON_EXPORT_PIPELINE_ARGS_PATH')
    self._metadata_connection_config = metadata.sqlite_metadata_connection_config(
        os.path.join(self._tmp_dir, 'metadata'))

  def tearDown(self):
    super(PipelineTest, self).tearDown()
    # Restore the environmental variable. None means it was unset.
    if self._original_tmp_value is None:
      if 'TFX_JSON_EXPORT_PIPELINE_ARGS_PATH' in os.environ:
        del os.environ['TFX_JSON_EXPORT_PIPELINE_ARGS_PATH']
    else:
      os.environ[
          'TFX_JSON_EXPORT_PIPELINE_ARGS_PATH'] = self._original_tmp_value

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
        'component_e', _OutputTypeE, {
            'a': component_a.outputs['output'],
            'b': component_b.outputs['output'],
            'd': component_d.outputs['output']
        }, {})

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

  def testPipelineWithDuplicatedComponentId(self):
    component_a = _make_fake_component_instance('component_a', _OutputTypeA, {},
                                                {})
    component_b = _make_fake_component_instance('component_a', _OutputTypeA, {},
                                                {})
    component_c = _make_fake_component_instance('component_a', _OutputTypeA, {},
                                                {})

    with self.assertRaises(RuntimeError):
      pipeline.Pipeline(
          pipeline_name='a',
          pipeline_root='b',
          components=[component_c, component_b, component_a],
          metadata_connection_config=self._metadata_connection_config)

  def testPipelineSavePipelineArgs(self):
    os.environ['TFX_JSON_EXPORT_PIPELINE_ARGS_PATH'] = self._tmp_file
    pipeline.Pipeline(
        pipeline_name='a',
        pipeline_root='b',
        log_root='c',
        components=[
            _make_fake_component_instance('component_a', _OutputTypeA, {}, {})
        ],
        metadata_connection_config=self._metadata_connection_config)
    self.assertTrue(fileio.exists(self._tmp_file))

  def testPipelineNoTmpFolder(self):
    pipeline.Pipeline(
        pipeline_name='a',
        pipeline_root='b',
        log_root='c',
        components=[
            _make_fake_component_instance('component_a', _OutputTypeA, {}, {})
        ],
        metadata_connection_config=self._metadata_connection_config)
    self.assertNotIn('TFX_JSON_EXPORT_PIPELINE_ARGS_PATH', os.environ)

  def testPipelineWithBeamPipelineArgs(self):
    p = pipeline.Pipeline(
        pipeline_name='a',
        pipeline_root='b',
        log_root='c',
        components=[
            _make_fake_component_instance('component_a', _OutputTypeA, {}, {})
        ],
        beam_pipeline_args=['--my_testing_beam_pipeline_args=foo'],
        metadata_connection_config=self._metadata_connection_config)
    self.assertIn('--my_testing_beam_pipeline_args=foo',
                  p.components[0].executor_spec.extra_flags)


if __name__ == '__main__':
  tf.test.main()
