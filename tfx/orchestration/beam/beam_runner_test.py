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
"""Tests for tfx.orchestration.beam.runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
import tensorflow as tf
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base.base_component import ChannelParameter
from tfx.orchestration import pipeline
from tfx.orchestration.beam import beam_runner
from tfx.utils import channel

_executed_components = []


class _FakeComponentAsDoFn(beam_runner._ComponentAsDoFn):

  def _run_component(self):
    _executed_components.append(self._name)


# We define fake component spec classes below for testing. Note that we can't
# programmatically generate component using anonymous classes for testing
# because of a limitation in the "dill" pickler component used by Apache Beam.
# An alternative we considered but rejected here was to write a function that
# returns anonymous classes within that function's closure (as is done in
# tfx/orchestration/pipeline_test.py), but that strategy does not work here
# as these anonymous classes cannot be used with Beam, since they cannot be
# pickled with the "dill" library.
class _FakeComponentSpecA(base_component.ComponentSpec):
  COMPONENT_NAME = 'component_a'
  PARAMETERS = {}
  INPUTS = {}
  OUTPUTS = {'output': ChannelParameter(type_name='a')}


class _FakeComponentSpecB(base_component.ComponentSpec):
  COMPONENT_NAME = 'component_b'
  PARAMETERS = {}
  INPUTS = {'a': ChannelParameter(type_name='a')}
  OUTPUTS = {'output': ChannelParameter(type_name='b')}


class _FakeComponentSpecC(base_component.ComponentSpec):
  COMPONENT_NAME = 'component_c'
  PARAMETERS = {}
  INPUTS = {'a': ChannelParameter(type_name='a')}
  OUTPUTS = {'output': ChannelParameter(type_name='c')}


class _FakeComponentSpecD(base_component.ComponentSpec):
  COMPONENT_NAME = 'component_d'
  PARAMETERS = {}
  INPUTS = {
      'b': ChannelParameter(type_name='b'),
      'c': ChannelParameter(type_name='c'),
  }
  OUTPUTS = {'output': ChannelParameter(type_name='d')}


class _FakeComponentSpecE(base_component.ComponentSpec):
  COMPONENT_NAME = 'component_e'
  PARAMETERS = {}
  INPUTS = {
      'a': ChannelParameter(type_name='a'),
      'b': ChannelParameter(type_name='b'),
      'd': ChannelParameter(type_name='d'),
  }
  OUTPUTS = {'output': ChannelParameter(type_name='e')}


class _FakeComponent(base_component.BaseComponent):

  SPEC_CLASS = base_component.ComponentSpec
  EXECUTOR_CLASS = base_executor.BaseExecutor

  def __init__(self, spec: base_component.ComponentSpec):
    super(_FakeComponent, self).__init__(spec=spec)


class BeamRunnerTest(tf.test.TestCase):

  @mock.patch.multiple(
      beam_runner,
      _ComponentAsDoFn=_FakeComponentAsDoFn,
  )
  def test_run(self):
    component_a = _FakeComponent(_FakeComponentSpecA(
        output=channel.Channel(type_name='a')))
    component_b = _FakeComponent(
        _FakeComponentSpecB(
            a=component_a.outputs['output'],
            output=channel.Channel(type_name='b')))
    component_c = _FakeComponent(
        _FakeComponentSpecC(
            a=component_a.outputs['output'],
            output=channel.Channel(type_name='c')))
    component_d = _FakeComponent(
        _FakeComponentSpecD(
            b=component_b.outputs['output'],
            c=component_c.outputs['output'],
            output=channel.Channel(type_name='d')))
    component_e = _FakeComponent(
        _FakeComponentSpecE(
            a=component_a.outputs['output'],
            b=component_b.outputs['output'],
            d=component_d.outputs['output'],
            output=channel.Channel(type_name='e')))

    test_pipeline = pipeline.Pipeline(
        pipeline_name='x',
        pipeline_root='y',
        metadata_connection_config=None,
        components=[
            component_d, component_c, component_a, component_b, component_e
        ])

    beam_runner.BeamRunner().run(test_pipeline)
    self.assertItemsEqual(_executed_components, [
        'component_a', 'component_b', 'component_c', 'component_d',
        'component_e'
    ])
    self.assertEqual(_executed_components[0], 'component_a')
    self.assertEqual(_executed_components[3], 'component_d')
    self.assertEqual(_executed_components[4], 'component_e')


if __name__ == '__main__':
  tf.test.main()
