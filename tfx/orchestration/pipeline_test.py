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

import tensorflow as tf
from typing import Any, Dict, Text

from tfx.components.base import base_component
from tfx.components.base.base_component import ChannelInput
from tfx.components.base.base_component import ChannelOutput
from tfx.orchestration import pipeline
from tfx.utils import channel


def get_fake_component_instance(
    name: Text,
    inputs: Dict[Text, channel.Channel],
    outputs: Dict[Text, channel.Channel]):

  class _FakeComponentSpec(base_component.ComponentSpec):
    COMPONENT_NAME = name
    PARAMETERS = []
    INPUTS = [
        ChannelInput(arg, type=channel.type_name)
        for arg, channel in inputs.items()
    ]
    OUTPUTS = [
        ChannelOutput(arg, type=channel.type_name)
        for arg, channel in outputs.items()
    ] + [ChannelOutput('output', type=name)]

  class _FakeComponent(base_component.BaseComponent):

    def __init__(self,
                 name: Text,
                 spec_kwargs: Dict[Text, Any]):
      spec = _FakeComponentSpec(
          output=channel.Channel(type_name=name),
          **spec_kwargs)
      super(_FakeComponent, self).__init__(
          unique_name=name,
          spec=spec,
          executor=None)

  spec_kwargs = dict(itertools.chain(inputs.items(), outputs.items()))
  return _FakeComponent(name, spec_kwargs)


class PipelineTest(tf.test.TestCase):

  def test_pipeline(self):
    component_a = get_fake_component_instance('component_a', {}, {})
    component_b = get_fake_component_instance(
        'component_b', {'a': component_a.outputs.output}, {})
    component_c = get_fake_component_instance(
        'component_c', {'a': component_a.outputs.output}, {})
    component_d = get_fake_component_instance('component_d', {
        'b': component_b.outputs.output,
        'c': component_c.outputs.output
    }, {})
    component_e = get_fake_component_instance(
        'component_e', {
            'a': component_a.outputs.output,
            'b': component_b.outputs.output,
            'd': component_d.outputs.output
        }, {})

    my_pipeline = pipeline.Pipeline(
        pipeline_name='a',
        pipeline_root='b',
        log_root='c',
        components=[
            component_d, component_c, component_a, component_b, component_e,
            component_a
        ])
    self.assertItemsEqual(
        my_pipeline.components,
        [component_a, component_b, component_c, component_d, component_e])
    self.assertItemsEqual(my_pipeline.components[0].downstream_nodes,
                          [component_b, component_c, component_e])
    self.assertEqual(my_pipeline.components[-1], component_e)
    self.assertDictEqual(my_pipeline.pipeline_args, {
        'pipeline_name': 'a',
        'pipeline_root': 'b',
        'log_root': 'c'
    })

  def test_pipeline_with_loop(self):
    channel_one = channel.Channel(type_name='channel_one')
    channel_two = channel.Channel(type_name='channel_two')
    channel_three = channel.Channel(type_name='channel_three')
    component_a = get_fake_component_instance('component_a', {}, {})
    component_b = get_fake_component_instance(
        name='component_b',
        inputs={
            'a': component_a.outputs.output,
            'one': channel_one
        },
        outputs={
            'two': channel_two
        })
    component_c = get_fake_component_instance(
        name='component_b',
        inputs={
            'a': component_a.outputs.output,
            'two': channel_two
        },
        outputs={
            'three': channel_three
        })
    component_d = get_fake_component_instance(
        name='component_b',
        inputs={
            'a': component_a.outputs.output,
            'three': channel_three
        },
        outputs={
            'one': channel_one
        })

    with self.assertRaises(RuntimeError):
      pipeline.Pipeline(
          pipeline_name='a',
          pipeline_root='b',
          log_root='c',
          components=[component_c, component_d, component_b, component_a])

  def test_pipeline_decorator(self):

    @pipeline.PipelineDecorator(
        pipeline_name='a', pipeline_root='b', log_root='c')
    def create_pipeline():
      self.component_a = get_fake_component_instance('component_a', {}, {})
      self.component_b = get_fake_component_instance('component_b', {}, {})
      return [self.component_a, self.component_b]

    my_pipeline = create_pipeline()

    self.assertItemsEqual(my_pipeline.components,
                          [self.component_a, self.component_b])
    self.assertDictEqual(my_pipeline.pipeline_args, {
        'pipeline_name': 'a',
        'pipeline_root': 'b',
        'log_root': 'c'
    })


if __name__ == '__main__':
  tf.test.main()
