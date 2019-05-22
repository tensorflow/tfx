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

import tensorflow as tf
from typing import Any, Dict, Optional, Text

from tfx.components.base import base_component
from tfx.orchestration import pipeline
from tfx.utils import channel


class _FakeComponent(base_component.BaseComponent):

  def __init__(self,
               name: Text,
               input_dict: Dict[Text, channel.Channel],
               outputs: Optional[base_component.ComponentOutputs] = None):
    super(_FakeComponent, self).__init__(
        component_name=name,
        driver=None,
        executor=None,
        input_dict=input_dict,
        outputs=outputs,
        exec_properties={})
    self.name = name

  def _create_outputs(self) -> base_component.ComponentOutputs:
    return base_component.ComponentOutputs({
        'output': channel.Channel(type_name=self.component_name),
    })

  def _type_check(self, input_dict: Dict[Text, channel.Channel],
                  exec_properties: Dict[Text, Any]) -> None:
    return None


class PipelineTest(tf.test.TestCase):

  def test_pipeline(self):
    component_a = _FakeComponent('component_a', {})
    component_b = _FakeComponent('component_b',
                                 {'a': component_a.outputs.output})
    component_c = _FakeComponent('component_c',
                                 {'a': component_a.outputs.output})
    component_d = _FakeComponent('component_d', {
        'b': component_b.outputs.output,
        'c': component_c.outputs.output
    })
    component_e = _FakeComponent(
        'component_e', {
            'a': component_a.outputs.output,
            'b': component_b.outputs.output,
            'd': component_d.outputs.output
        })

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
    component_a = _FakeComponent('component_a', {})
    component_b = _FakeComponent(
        name='component_b',
        input_dict={
            'a': component_a.outputs.output,
            'one': channel_one
        },
        outputs=base_component.ComponentOutputs({'two': channel_two}))
    component_c = _FakeComponent(
        name='component_b',
        input_dict={
            'a': component_a.outputs.output,
            'two': channel_two
        },
        outputs=base_component.ComponentOutputs({'three': channel_three}))
    component_d = _FakeComponent(
        name='component_b',
        input_dict={
            'a': component_a.outputs.output,
            'three': channel_three
        },
        outputs=base_component.ComponentOutputs({'one': channel_one}))

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
      self.component_a = _FakeComponent('component_a', {})
      self.component_b = _FakeComponent('component_b', {})
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
