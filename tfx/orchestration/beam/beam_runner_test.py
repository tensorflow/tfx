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
from typing import Any, Dict, Optional, Text
from tfx.components.base import base_component
from tfx.orchestration import pipeline
from tfx.orchestration.beam import beam_runner
from tfx.utils import channel

_executed_components = []


class _FakeComponentAsDoFn(beam_runner._ComponentAsDoFn):

  def _run_component(self):
    _executed_components.append(self._name)


# TODO(jyzhao): move to a separate file to reduce duplication.
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


class BeamRunnerTest(tf.test.TestCase):

  @mock.patch.multiple(
      beam_runner,
      _ComponentAsDoFn=_FakeComponentAsDoFn,
  )
  def test_run(self):
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

    test_pipeline = pipeline.Pipeline(
        pipeline_name='x',
        pipeline_root='y',
        log_root='z',
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
