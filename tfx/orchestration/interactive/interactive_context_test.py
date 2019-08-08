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
"""Tests for tfx.orchestration.interactive.interactive_context."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from typing import Any, Dict, List, Text

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.orchestration.interactive import interactive_context


class InteractiveContextTest(tf.test.TestCase):

  def testBasicRun(self):

    class _FakeComponentSpec(base_component.ComponentSpec):
      COMPONENT_NAME = '_FakeComponent'
      PARAMETERS = {}
      INPUTS = {}
      OUTPUTS = {}

    class _FakeExecutor(base_executor.BaseExecutor):
      CALLED = False

      def Do(self, input_dict: Dict[Text, List[types.Artifact]],
             output_dict: Dict[Text, List[types.Artifact]],
             exec_properties: Dict[Text, Any]) -> None:
        _FakeExecutor.CALLED = True

    class _FakeComponent(base_component.BaseComponent):
      SPEC_CLASS = _FakeComponentSpec
      EXECUTOR_CLASS = _FakeExecutor

      def __init__(self, spec: base_component.ComponentSpec):
        super(_FakeComponent, self).__init__(spec=spec)

    c = interactive_context.InteractiveContext()
    component = _FakeComponent(_FakeComponentSpec())
    c.run(component)
    self.assertTrue(_FakeExecutor.CALLED)

  def testUnresolvedChannel(self):

    class _FakeComponentSpec(base_component.ComponentSpec):
      COMPONENT_NAME = '_FakeComponent'
      PARAMETERS = {}
      INPUTS = {
          'input': base_component.ChannelParameter(type_name='Foo')
      }
      OUTPUTS = {}

    class _FakeExecutor(base_executor.BaseExecutor):
      CALLED = False

      def Do(self, input_dict: Dict[Text, List[types.Artifact]],
             output_dict: Dict[Text, List[types.Artifact]],
             exec_properties: Dict[Text, Any]) -> None:
        _FakeExecutor.CALLED = True

    class _FakeComponent(base_component.BaseComponent):
      SPEC_CLASS = _FakeComponentSpec
      EXECUTOR_CLASS = _FakeExecutor

      def __init__(self, spec: base_component.ComponentSpec):
        super(_FakeComponent, self).__init__(spec=spec)

    c = interactive_context.InteractiveContext()
    foo = types.Channel(type_name='Foo', artifacts=[types.Artifact('Foo')])
    component = _FakeComponent(_FakeComponentSpec(input=foo))
    with self.assertRaisesRegexp(ValueError, 'Unresolved input channel'):
      c.run(component)


if __name__ == '__main__':
  tf.test.main()
