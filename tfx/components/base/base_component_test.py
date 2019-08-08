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
"""Tests for tfx.components.base.base_component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf
from tfx import types
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base.base_component import ChannelParameter
from tfx.components.base.base_component import ExecutionParameter
from tfx.proto import example_gen_pb2


class _BasicComponentSpec(base_component.ComponentSpec):

  PARAMETERS = {
      'folds': ExecutionParameter(type=int),
      'proto': ExecutionParameter(type=example_gen_pb2.Input, optional=True),
  }
  INPUTS = {
      'input': ChannelParameter(type_name='InputType'),
  }
  OUTPUTS = {
      'output': ChannelParameter(type_name='OutputType'),
  }


class _BasicComponent(base_component.BaseComponent):

  COMPONENT_NAME = 'MyBasicComponent'
  SPEC_CLASS = _BasicComponentSpec
  EXECUTOR_CLASS = base_executor.BaseExecutor

  def __init__(self,
               spec: base_component.ComponentSpec = None,
               folds: int = None,
               input: types.Channel = None):  # pylint: disable=redefined-builtin
    if not spec:
      output = types.Channel(type_name='OutputType')
      spec = _BasicComponentSpec(
          folds=folds, input=input, output=output)
    super(_BasicComponent, self).__init__(spec=spec)


class ComponentSpecTest(tf.test.TestCase):

  def testComponentspecEmpty(self):

    class EmptyComponentSpec(base_component.ComponentSpec):
      PARAMETERS = {}
      INPUTS = {}
      OUTPUTS = {}

    _ = EmptyComponentSpec()

  def testComponentspecBasic(self):
    proto = example_gen_pb2.Input()
    proto.splits.extend([
        example_gen_pb2.Input.Split(name='name1', pattern='pattern1'),
        example_gen_pb2.Input.Split(name='name2', pattern='pattern2'),
        example_gen_pb2.Input.Split(name='name3', pattern='pattern3'),])
    input_channel = types.Channel(type_name='InputType')
    output_channel = types.Channel(type_name='OutputType')
    spec = _BasicComponentSpec(folds=10,
                               proto=proto,
                               input=input_channel,
                               output=output_channel)
    # Verify proto property.
    self.assertIsInstance(spec.exec_properties['proto'], str)
    decoded_proto = json.loads(spec.exec_properties['proto'])
    self.assertCountEqual(['splits'], decoded_proto.keys())
    self.assertEqual(3, len(decoded_proto['splits']))
    self.assertCountEqual(
        ['name1', 'name2', 'name3'],
        list(s['name'] for s in decoded_proto['splits']))
    self.assertCountEqual(
        ['pattern1', 'pattern2', 'pattern3'],
        list(s['pattern'] for s in decoded_proto['splits']))

    # Verify other properties.
    self.assertEqual(10, spec.exec_properties['folds'])
    self.assertIs(spec.inputs.input, input_channel)
    self.assertIs(spec.outputs.output, output_channel)

    with self.assertRaisesRegexp(
        TypeError,
        "Expected type <(class|type) 'int'> for parameter 'folds' but got "
        "string."):
      spec = _BasicComponentSpec(folds='string', input=input_channel,
                                 output=output_channel)

    with self.assertRaisesRegexp(
        TypeError,
        'Expected InputType but found WrongType'):
      spec = _BasicComponentSpec(
          folds=10,
          input=types.Channel(type_name='WrongType'),
          output=output_channel)

    with self.assertRaisesRegexp(
        TypeError,
        'Expected OutputType but found WrongType'):
      spec = _BasicComponentSpec(
          folds=10,
          input=input_channel,
          output=types.Channel(type_name='WrongType'))

  def testInvalidComponentspecMissingProperties(self):

    with self.assertRaisesRegexp(TypeError, "Can't instantiate abstract class"):
      class InvalidComponentSpecA(base_component.ComponentSpec):
        # Missing PARAMETERS.
        INPUTS = {}
        OUTPUTS = {}

      InvalidComponentSpecA()

    with self.assertRaisesRegexp(TypeError, "Can't instantiate abstract class"):
      class InvalidComponentSpecB(base_component.ComponentSpec):
        PARAMETERS = {}
        # Missing INPUTS.
        OUTPUTS = {}

      InvalidComponentSpecB()

    with self.assertRaisesRegexp(TypeError, "Can't instantiate abstract class"):
      class InvalidComponentSpecC(base_component.ComponentSpec):
        PARAMETERS = {}
        INPUTS = {}
        # Missing OUTPUTS.

      InvalidComponentSpecC()

  def testInvalidComponentspecWrongProperties(self):

    with self.assertRaisesRegexp(TypeError,
                                 'must override PARAMETERS with a dict'):
      class InvalidComponentSpecA(base_component.ComponentSpec):
        PARAMETERS = object()
        INPUTS = {}
        OUTPUTS = {}

      InvalidComponentSpecA()

    with self.assertRaisesRegexp(TypeError,
                                 'must override INPUTS with a dict'):
      class InvalidComponentSpecB(base_component.ComponentSpec):
        PARAMETERS = {}
        INPUTS = object()
        OUTPUTS = {}

      InvalidComponentSpecB()

    with self.assertRaisesRegexp(TypeError,
                                 'must override OUTPUTS with a dict'):
      class InvalidComponentSpecC(base_component.ComponentSpec):
        PARAMETERS = {}
        INPUTS = {}
        OUTPUTS = object()

      InvalidComponentSpecC()

  def testInvalidComponentspecWrongType(self):

    class WrongTypeComponentSpecA(base_component.ComponentSpec):
      PARAMETERS = {'x': object()}
      INPUTS = {}
      OUTPUTS = {}

    with self.assertRaisesRegexp(ValueError,
                                 'expects .* dicts are _ComponentParameter'):
      _ = WrongTypeComponentSpecA()

    class WrongTypeComponentSpecB(base_component.ComponentSpec):
      PARAMETERS = {'x': ChannelParameter(type_name='X')}
      INPUTS = {}
      OUTPUTS = {}

    with self.assertRaisesRegexp(TypeError,
                                 'expects values of type ExecutionParameter'):
      _ = WrongTypeComponentSpecB()

    class WrongTypeComponentSpecC(base_component.ComponentSpec):
      PARAMETERS = {}
      INPUTS = {'x': ExecutionParameter(type=int)}
      OUTPUTS = {}

    with self.assertRaisesRegexp(TypeError,
                                 'expect values of type ChannelParameter'):
      _ = WrongTypeComponentSpecC()

    class WrongTypeComponentSpecD(base_component.ComponentSpec):
      PARAMETERS = {}
      INPUTS = {'x': ExecutionParameter(type=int)}
      OUTPUTS = {}

    with self.assertRaisesRegexp(TypeError,
                                 'expect values of type ChannelParameter'):
      _ = WrongTypeComponentSpecD()

  def testInvalidComponentspecDuplicateProperty(self):

    class DuplicatePropertyComponentSpec(base_component.ComponentSpec):
      PARAMETERS = {'x': ExecutionParameter(type=int)}
      INPUTS = {'x': ChannelParameter(type_name='X')}
      OUTPUTS = {}

    with self.assertRaisesRegexp(ValueError,
                                 'has a duplicate argument'):
      _ = DuplicatePropertyComponentSpec()

  def testComponentspecMissingArguments(self):

    class SimpleComponentSpec(base_component.ComponentSpec):
      PARAMETERS = {
          'x': ExecutionParameter(type=int),
          'y': ExecutionParameter(type=int, optional=True),
      }
      INPUTS = {'z': ChannelParameter(type_name='Z')}
      OUTPUTS = {}

    with self.assertRaisesRegexp(ValueError,
                                 'Missing argument'):
      _ = SimpleComponentSpec(x=10)

    with self.assertRaisesRegexp(ValueError,
                                 'Missing argument'):
      _ = SimpleComponentSpec(z=types.Channel(type_name='Z'))

    # Okay since y is optional.
    _ = SimpleComponentSpec(x=10, z=types.Channel(type_name='Z'))


class ComponentTest(tf.test.TestCase):

  def testComponentBasic(self):
    input_channel = types.Channel(type_name='InputType')
    component = _BasicComponent(folds=10, input=input_channel)
    self.assertEqual(component.component_name, 'MyBasicComponent')
    self.assertIs(input_channel, component.inputs.input)
    self.assertIsInstance(component.outputs.output, types.Channel)
    self.assertEqual(component.outputs.output.type_name, 'OutputType')

  def testComponentSpecType(self):

    with self.assertRaisesRegexp(
        ValueError,
        'expects "spec" argument to be an instance of ComponentSpec'):
      _ = _BasicComponent(spec=object())  # pytype: disable=wrong-arg-types

  def testComponentSpecClass(self):

    class MissingSpecComponent(base_component.BaseComponent):

      EXECUTOR_CLASS = base_executor.BaseExecutor

    with self.assertRaisesRegexp(
        TypeError,
        "Can't instantiate abstract class"):
      MissingSpecComponent(spec=object())  # pytype: disable=wrong-arg-types

    with self.assertRaisesRegexp(
        TypeError,
        'expects SPEC_CLASS property to be a subclass of '
        'base_component.ComponentSpec'):
      MissingSpecComponent._validate_component_class()

    class InvalidSpecComponent(base_component.BaseComponent):

      SPEC_CLASSES = object()
      EXECUTOR_CLASS = base_executor.BaseExecutor

    with self.assertRaisesRegexp(
        TypeError,
        'expects SPEC_CLASS property to be a subclass of '
        'base_component.ComponentSpec'):
      InvalidSpecComponent._validate_component_class()

  def testComponentExecutorClass(self):

    class MissingExecutorComponent(base_component.BaseComponent):

      SPEC_CLASS = _BasicComponentSpec

    with self.assertRaisesRegexp(
        TypeError,
        "Can't instantiate abstract class"):
      MissingExecutorComponent(spec=object())  # pytype: disable=wrong-arg-types

    with self.assertRaisesRegexp(
        TypeError,
        'expects EXECUTOR_CLASS property to be a subclass of '
        'base_executor.BaseExecutor'):
      MissingExecutorComponent._validate_component_class()

    class InvalidExecutorComponent(base_component.BaseComponent):

      SPEC_CLASS = _BasicComponentSpec
      EXECUTOR_CLASS = object()

    with self.assertRaisesRegexp(
        TypeError,
        'expects EXECUTOR_CLASS property to be a subclass of '
        'base_executor.BaseExecutor'):
      InvalidExecutorComponent._validate_component_class()

  def testComponentCustomExecutor(self):

    class EmptyComponentSpec(base_component.ComponentSpec):
      PARAMETERS = {}
      INPUTS = {}
      OUTPUTS = {}

    class MyComponent(base_component.BaseComponent):

      SPEC_CLASS = EmptyComponentSpec
      EXECUTOR_CLASS = base_executor.BaseExecutor

    class MyCustomExecutor(base_executor.BaseExecutor):
      pass

    custom_executor_component = MyComponent(
        spec=EmptyComponentSpec(),
        custom_executor_class=MyCustomExecutor)
    self.assertEqual(custom_executor_component.executor_class, MyCustomExecutor)

    with self.assertRaisesRegexp(
        TypeError,
        'should be a subclass of base_executor.BaseExecutor'):
      MyComponent(
          spec=EmptyComponentSpec(),
          custom_executor_class=object)

  def testComponentDriverClass(self):

    class InvalidDriverComponent(base_component.BaseComponent):

      SPEC_CLASS = _BasicComponentSpec
      EXECUTOR_CLASS = base_executor.BaseExecutor
      DRIVER_CLASS = object()

    with self.assertRaisesRegexp(
        TypeError,
        'expects DRIVER_CLASS property to be a subclass of '
        'base_driver.BaseDriver'):
      InvalidDriverComponent._validate_component_class()


if __name__ == '__main__':
  tf.test.main()
