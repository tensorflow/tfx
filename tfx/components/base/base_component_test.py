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
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base.base_component import ChannelParameter
from tfx.components.base.base_component import ExecutionParameter
from tfx.proto import example_gen_pb2
from tfx.utils import channel


class _BasicComponentSpec(base_component.ComponentSpec):

  COMPONENT_NAME = '_BasicComponent'
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

  SPEC_CLASS = _BasicComponentSpec
  EXECUTOR_CLASS = base_executor.BaseExecutor

  def __init__(self,
               spec: base_component.ComponentSpec = None,
               folds: int = None,
               input: channel.Channel = None):  # pylint: disable=redefined-builtin
    if not spec:
      output = channel.Channel(type_name='OutputType')
      spec = _BasicComponentSpec(
          folds=folds, input=input, output=output)
    super(_BasicComponent, self).__init__(spec=spec)


class ComponentSpecTest(tf.test.TestCase):

  def test_componentspec_empty(self):

    class EmptyComponentSpec(base_component.ComponentSpec):
      COMPONENT_NAME = 'EmptyComponent'
      PARAMETERS = {}
      INPUTS = {}
      OUTPUTS = {}

    _ = EmptyComponentSpec()

  def test_componentspec_basic(self):
    proto = example_gen_pb2.Input()
    proto.splits.extend([
        example_gen_pb2.Input.Split(name='name1', pattern='pattern1'),
        example_gen_pb2.Input.Split(name='name2', pattern='pattern2'),
        example_gen_pb2.Input.Split(name='name3', pattern='pattern3'),])
    input_channel = channel.Channel(type_name='InputType')
    output_channel = channel.Channel(type_name='OutputType')
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
    self.assertIs(spec.inputs['input'], input_channel)
    self.assertIs(spec.outputs['output'], output_channel)

    with self.assertRaisesRegexp(
        TypeError,
        "Expected type <(class|type) 'int'> for parameter 'folds' but got "
        "string."):
      spec = _BasicComponentSpec(folds='string', input=input_channel,
                                 output=output_channel)

    with self.assertRaisesRegexp(
        TypeError,
        'Expected InputType but found WrongType'):
      spec = _BasicComponentSpec(folds=10,
                                 input=channel.Channel(type_name='WrongType'),
                                 output=output_channel)

    with self.assertRaisesRegexp(
        TypeError,
        'Expected OutputType but found WrongType'):
      spec = _BasicComponentSpec(folds=10,
                                 input=input_channel,
                                 output=channel.Channel(type_name='WrongType'))

  def test_invalid_componentspec_missing_properties(self):

    with self.assertRaisesRegexp(TypeError, "Can't instantiate abstract class"):
      class InvalidComponentSpecA(base_component.ComponentSpec):
        # Missing COMPONENT_NAME.
        PARAMETERS = {}
        INPUTS = {}
        OUTPUTS = {}

      InvalidComponentSpecA()

    with self.assertRaisesRegexp(TypeError, "Can't instantiate abstract class"):
      class InvalidComponentSpecB(base_component.ComponentSpec):
        COMPONENT_NAME = 'InvalidComponentB'
        # Missing PARAMETERS.
        INPUTS = {}
        OUTPUTS = {}

      InvalidComponentSpecB()

    with self.assertRaisesRegexp(TypeError, "Can't instantiate abstract class"):
      class InvalidComponentSpecC(base_component.ComponentSpec):
        COMPONENT_NAME = 'InvalidComponentC'
        PARAMETERS = {}
        # Missing INPUTS.
        OUTPUTS = {}

      InvalidComponentSpecC()

    with self.assertRaisesRegexp(TypeError, "Can't instantiate abstract class"):
      class InvalidComponentSpecD(base_component.ComponentSpec):
        COMPONENT_NAME = 'InvalidComponentD'
        PARAMETERS = {}
        INPUTS = {}
        # Missing OUTPUTS.

      InvalidComponentSpecD()

  def test_invalid_componentspec_wrong_properties(self):

    with self.assertRaisesRegexp(TypeError,
                                 'must override COMPONENT_NAME with a string'):
      class InvalidComponentSpecA(base_component.ComponentSpec):
        COMPONENT_NAME = object()
        PARAMETERS = {}
        INPUTS = {}
        OUTPUTS = {}

      InvalidComponentSpecA()

    with self.assertRaisesRegexp(TypeError,
                                 'must override PARAMETERS with a dict'):
      class InvalidComponentSpecB(base_component.ComponentSpec):
        COMPONENT_NAME = 'InvalidComponentB'
        PARAMETERS = object()
        INPUTS = {}
        OUTPUTS = {}

      InvalidComponentSpecB()

    with self.assertRaisesRegexp(TypeError,
                                 'must override INPUTS with a dict'):
      class InvalidComponentSpecC(base_component.ComponentSpec):
        COMPONENT_NAME = 'InvalidComponentC'
        PARAMETERS = {}
        INPUTS = object()
        OUTPUTS = {}

      InvalidComponentSpecC()

    with self.assertRaisesRegexp(TypeError,
                                 'must override OUTPUTS with a dict'):
      class InvalidComponentSpecD(base_component.ComponentSpec):
        COMPONENT_NAME = 'InvalidComponentD'
        PARAMETERS = {}
        INPUTS = {}
        OUTPUTS = object()

      InvalidComponentSpecD()

  def test_invalid_componentspec_wrong_type(self):

    class WrongTypeComponentSpecA(base_component.ComponentSpec):
      COMPONENT_NAME = 'WrongTypeComponentA'
      PARAMETERS = {'x': object()}
      INPUTS = {}
      OUTPUTS = {}

    with self.assertRaisesRegexp(ValueError,
                                 'expects .* dicts are _ComponentParameter'):
      _ = WrongTypeComponentSpecA()

    class WrongTypeComponentSpecB(base_component.ComponentSpec):
      COMPONENT_NAME = 'WrongTypeComponentB'
      PARAMETERS = {'x': ChannelParameter(type_name='X')}
      INPUTS = {}
      OUTPUTS = {}

    with self.assertRaisesRegexp(TypeError,
                                 'expects values of type ExecutionParameter'):
      _ = WrongTypeComponentSpecB()

    class WrongTypeComponentSpecC(base_component.ComponentSpec):
      COMPONENT_NAME = 'WrongTypeComponentC'
      PARAMETERS = {}
      INPUTS = {'x': ExecutionParameter(type=int)}
      OUTPUTS = {}

    with self.assertRaisesRegexp(TypeError,
                                 'expect values of type ChannelParameter'):
      _ = WrongTypeComponentSpecC()

    class WrongTypeComponentSpecD(base_component.ComponentSpec):
      COMPONENT_NAME = 'WrongTypeComponentD'
      PARAMETERS = {}
      INPUTS = {'x': ExecutionParameter(type=int)}
      OUTPUTS = {}

    with self.assertRaisesRegexp(TypeError,
                                 'expect values of type ChannelParameter'):
      _ = WrongTypeComponentSpecD()

  def test_invalid_componentspec_duplicate_property(self):

    class DuplicatePropertyComponentSpec(base_component.ComponentSpec):
      COMPONENT_NAME = 'DuplicatePropertyComponent'
      PARAMETERS = {'x': ExecutionParameter(type=int)}
      INPUTS = {'x': ChannelParameter(type_name='X')}
      OUTPUTS = {}

    with self.assertRaisesRegexp(ValueError,
                                 'has a duplicate argument'):
      _ = DuplicatePropertyComponentSpec()

  def test_componentspec_missing_arguments(self):

    class SimpleComponentSpec(base_component.ComponentSpec):
      COMPONENT_NAME = 'SimpleComponent'
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
      _ = SimpleComponentSpec(z=channel.Channel(type_name='Z'))

    # Okay since y is optional.
    _ = SimpleComponentSpec(x=10, z=channel.Channel(type_name='Z'))


class ComponentTest(tf.test.TestCase):

  def test_component_basic(self):
    input_channel = channel.Channel(type_name='InputType')
    component = _BasicComponent(folds=10, input=input_channel)
    self.assertIs(input_channel, component.inputs['input'])
    self.assertIsInstance(component.outputs['output'], channel.Channel)
    self.assertEqual(component.outputs['output'].type_name, 'OutputType')

  def test_component_spec_type(self):

    with self.assertRaisesRegexp(
        ValueError,
        'expects "spec" argument to be an instance of ComponentSpec'):
      _ = _BasicComponent(spec=object())  # pytype: disable=wrong-arg-types

  def test_component_spec_class(self):

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

  def test_component_executor_class(self):

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

  def test_component_custom_executor(self):

    class EmptyComponentSpec(base_component.ComponentSpec):
      COMPONENT_NAME = 'EmptyComponent'
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

  def test_component_driver_class(self):

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
