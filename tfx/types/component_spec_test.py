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
"""Tests for tfx.types.artifact_utils."""

import json
from typing import Dict, List

import tensorflow as tf
from tfx.dsl.placeholder import placeholder
from tfx.proto import example_gen_pb2
from tfx.types import artifact
from tfx.types import channel
from tfx.types import component_spec
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ComponentSpec
from tfx.types.component_spec import ExecutionParameter
from tfx.types.standard_artifacts import Examples
from tfx.utils import proto_utils

from google.protobuf import json_format
from google.protobuf import text_format


class _InputArtifact(artifact.Artifact):
  TYPE_NAME = 'InputArtifact'


class _OutputArtifact(artifact.Artifact):
  TYPE_NAME = 'OutputArtifact'


class _X(artifact.Artifact):
  TYPE_NAME = 'X'


class _Z(artifact.Artifact):
  TYPE_NAME = 'Z'


class _BasicComponentSpec(ComponentSpec):

  PARAMETERS = {
      'folds': ExecutionParameter(type=int),
      'proto': ExecutionParameter(type=example_gen_pb2.Input, optional=True),
  }
  INPUTS = {
      'input': ChannelParameter(type=_InputArtifact),
  }
  OUTPUTS = {
      'output': ChannelParameter(type=_OutputArtifact),
  }


class ComponentSpecTest(tf.test.TestCase):

  def testComponentspecEmpty(self):

    class EmptyComponentSpec(ComponentSpec):
      PARAMETERS = {}
      INPUTS = {}
      OUTPUTS = {}

    _ = EmptyComponentSpec()

  def testComponentspecBasic(self):
    proto = example_gen_pb2.Input()
    proto.splits.extend([
        example_gen_pb2.Input.Split(name='name1', pattern='pattern1'),
        example_gen_pb2.Input.Split(name='name2', pattern='pattern2'),
        example_gen_pb2.Input.Split(name='name3', pattern='pattern3'),
    ])
    input_channel = channel.Channel(type=_InputArtifact)
    output_channel = channel.Channel(type=_OutputArtifact)
    spec = _BasicComponentSpec(
        folds=10, proto=proto, input=input_channel, output=output_channel)
    # Verify proto property.
    self.assertIsInstance(spec.exec_properties['proto'], str)
    decoded_proto = json.loads(spec.exec_properties['proto'])
    self.assertCountEqual(['splits'], decoded_proto.keys())
    self.assertEqual(3, len(decoded_proto['splits']))
    self.assertCountEqual(['name1', 'name2', 'name3'],
                          list(s['name'] for s in decoded_proto['splits']))
    self.assertCountEqual(['pattern1', 'pattern2', 'pattern3'],
                          list(s['pattern'] for s in decoded_proto['splits']))

    # Verify other properties.
    self.assertEqual(10, spec.exec_properties['folds'])
    self.assertIs(spec.inputs['input'], input_channel)
    self.assertIs(spec.outputs['output'], output_channel)

    with self.assertRaisesRegex(
        TypeError,
        "Expected type <(class|type) 'int'> for parameter u?'folds' but got "
        'string.'):
      spec = _BasicComponentSpec(
          folds='string', input=input_channel, output=output_channel)

    with self.assertRaisesRegex(
        TypeError,
        '.*should be a Channel of .*InputArtifact.*got (.|\\s)*Examples.*'):
      spec = _BasicComponentSpec(
          folds=10, input=channel.Channel(type=Examples), output=output_channel)

    with self.assertRaisesRegex(
        TypeError,
        '.*should be a Channel of .*OutputArtifact.*got (.|\\s)*Examples.*'):
      spec = _BasicComponentSpec(
          folds=10, input=input_channel, output=channel.Channel(type=Examples))

  def testComponentSpecJsonProto(self):
    proto_str = '{"splits": [{"name": "name1", "pattern": "pattern1"}]}'
    spec = _BasicComponentSpec(
        folds=10,
        proto=proto_str,
        input=channel.Channel(type=_InputArtifact),
        output=channel.Channel(type=_OutputArtifact))
    self.assertIsInstance(spec.exec_properties['proto'], str)
    self.assertEqual(spec.exec_properties['proto'], proto_str)

  def testComponentspecWithUnionChannel(self):
    input_channel_1 = channel.Channel(type=_InputArtifact)
    input_channel_2 = channel.Channel(type=_InputArtifact)
    output_channel = channel.Channel(type=_OutputArtifact)
    spec = _BasicComponentSpec(
        folds=10,
        input=channel.union([input_channel_1, input_channel_2]),
        output=output_channel)

    # Verify properties.
    self.assertEqual(10, spec.exec_properties['folds'])
    self.assertEqual(spec.inputs['input'].type, _InputArtifact)
    self.assertEqual(spec.inputs['input'].channels,
                     [input_channel_1, input_channel_2])
    self.assertIs(spec.outputs['output'], output_channel)

  def testInvalidComponentspecMissingProperties(self):

    with self.assertRaisesRegex(TypeError, "Can't instantiate abstract class"):

      class InvalidComponentSpecA(ComponentSpec):
        # Missing PARAMETERS.
        INPUTS = {}
        OUTPUTS = {}

      InvalidComponentSpecA()

    with self.assertRaisesRegex(TypeError, "Can't instantiate abstract class"):

      class InvalidComponentSpecB(ComponentSpec):
        PARAMETERS = {}
        # Missing INPUTS.
        OUTPUTS = {}

      InvalidComponentSpecB()

    with self.assertRaisesRegex(TypeError, "Can't instantiate abstract class"):

      class InvalidComponentSpecC(ComponentSpec):
        PARAMETERS = {}
        INPUTS = {}
        # Missing OUTPUTS.

      InvalidComponentSpecC()

  def testInvalidComponentspecWrongProperties(self):

    with self.assertRaisesRegex(TypeError,
                                'must override PARAMETERS with a dict'):

      class InvalidComponentSpecA(ComponentSpec):
        PARAMETERS = object()
        INPUTS = {}
        OUTPUTS = {}

      InvalidComponentSpecA()

    with self.assertRaisesRegex(TypeError, 'must override INPUTS with a dict'):

      class InvalidComponentSpecB(ComponentSpec):
        PARAMETERS = {}
        INPUTS = object()
        OUTPUTS = {}

      InvalidComponentSpecB()

    with self.assertRaisesRegex(TypeError,
                                'must override OUTPUTS with a dict'):

      class InvalidComponentSpecC(ComponentSpec):
        PARAMETERS = {}
        INPUTS = {}
        OUTPUTS = object()

      InvalidComponentSpecC()

  def testInvalidComponentspecWrongType(self):

    class WrongTypeComponentSpecA(ComponentSpec):
      PARAMETERS = {'x': object()}
      INPUTS = {}
      OUTPUTS = {}

    with self.assertRaisesRegex(ValueError,
                                'expects .* dicts are _ComponentParameter'):
      _ = WrongTypeComponentSpecA()

    class WrongTypeComponentSpecB(ComponentSpec):
      PARAMETERS = {'x': ChannelParameter(type=_X)}
      INPUTS = {}
      OUTPUTS = {}

    with self.assertRaisesRegex(TypeError,
                                'expects values of type ExecutionParameter'):
      _ = WrongTypeComponentSpecB()

    class WrongTypeComponentSpecC(ComponentSpec):
      PARAMETERS = {}
      INPUTS = {'x': ExecutionParameter(type=int)}
      OUTPUTS = {}

    with self.assertRaisesRegex(TypeError,
                                'expect values of type ChannelParameter'):
      _ = WrongTypeComponentSpecC()

    class WrongTypeComponentSpecD(ComponentSpec):
      PARAMETERS = {}
      INPUTS = {'x': ExecutionParameter(type=int)}
      OUTPUTS = {}

    with self.assertRaisesRegex(TypeError,
                                'expect values of type ChannelParameter'):
      _ = WrongTypeComponentSpecD()

  def testInvalidComponentspecDuplicateProperty(self):

    class DuplicatePropertyComponentSpec(ComponentSpec):
      PARAMETERS = {'x': ExecutionParameter(type=int)}
      INPUTS = {'x': ChannelParameter(type=_X)}
      OUTPUTS = {}

    with self.assertRaisesRegex(ValueError, 'has a duplicate argument'):
      _ = DuplicatePropertyComponentSpec()

  def testComponentspecMissingArguments(self):

    class SimpleComponentSpec(ComponentSpec):
      PARAMETERS = {
          'x': ExecutionParameter(type=int),
          'y': ExecutionParameter(type=int, optional=True),
      }
      INPUTS = {'z': ChannelParameter(type=_Z)}
      OUTPUTS = {}

    with self.assertRaisesRegex(ValueError, 'Missing argument'):
      _ = SimpleComponentSpec(x=10)

    with self.assertRaisesRegex(ValueError, 'Missing argument'):
      _ = SimpleComponentSpec(z=channel.Channel(type=_Z))

    # Okay since y is optional.
    _ = SimpleComponentSpec(x=10, z=channel.Channel(type=_Z))

  def testOptionalInputs(self):

    class SpecWithOptionalInput(ComponentSpec):
      PARAMETERS = {}
      INPUTS = {'x': ChannelParameter(type=_Z, optional=True)}
      OUTPUTS = {}

    optional_not_specified = SpecWithOptionalInput()
    self.assertNotIn('x', optional_not_specified.inputs.keys())
    self.assertTrue(optional_not_specified.is_optional_input('x'))
    optional_specified = SpecWithOptionalInput(x=channel.Channel(type=_Z))
    self.assertIn('x', optional_specified.inputs.keys())

  def testOptionalOutputs(self):

    class SpecWithOptionalOutput(ComponentSpec):
      PARAMETERS = {}
      INPUTS = {}
      OUTPUTS = {'x': ChannelParameter(type=_Z, optional=True)}

    optional_not_specified = SpecWithOptionalOutput()
    self.assertNotIn('x', optional_not_specified.outputs.keys())
    self.assertTrue(optional_not_specified.is_optional_output('x'))
    optional_specified = SpecWithOptionalOutput(x=channel.Channel(type=_Z))
    self.assertIn('x', optional_specified.outputs.keys())

  def testChannelParameterType(self):
    arg_name = 'foo'

    class _FooArtifact(artifact.Artifact):
      TYPE_NAME = 'FooArtifact'

    class _BarArtifact(artifact.Artifact):
      TYPE_NAME = 'BarArtifact'

    channel_parameter = ChannelParameter(type=_FooArtifact)
    # Following should pass.
    channel_parameter.type_check(arg_name, channel.Channel(type=_FooArtifact))

    with self.assertRaisesRegex(TypeError, arg_name):
      channel_parameter.type_check(arg_name, 42)  # Wrong value.

    with self.assertRaisesRegex(TypeError, arg_name):
      channel_parameter.type_check(arg_name, channel.Channel(type=_BarArtifact))

    setattr(_FooArtifact, component_spec.COMPATIBLE_TYPES_KEY, {_BarArtifact})
    channel_parameter.type_check(arg_name, channel.Channel(type=_BarArtifact))

  def testExecutionParameterTypeCheck(self):
    int_parameter = ExecutionParameter(type=int)
    int_parameter.type_check('int_parameter', 8)
    with self.assertRaisesRegex(TypeError, "Expected type <(class|type) 'int'>"
                                " for parameter u?'int_parameter'"):
      int_parameter.type_check('int_parameter', 'string')

    list_parameter = ExecutionParameter(type=List[int])
    list_parameter.type_check('list_parameter', [])
    list_parameter.type_check('list_parameter', [42])
    with self.assertRaisesRegex(TypeError, 'Expecting a list for parameter'):
      list_parameter.type_check('list_parameter', 42)

    with self.assertRaisesRegex(TypeError, "Expecting item type <(class|type) "
                                "'int'> for parameter u?'list_parameter'"):
      list_parameter.type_check('list_parameter', [42, 'wrong item'])

    dict_parameter = ExecutionParameter(type=Dict[str, int])
    dict_parameter.type_check('dict_parameter', {})
    dict_parameter.type_check('dict_parameter', {'key1': 1, 'key2': 2})
    with self.assertRaisesRegex(TypeError, 'Expecting a dict for parameter'):
      dict_parameter.type_check('dict_parameter', 'simple string')

    with self.assertRaisesRegex(TypeError, "Expecting value type "
                                "<(class|type) 'int'>"):
      dict_parameter.type_check('dict_parameter', {'key1': '1'})

    proto_parameter = ExecutionParameter(type=example_gen_pb2.Input)
    proto_parameter.type_check('proto_parameter', example_gen_pb2.Input())
    proto_parameter.type_check(
        'proto_parameter', proto_utils.proto_to_json(example_gen_pb2.Input()))
    proto_parameter.type_check('proto_parameter',
                               {'splits': [{
                                   'name': 'hello'
                               }]})
    proto_parameter.type_check('proto_parameter', {'wrong_field': 42})
    with self.assertRaisesRegex(
        TypeError, "Expected type <class 'tfx.proto.example_gen_pb2.Input'>"):
      proto_parameter.type_check('proto_parameter', 42)
    with self.assertRaises(json_format.ParseError):
      proto_parameter.type_check('proto_parameter', {'splits': 42})

    placeholder_parameter = ExecutionParameter(type=str)
    placeholder_parameter.type_check(
        'placeholder_parameter',
        placeholder.runtime_info('platform_config').base_dir)
    with self.assertRaisesRegex(
        TypeError, 'Only simple RuntimeInfoPlaceholders are supported'):
      placeholder_parameter.type_check(
          'placeholder_parameter',
          placeholder.runtime_info('platform_config').base_dir +
          placeholder.exec_property('version'))

  def testExecutionParameterUseProto(self):

    class SpecWithNonPrimitiveTypes(ComponentSpec):
      PARAMETERS = {
          'config_proto':
              ExecutionParameter(type=example_gen_pb2.Input, use_proto=True),
          'boolean':
              ExecutionParameter(type=bool, use_proto=True),
          'list_config_proto':
              ExecutionParameter(
                  type=List[example_gen_pb2.Input], use_proto=True),
          'list_boolean':
              ExecutionParameter(type=List[bool], use_proto=True),
      }
      INPUTS = {
          'input': ChannelParameter(type=_InputArtifact),
      }
      OUTPUTS = {
          'output': ChannelParameter(type=_OutputArtifact),
      }

    spec = SpecWithNonPrimitiveTypes(
        config_proto='{"splits": [{"name": "name", "pattern": "pattern"}]}',
        boolean=True,
        list_config_proto=[
            example_gen_pb2.Input(splits=[
                example_gen_pb2.Input.Split(
                    name='trainer', pattern='train.data')
            ]),
            example_gen_pb2.Input(splits=[
                example_gen_pb2.Input.Split(name='eval', pattern='*eval.data')
            ])
        ],
        list_boolean=[False, True],
        input=channel.Channel(type=_InputArtifact),
        output=channel.Channel(type=_OutputArtifact))

    # Verify exec_properties store parsed value when use_proto set to True.
    expected_proto = text_format.Parse(
        """
            splits {
              name: "name"
              pattern: "pattern"
            }
          """, example_gen_pb2.Input())
    self.assertProtoEquals(expected_proto, spec.exec_properties['config_proto'])
    self.assertEqual(True, spec.exec_properties['boolean'])
    self.assertIsInstance(spec.exec_properties['list_config_proto'], list)
    self.assertEqual(spec.exec_properties['list_boolean'], [False, True])


if __name__ == '__main__':
  tf.test.main()
