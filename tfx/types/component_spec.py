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
"""ComponentSpec for defining inputs/outputs/properties of TFX components."""

import copy
import inspect
import itertools
from typing import Any, Dict, List, Optional, Type, cast

from tfx.dsl.placeholder import placeholder
from tfx.types import artifact
from tfx.types import channel
from tfx.utils import abc_utils
from tfx.utils import json_utils
from tfx.utils import proto_utils

from google.protobuf import message


def _is_runtime_param(data: Any) -> bool:
  return data.__class__.__name__ == 'RuntimeParameter'


def _make_default(data: Any) -> Any:
  """Replaces RuntimeParameter by its ptype's default.

  Args:
    data: an object possibly containing RuntimeParameter.

  Returns:
    A version of input data where RuntimeParameters are replaced with
    the default values of their ptype.
  """
  if isinstance(data, dict):
    copy_data = copy.deepcopy(data)
    _put_default_dict(copy_data)
    return copy_data
  if isinstance(data, list):
    copy_data = copy.deepcopy(data)
    _put_default_list(copy_data)
    return copy_data
  if _is_runtime_param(data):
    ptype = data.ptype
    return ptype.__new__(ptype)

  return data


def _put_default_dict(dict_data: Dict[str, Any]) -> None:
  """Helper function to replace RuntimeParameter with its default value."""
  for k, v in dict_data.items():
    if isinstance(v, dict):
      _put_default_dict(v)
    elif isinstance(v, list):
      _put_default_list(v)
    elif v.__class__.__name__ == 'RuntimeParameter':
      # Currently supporting int, float, bool, Text
      ptype = v.ptype
      dict_data[k] = ptype.__new__(ptype)


def _put_default_list(list_data: List[Any]) -> None:
  """Helper function to replace RuntimeParameter with its default value."""
  for index, item in enumerate(list_data):
    if isinstance(item, dict):
      _put_default_dict(item)
    elif isinstance(item, list):
      _put_default_list(item)
    elif item.__class__.__name__ == 'RuntimeParameter':
      # Currently supporting int, float, bool, Text
      ptype = item.ptype
      list_data[index] = ptype.__new__(ptype)


class ComponentSpec(json_utils.Jsonable):
  """A specification of the inputs, outputs and parameters for a component.

  Components should have a corresponding ComponentSpec inheriting from this
  class and must override:

    - PARAMETERS (as a dict of string keys and ExecutionParameter values),
    - INPUTS (as a dict of string keys and ChannelParameter values) and
    - OUTPUTS (also a dict of string keys and ChannelParameter values).

  Here is an example of how a ComponentSpec may be defined:

  class MyCustomComponentSpec(ComponentSpec):
    PARAMETERS = {
        'internal_option': ExecutionParameter(type=str),
    }
    INPUTS = {
        'input_examples': ChannelParameter(type=standard_artifacts.Examples),
    }
    OUTPUTS = {
        'output_examples': ChannelParameter(type=standard_artifacts.Examples),
    }

  To create an instance of a subclass, call it directly with any execution
  parameters / inputs / outputs as kwargs.  For example:

  spec = MyCustomComponentSpec(
      internal_option='abc',
      input_examples=input_examples_channel,
      output_examples=output_examples_channel)

  Attributes:
    PARAMETERS: a dict of string keys and ExecutionParameter values.
    INPUTS: a dict of string keys and ChannelParameter values.
    OUTPUTS: a dict of string keys and ChannelParameter values.
  """

  PARAMETERS = abc_utils.abstract_property()
  INPUTS = abc_utils.abstract_property()
  OUTPUTS = abc_utils.abstract_property()

  def __init__(self, **kwargs):
    """Initialize a ComponentSpec.

    Args:
      **kwargs: Any inputs, outputs and execution parameters for this instance
        of the component spec.
    """
    self._raw_args = kwargs
    self._validate_spec()
    self._verify_parameter_types()
    self._parse_parameters()

  def __eq__(self, other):
    return (isinstance(other.__class__, self.__class__) and
            self.to_json_dict() == other.to_json_dict())

  def _validate_spec(self):
    """Check the parameters and types passed to this ComponentSpec."""
    for param_name, param in [('PARAMETERS', self.PARAMETERS),
                              ('INPUTS', self.INPUTS),
                              ('OUTPUTS', self.OUTPUTS)]:
      if not isinstance(param, dict):
        raise TypeError(
            ('Subclass %s of ComponentSpec must override %s with a '
             'dict; got %s instead.') % (self.__class__, param_name, param))

    # Validate that the ComponentSpec class is well-formed.
    seen_arg_names = set()
    for arg_name, arg in itertools.chain(self.PARAMETERS.items(),
                                         self.INPUTS.items(),
                                         self.OUTPUTS.items()):
      if not isinstance(arg, _ComponentParameter):
        raise ValueError(
            ('The ComponentSpec subclass %s expects that the values of its '
             'PARAMETERS, INPUTS, and OUTPUTS dicts are _ComponentParameter '
             'objects (i.e. ChannelParameter or ExecutionParameter objects); '
             'got %s (for argument %s) instead.') %
            (self.__class__, arg, arg_name))
      if arg_name in seen_arg_names:
        raise ValueError(
            ('The ComponentSpec subclass %s has a duplicate argument with '
             'name %s. Argument names should be unique across the PARAMETERS, '
             'INPUTS and OUTPUTS dicts.') % (self.__class__, arg_name))
      seen_arg_names.add(arg_name)

  def _verify_parameter_types(self):
    """Verify spec parameter types."""
    for arg in self.PARAMETERS.values():
      if not isinstance(arg, ExecutionParameter):
        raise TypeError(
            ('PARAMETERS dict expects values of type ExecutionParameter, '
             'got {}.').format(arg))
    for arg in itertools.chain(self.INPUTS.values(), self.OUTPUTS.values()):
      if not isinstance(arg, ChannelParameter):
        raise TypeError(
            ('INPUTS and OUTPUTS dicts expect values of type ChannelParameter, '
             ' got {}.').format(arg))

  def _parse_parameters(self):
    """Parse arguments to ComponentSpec."""
    unparsed_args = set(self._raw_args.keys())
    inputs = {}
    outputs = {}
    self.exec_properties = {}

    # First, check that the arguments are set.
    for arg_name, arg in itertools.chain(self.PARAMETERS.items(),
                                         self.INPUTS.items(),
                                         self.OUTPUTS.items()):
      if arg_name not in unparsed_args:
        if arg.optional:
          continue
        else:
          raise ValueError('Missing argument %r to %s.' %
                           (arg_name, self.__class__))
      unparsed_args.remove(arg_name)

      # Type check the argument.
      value = self._raw_args[arg_name]
      if arg.optional and value is None:
        continue
      arg.type_check(arg_name, value)

    # Populate the appropriate dictionary for each parameter type.
    for arg_name, arg in self.PARAMETERS.items():
      if arg.optional and arg_name not in self._raw_args:
        continue
      value = self._raw_args[arg_name]

      if (inspect.isclass(arg.type) and
          issubclass(arg.type, message.Message) and value and
          not _is_runtime_param(value)):
        if arg.use_proto:
          if isinstance(value, dict):
            value = proto_utils.dict_to_proto(value, arg.type())
          elif isinstance(value, str):
            value = proto_utils.json_to_proto(value, arg.type())
        else:
          # Create deterministic json string as it will be stored in metadata
          # for cache check.
          if isinstance(value, dict):
            value = json_utils.dumps(value)
          elif not isinstance(value, str):
            value = proto_utils.proto_to_json(value)

      self.exec_properties[arg_name] = value

    for arg_dict, param_dict in (
        (self.INPUTS, inputs), (self.OUTPUTS, outputs)):
      for arg_name, arg in arg_dict.items():
        if arg.optional and not self._raw_args.get(arg_name):
          continue
        value = self._raw_args[arg_name]
        param_dict[arg_name] = value

    self.inputs = inputs
    self.outputs = outputs

  def is_optional_input(self, key: str) -> bool:
    """Whether the input channel of the key is optional."""
    try:
      return cast(ChannelParameter, self.INPUTS[key]).optional
    except KeyError as e:
      raise KeyError(f'self.INPUTS = {self.INPUTS}') from e

  def is_optional_output(self, key: str) -> bool:
    """Whether the output channel of the key is optional."""
    return cast(ChannelParameter, self.OUTPUTS[key]).optional

  def is_optional_exec_property(self, key: str) -> bool:
    """Whether the exec_properties of the key is optional."""
    return cast(ExecutionParameter, self.PARAMETERS[key]).optional

  def to_json_dict(self) -> Dict[str, Any]:
    """Convert from an object to a JSON serializable dictionary."""
    return {
        'inputs': self.inputs,
        'outputs': self.outputs,
        'exec_properties': self.exec_properties,
    }


class _ComponentParameter:
  """An abstract parameter that forms a part of a ComponentSpec.

  Properties:
    optional: whether the given parameter is optional.
  """
  pass


class ExecutionParameter(_ComponentParameter):
  """An execution parameter in a ComponentSpec.

  This type of parameter should be specified in the PARAMETERS dict of a
  ComponentSpec:

  class MyCustomComponentSpec(ComponentSpec):
    # ...
    PARAMETERS = {
        'internal_option': ExecutionParameter(type=str),
    }
    # ...

  Attributes:
    type: Type of the execution parameter.
    optional: Boolean value indicating whether the parameter is optional.
    use_proto: Boolean value indicating whether pb message (and other
      non-primitive types like lists) should be stored in its original form.
  """

  def __init__(self, type=None, optional=False, use_proto=False):  # pylint: disable=redefined-builtin
    self.type = type
    self.optional = optional
    self.use_proto = use_proto

    if self.type in [int, float, str] and self.use_proto:
      raise ValueError('use_proto set for primitive type %s' % self.type)

  def __repr__(self):
    return 'ExecutionParameter(type: %s, optional: %s, use_proto: %s)' % (
        self.type, self.optional, self.use_proto)

  def __eq__(self, other):
    return (isinstance(other.__class__, self.__class__) and
            other.type == self.type and other.optional == self.optional and
            other.use_proto == self.use_proto)

  def type_check(self, arg_name: str, value: Any):
    """Perform type check to the parameter passed in."""

    # Following helper function is needed due to the lack of subscripted
    # type check support in Python 3.7. Here we hold the assumption that no
    # nested container type is declared as the parameter type.
    # For example:
    # Dict[Text, List[str]] <------ Not allowed.
    # Dict[Text, Any] <------ Okay.
    def _type_check_helper(value: Any, declared: Type):  # pylint: disable=g-bare-generic
      """Helper type-checking function."""
      if isinstance(value, placeholder.Placeholder):
        placeholders_involved = value.placeholders_involved()
        if (len(placeholders_involved) != 1 or not isinstance(
            placeholders_involved[0], placeholder.RuntimeInfoPlaceholder)):
          placeholders_involved_str = [
              x.__class__.__name__ for x in placeholders_involved
          ]
          raise TypeError(
              'Only simple RuntimeInfoPlaceholders are supported, but while '
              'checking parameter %r, the following placeholders were '
              'involved: %s' % (arg_name, placeholders_involved_str))
        if not issubclass(declared, str):
          raise TypeError(
              'Cannot use Placeholders except for str parameter, but parameter '
              '%r was of type %s' % (arg_name, declared))
        return

      is_runtime_param = _is_runtime_param(value)
      value = _make_default(value)
      if declared == Any:
        return
      if declared.__class__.__name__ in ('_GenericAlias', 'GenericMeta'):
        # Should be dict or list
        if declared.__origin__ in [Dict, dict]:  # pylint: disable=protected-access
          key_type, val_type = declared.__args__[0], declared.__args__[1]
          if not isinstance(value, dict):
            raise TypeError('Expecting a dict for parameter %r, but got %s '
                            'instead' % (arg_name, type(value)))
          for k, v in value.items():
            if key_type != Any and not isinstance(k, key_type):
              raise TypeError('Expecting key type %s for parameter %r, '
                              'but got %s instead.' %
                              (str(key_type), arg_name, type(k)))
            if val_type != Any and not isinstance(v, val_type):
              raise TypeError('Expecting value type %s for parameter %r, '
                              'but got %s instead.' % (
                                  str(val_type), arg_name, type(v)))
        elif declared.__origin__ in [List, list]:  # pylint: disable=protected-access
          val_type = declared.__args__[0]
          if not isinstance(value, list):
            raise TypeError('Expecting a list for parameter %r, '
                            'but got %s instead.' % (arg_name, type(value)))
          if val_type == Any:
            return
          for item in value:
            if not isinstance(item, val_type):
              raise TypeError('Expecting item type %s for parameter %r, '
                              'but got %s instead.' % (
                                  str(val_type), arg_name, type(item)))
        else:
          raise TypeError('Unexpected type of parameter: %r' % arg_name)
      elif isinstance(value, dict) and issubclass(declared, message.Message):
        # If a dict is passed in and is compared against a pb message,
        # do the type-check by converting it to pb message.
        proto_utils.dict_to_proto(value, declared())
      elif (isinstance(value, str) and not isinstance(declared, tuple) and
            issubclass(declared, message.Message)):
        # Skip check for runtime param string proto.
        if not is_runtime_param:
          # If a text is passed in and is compared against a pb message,
          # do the type-check by converting text (as json) to pb message.
          proto_utils.json_to_proto(value, declared())
      else:
        if not isinstance(value, declared):
          raise TypeError('Expected type %s for parameter %r '
                          'but got %s instead.' % (
                              str(declared), arg_name, value))

    _type_check_helper(value, self.type)


COMPATIBLE_TYPES_KEY = '_compatible_types'


class ChannelParameter(_ComponentParameter):
  """An channel parameter that forms part of a ComponentSpec.

  This type of parameter should be specified in the INPUTS and OUTPUTS dict
  fields of a ComponentSpec:

  class MyCustomComponentSpec(ComponentSpec):
    # ...
    INPUTS = {
        'input_examples': ChannelParameter(type=standard_artifacts.Examples),
    }
    OUTPUTS = {
        'output_examples': ChannelParameter(type=standard_artifacts.Examples),
    }
    # ...
  """

  def __init__(
      self,
      type: Optional[Type[artifact.Artifact]] = None,  # pylint: disable=redefined-builtin
      optional: bool = False):
    if not (inspect.isclass(type) and issubclass(type, artifact.Artifact)):  # pytype: disable=wrong-arg-types
      raise ValueError(
          'Argument "type" of Channel constructor must be a subclass of '
          'tfx.types.Artifact.')
    self.type = type
    self.optional = optional

  def __repr__(self):
    return 'ChannelParameter(type: %s)' % self.type

  def __eq__(self, other):
    return (isinstance(other.__class__, self.__class__) and
            other.type == self.type and other.optional == self.optional)

  def type_check(self, arg_name: str, value: channel.BaseChannel):
    if ((not isinstance(value, channel.BaseChannel)) or
        not (value.type is self.type or
             value.type in getattr(self.type, COMPATIBLE_TYPES_KEY, ()))):
      raise TypeError('Argument %s should be a Channel of type %r (got %s).' %
                      (arg_name, self.type, value))
