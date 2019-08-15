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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import inspect
import itertools

from six import with_metaclass

from typing import Any, Dict, Optional, Text, Type

from google.protobuf import json_format
from google.protobuf import message
from tfx.types.artifact import Artifact
from tfx.types.channel import Channel


class _PropertyDictWrapper(object):
  """Helper class to wrap inputs/outputs from TFX components.

  Currently, this class is read-only (setting properties is not implemented).

  Internal class: no backwards compatibility guarantees.
  """

  def __init__(self,
               data: Dict[Text, Channel],
               compat_aliases: Optional[Dict[Text, Text]] = None):
    self._data = data
    self._compat_aliases = compat_aliases or {}

  def __getitem__(self, key):
    if key in self._compat_aliases:
      key = self._compat_aliases[key]
    return self._data[key]

  def __getattr__(self, key):
    if key in self._compat_aliases:
      key = self._compat_aliases[key]
    try:
      return self._data[key]
    except KeyError:
      raise AttributeError

  def __repr__(self):
    return repr(self._data)

  def get_all(self) -> Dict[Text, Channel]:
    return self._data


def _abstract_property() -> Any:
  """Returns an abstract property for use in an ABC abstract class."""
  return abc.abstractmethod(lambda: None)


class ComponentSpec(with_metaclass(abc.ABCMeta, object)):
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

  PARAMETERS = _abstract_property()
  INPUTS = _abstract_property()
  OUTPUTS = _abstract_property()

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
          issubclass(arg.type, message.Message) and value):
        # Create deterministic json string as it will be stored in metadata for
        # cache check.
        value = json_format.MessageToJson(value, sort_keys=True)
      self.exec_properties[arg_name] = value
    for arg_name, arg in self.INPUTS.items():
      if arg.optional and not self._raw_args.get(arg_name):
        continue
      value = self._raw_args[arg_name]
      inputs[arg_name] = value
    for arg_name, arg in self.OUTPUTS.items():
      value = self._raw_args[arg_name]
      outputs[arg_name] = value

    # Note: for forwards compatibility, ComponentSpec objects may provide an
    # attribute mapping virtual keys to physical keys in the outputs dictionary,
    # and when the value for a virtual key is accessed, the value for the
    # physical key will be returned instead. This is intended to provide
    # forwards compatibility. This feature will be removed once attribute
    # renaming is completed and *should not* be used by ComponentSpec authors
    # outside the TFX package.
    #
    # TODO(b/139281215): remove this functionality.
    self.inputs = _PropertyDictWrapper(
        inputs, compat_aliases=getattr(
            self, '_INPUT_COMPATIBILITY_ALIASES', None))
    self.outputs = _PropertyDictWrapper(
        outputs, compat_aliases=getattr(
            self, '_OUTPUT_COMPATIBILITY_ALIASES', None))


class _ComponentParameter(object):
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
  """

  def __init__(self, type=None, optional=False):  # pylint: disable=redefined-builtin
    self.type = type
    self.optional = optional

  def __repr__(self):
    return 'ExecutionParameter(type: %s, optional: %s)' % (self.type,
                                                           self.optional)

  def type_check(self, arg_name: Text, value: Any):
    # Can't type check generics. Note that we need to do this strange check form
    # since typing.GenericMeta is not exposed.
    if self.type.__class__.__name__ == 'GenericMeta':
      return
    if not isinstance(value, self.type):
      raise TypeError('Expected type %s for parameter %r but got %s.' %
                      (self.type, arg_name, value))


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
      type_name: Optional[Text] = None,
      type: Optional[Type[Artifact]] = None,  # pylint: disable=redefined-builtin
      optional: Optional[bool] = False):
    # TODO(b/138664975): either deprecate or remove string-based artifact type
    # definition before 0.14.0 release.
    if bool(type_name) == bool(type):
      raise ValueError(
          'Exactly one of "type" or "type_name" must be passed to the '
          'constructor of Channel.')
    if type:
      if not issubclass(type, Artifact):  # pytype: disable=wrong-arg-types
        raise ValueError(
            'Argument "type" of Channel constructor must be a subclass of'
            'tfx.types.Artifact.')
      type_name = type.TYPE_NAME  # pytype: disable=attribute-error
    self.type_name = type_name
    self.optional = optional

  def __repr__(self):
    return 'ChannelParameter(type_name: %s)' % (self.type_name,)

  def type_check(self, arg_name: Text, value: Channel):
    if not isinstance(value, Channel) or value.type_name != self.type_name:
      raise TypeError(
          'Argument %s should be a Channel of type_name %r (got %s).' %
          (arg_name, self.type_name, value))
