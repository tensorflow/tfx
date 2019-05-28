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
"""Base class for TFX components."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import inspect
import itertools

from six import with_metaclass
from typing import Any
from typing import Dict
from typing import Optional
from typing import Text
from typing import Type

from google.protobuf import json_format
from google.protobuf import message
from tfx.components.base import base_driver
from tfx.components.base import base_executor
from tfx.utils import channel


class _PropertyDictWrapper(object):
  """Helper class to wrap outputs from TFX components."""

  def __init__(self, d: Dict[Text, channel.Channel]):
    self.__dict__ = d

  def get_all(self) -> Dict[Text, channel.Channel]:
    return self.__dict__


class ComponentSpec(with_metaclass(abc.ABCMeta, object)):
  """A specification of the inputs, outputs and parameters for a component.

  Components should have a corresponding ComponentSpec inheriting from this
  class and should override COMPONENT_NAME (as a string) and ARGS (as a list of
  ComponentArg objects).
  """

  COMPONENT_NAME = '<unknown>'
  PARAMETERS = []
  INPUTS = []
  OUTPUTS = []

  def __init__(self, component_name: Optional[Text] = None, **kwargs):
    self.component_name = component_name or self.COMPONENT_NAME
    self._raw_args = kwargs
    self._validate_spec()
    self._verify_parameter_types()
    self._parse_parameters()

  def _validate_spec(self):
    """Check the parameters and types passed to this ComponentSpec."""
    # Validate that the ComponentSpec class is well-formed.
    seen_arg_names = set()
    for arg in itertools.chain(self.PARAMETERS, self.INPUTS, self.OUTPUTS):
      assert isinstance(arg, ComponentArg), arg
      assert arg.name not in seen_arg_names, (
          'Arg name %s duplicated in %s.' % (arg.name, self.__class__))
      seen_arg_names.add(arg.name)

    # Verify that the ComponentSpec subclass has overridden arguments.
    if (self.PARAMETERS is ComponentSpec.PARAMETERS
        or self.INPUTS is ComponentSpec.INPUTS
        or self.OUTPUTS is ComponentSpec.OUTPUTS):
      raise Exception(
          'Subclass of ComponentSpec must override the PARAMETERS, INPUTS and '
          'OUTPUTS properties.')

  def _verify_parameter_types(self):
    # Verify parameter types.
    for arg in self.PARAMETERS:
      if not isinstance(arg, Parameter):
        raise Exception(
            'PARAMETERS list expects object of type Parameter, got {}.'.format(
                arg))
    for arg in self.INPUTS:
      if not isinstance(arg, ChannelInput):
        raise Exception(
            'INPUTS list expects object of type ChannelInput, got {}.'.format(
                arg))
    for arg in self.OUTPUTS:
      if not isinstance(arg, ChannelOutput):
        raise Exception(
            'OUTPUTS list expects object of type ChannelOutput, got {}.'.format(
                arg))

  def _parse_parameters(self):
    # Parse arguments to the ComponentSpec.
    unparsed_args = set(self._raw_args.keys())
    inputs = {}
    outputs = {}
    self.exec_properties = {}
    for arg in itertools.chain(self.PARAMETERS, self.INPUTS, self.OUTPUTS):
      # Check that the argument is set.
      if arg.name not in unparsed_args:
        if arg.optional:
          continue
        else:
          raise Exception(
              'Missing argument %r to %s.' % (arg.name, self.__class__))
      unparsed_args.remove(arg.name)

      # Type check the argument.
      value = self._raw_args[arg.name]
      if arg.optional and value is None:
        continue
      arg.type_check(value)

      # Populate the appropriate dictionary.
      if isinstance(arg, Parameter):
        if inspect.isclass(arg.type) and issubclass(arg.type, message.Message):
          value = json_format.MessageToJson(value)
        self.exec_properties[arg.name] = value
      elif isinstance(arg, ChannelInput):
        inputs[arg.name] = value
      elif isinstance(arg, ChannelOutput):
        outputs[arg.name] = value
      else:
        raise Exception('Unknown argument type: %s.' % arg)
    self.inputs = _PropertyDictWrapper(inputs)
    self.outputs = _PropertyDictWrapper(outputs)


class ComponentArg(object):
  """An arg that is part of a ComponentSpec."""
  pass


class Parameter(ComponentArg):
  """An execution parameter in a ComponentSpec."""

  def __init__(self, name, type=None, optional=False):  # pylint: disable=redefined-builtin
    self.name = name
    self.type = type
    self.optional = optional

  def type_check(self, value):
    # Can't type check generics. Note that we need to do this strange check form
    # since typing.GenericMeta is not exposed.
    if self.type.__class__.__name__ == 'GenericMeta':
      return
    if not isinstance(value, self.type):
      raise ValueError(
          'Expected type %s for parameter %r but got %s.' % (
              self.type, self.name, value))


class _ChannelArg(ComponentArg):
  """An channel input of a ComponentSpec."""

  def __init__(self, name, type=None):  # pylint: disable=redefined-builtin
    self.name = name
    self.type = type
    self.optional = False

  def type_check(self, value):
    assert isinstance(value, channel.Channel), (
        'Argument %s should be a Channel of type %s (got %s).' % (
            self.name, self.type, value))
    value.type_check(self.type)


class ChannelInput(_ChannelArg):
  """An channel output of a ComponentSpec."""
  pass


class ChannelOutput(_ChannelArg):
  """An channel output of a ComponentSpec."""
  pass


class BaseComponent(object):
  """A component in a TFX pipeline.

  This is the parent class of any TFX component.

  Args:
    unique_name: Unique name for every component class instance.
    spec: ComponentSpec object for this component instance.
    driver: Driver class to handle pre-execution behaviors in a component.
    executor: Executor class to do the real execution work.
  """

  def __init__(self,
               unique_name: Optional[Text],
               spec: ComponentSpec,
               executor: Type[base_executor.BaseExecutor],
               driver: Optional[Type[base_driver.BaseDriver]] = None):
    self.unique_name = unique_name
    self.spec = spec
    self.executor = executor
    self.driver = driver or base_driver.BaseDriver
    self._upstream_nodes = set()
    self._downstream_nodes = set()

  def __repr__(self):
    return """
{{
  unique_name: {unique_name},
  spec: {spec}
  executor: {executor}
  driver: {driver}
}}
    """.format(  # pylint: disable=missing-format-argument-key
        unique_name=self.unique_name,
        spec=self.spec,
        executor=self.executor,
        driver=self.driver)

  @property
  def component_name(self) -> Text:
    return self.spec.component_name

  @property
  def inputs(self) -> _PropertyDictWrapper:
    return self.spec.inputs

  @property
  def outputs(self) -> _PropertyDictWrapper:
    return self.spec.outputs

  @property
  def exec_properties(self) -> Dict[Text, Any]:
    return self.spec.exec_properties

  @property
  def upstream_nodes(self):
    return self._upstream_nodes

  def add_upstream_node(self, upstream_node):
    self._upstream_nodes.add(upstream_node)

  @property
  def downstream_nodes(self):
    return self._downstream_nodes

  def add_downstream_node(self, downstream_node):
    self._downstream_nodes.add(downstream_node)
