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


def _abstract_property() -> Any:
  """Returns an abstract property for use in an ABC abstract class."""
  return abc.abstractmethod(lambda: None)


class ComponentSpec(with_metaclass(abc.ABCMeta, object)):
  """A specification of the inputs, outputs and parameters for a component.

  Components should have a corresponding ComponentSpec inheriting from this
  class and must override:

    - COMPONENT_NAME (as a string),
    - PARAMETERS (as a dict of string keys and ExecutionParameter values),
    - INPUTS (as a dict of string keys and ChannelParameter values) and
    - OUTPUTS (also a dict of string keys and ChannelParameter values).

  Here is an example of how a ComponentSpec may be defined:

  class MyCustomComponentSpec(ComponentSpec):
    COMPONENT_NAME = 'MyCustomComponent'
    PARAMETERS = {
        'internal_option': ExecutionParameter(type=str),
    }
    INPUTS = {
        'input_examples': ChannelParameter(type_name='ExamplesPath'),
    }
    OUTPUTS = {
        'output_examples': ChannelParameter(type_name='ExamplesPath'),
    }

  To create an instance of a subclass, call it directly with any execution
  parameters / inputs / outputs as kwargs.  For example:

  spec = MyCustomComponentSpec(
      internal_option='abc',
      input_examples=input_examples_channel,
      output_examples=output_examples_channel)

  Optionally, the name of the component may be overridden by passing the
  component_name kwarg.

  Attributes:
    COMPONENT_NAME: name of this component, as a string.
    PARAMETERS: a dict of string keys and ExecutionParameter values.
    INPUTS: a dict of string keys and ChannelParameter values.
    OUTPUTS: a dict of string keys and ChannelParameter values.
  """

  COMPONENT_NAME = _abstract_property()
  PARAMETERS = _abstract_property()
  INPUTS = _abstract_property()
  OUTPUTS = _abstract_property()

  def __init__(self, component_name: Optional[Text] = None, **kwargs):
    """Initialize a ComponentSpec.

    Args:
      component_name: Optional component name to use for this instance of the
        component spec, instead of the default one for this ComponentSpec
        subclass.
      **kwargs: Any inputs, outputs and execution parameters for this instance
        of the component spec.
    """
    self.component_name = component_name or self.COMPONENT_NAME
    self._raw_args = kwargs
    self._validate_spec()
    self._verify_parameter_types()
    self._parse_parameters()

  def _validate_spec(self):
    """Check the parameters and types passed to this ComponentSpec."""
    # Verify that the ComponentSpec subclass has overridden arguments of the
    # correct type.
    if not isinstance(self.COMPONENT_NAME,
                      (str, Text)):  # str needed for Python2 compatibility.
      raise TypeError(
          ('Subclass %s of ComponentSpec must override COMPONENT_NAME with a '
           'string; got %s instead.') % (self.__class__, self.COMPONENT_NAME))
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
        value = json_format.MessageToJson(value)
      self.exec_properties[arg_name] = value
    for arg_name, arg in self.INPUTS.items():
      value = self._raw_args[arg_name]
      inputs[arg_name] = value
    for arg_name, arg in self.OUTPUTS.items():
      value = self._raw_args[arg_name]
      outputs[arg_name] = value

    self.inputs = _PropertyDictWrapper(inputs)
    self.outputs = _PropertyDictWrapper(outputs)


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
        'input_examples': ChannelParameter(type_name='ExamplesPath'),
    }
    OUTPUTS = {
        'output_examples': ChannelParameter(type_name='ExamplesPath'),
    }
    # ...
  """

  def __init__(self, type_name: Text = None):
    self.type_name = type_name
    # Right now, channel parameters are never optional.
    self.optional = False

  def __repr__(self):
    return 'ChannelParameter(type_name: %s)' % (self.type_name,)

  def type_check(self, arg_name: Text, value: channel.Channel):
    if not isinstance(value, channel.Channel):
      raise TypeError(
          'Argument %s should be a Channel of type_name %r (got %s).' %
          (arg_name, self.type_name, value))
    value.type_check(self.type_name)


class BaseComponent(with_metaclass(abc.ABCMeta, object)):
  """Base class for a TFX pipeline component.

  An instance of a subclass of BaseComponent represents the parameters for a
  single execution of that TFX pipeline component.

  All subclasses of BaseComponent must override the SPEC_CLASSES field with a
  list of all valid ComponentSpec subclasses that this base component accepts.

  Attributes:
    SPEC_CLASS: a subclass of ComponentSpec used by this component (required).
    EXECUTOR_CLASS: a subclass of base_executor.BaseExecutor used to execute
      this component (required).
    DRIVER_CLASS: a subclass of base_driver.BaseDriver as a custom driver for
      this component (optional, defaults to base_driver.BaseDriver).
  """

  # Subclasses must override this property (by specifying a ComponentSpec class,
  # e.g. "SPEC_CLASS = MyComponentSpec").
  SPEC_CLASS = _abstract_property()
  # Subclasses must also override the executor class.
  EXECUTOR_CLASS = _abstract_property()
  # Subclasses will usually use the default driver class, but may override this
  # property as well.
  DRIVER_CLASS = base_driver.BaseDriver

  def __init__(
      self,
      spec: ComponentSpec,
      custom_executor_class: Optional[Type[base_executor.BaseExecutor]] = None,
      name: Optional[Text] = None):
    """Initialize a component.

    Args:
      spec: ComponentSpec object for this component instance.
      custom_executor_class: Optional custom executor class overriding the
        default executor specified in the component attribute.
      name: Optional unique identifying name for this instance of the component
        in the pipeline. Required if two instances of the same component is used
        in the pipeline.
    """
    self.spec = spec
    if custom_executor_class:
      if not issubclass(custom_executor_class, base_executor.BaseExecutor):
        raise TypeError(
            ('Custom executor class override %s for %s should be a subclass of '
             'base_executor.BaseExecutor') %
            (custom_executor_class, self.__class__))
    self.executor_class = (
        custom_executor_class or self.__class__.EXECUTOR_CLASS)
    self.driver_class = self.__class__.DRIVER_CLASS
    self.name = name
    self._upstream_nodes = set()
    self._downstream_nodes = set()
    self._validate_component_class()
    self._validate_spec(spec)

  @classmethod
  def _validate_component_class(cls):
    """Validate that the SPEC_CLASSES property of this class is set properly."""
    if not (inspect.isclass(cls.SPEC_CLASS) and
            issubclass(cls.SPEC_CLASS, ComponentSpec)):
      raise TypeError(
          ('Component class %s expects SPEC_CLASS property to be a subclass '
           'of base_component.ComponentSpec; got %s instead.') %
          (cls, cls.SPEC_CLASS))
    if not (inspect.isclass(cls.EXECUTOR_CLASS) and
            issubclass(cls.EXECUTOR_CLASS, base_executor.BaseExecutor)):
      raise TypeError((
          'Component class %s expects EXECUTOR_CLASS property to be a subclass '
          'of base_executor.BaseExecutor; got %s instead.') %
                      (cls, cls.EXECUTOR_CLASS))
    if not (inspect.isclass(cls.DRIVER_CLASS) and
            issubclass(cls.DRIVER_CLASS, base_driver.BaseDriver)):
      raise TypeError(
          ('Component class %s expects DRIVER_CLASS property to be a subclass '
           'of base_driver.BaseDriver; got %s instead.') %
          (cls, cls.DRIVER_CLASS))

  def _validate_spec(self, spec):
    """Verify given spec is valid given the component's SPEC_CLASS."""
    if not isinstance(spec, ComponentSpec):
      raise ValueError((
          'BaseComponent (parent class of %s) expects "spec" argument to be an '
          'instance of ComponentSpec, got %s instead.') %
                       (self.__class__, spec))
    if not isinstance(spec, self.__class__.SPEC_CLASS):
      raise ValueError(
          ('%s expects the "spec" argument to be an instance of %s; '
           'got %s instead.') %
          (self.__class__, self.__class__.SPEC_CLASS, spec))

  def __repr__(self):
    return '%s(spec: %s, executor_class: %s, driver_class: %s, name: %s)' % (
        self.__class__.__name__, self.spec, self.executor_class,
        self.driver_class, self.name)

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

  # TODO(ruoyu): Consolidate the usage of component identifier. Moving forward,
  # we will have two component level keys:
  # - component_type: the path of the python executor or the image uri of the
  #   executor.
  # - component_id: <component_name>.<unique_name>
  @property
  def component_id(self):
    """Component id.

    If unique name is available, component_id will be:
      <component_name>.<unique_name>
    otherwise, component_id will be:
      <component_name>.default

    Returns:
      component id.
    """
    if self.name:
      return '{}.{}'.format(self.component_name, self.name or 'default')
    else:
      return self.component_name

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
