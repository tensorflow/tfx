# Lint as: python2, python3
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
import typing
from typing import Any, Dict, List, Optional, Text, Type

from six import with_metaclass

from tfx import types
from tfx.components.base import base_driver
from tfx.components.base import base_executor
from tfx.components.base import base_node
from tfx.components.base import executor_spec
from tfx.types import channel_utils
from tfx.types import component_spec
from tfx.types import node_common
from tfx.types import standard_artifacts
from tfx.utils import abc_utils

# Constants that used for serializing and de-serializing components.
_DRIVER_CLASS_KEY = 'driver_class'
_EXECUTOR_SPEC_KEY = 'executor_spec'
_INSTANCE_NAME_KEY = '_instance_name'
_SPEC_KEY = 'spec'


class BaseComponent(with_metaclass(abc.ABCMeta, base_node.BaseNode)):
  """Base class for a TFX pipeline component.

  An instance of a subclass of BaseComponent represents the parameters for a
  single execution of that TFX pipeline component.

  All subclasses of BaseComponent must override the SPEC_CLASS field with the
  ComponentSpec subclass that defines the interface of this component.

  Attributes:
    SPEC_CLASS: a subclass of types.ComponentSpec used by this component
      (required).
    EXECUTOR_SPEC: an instance of executor_spec.ExecutorSpec which describes how
      to execute this component (required).
    DRIVER_CLASS: a subclass of base_driver.BaseDriver as a custom driver for
      this component (optional, defaults to base_driver.BaseDriver).
  """

  # Subclasses must override this property (by specifying a types.ComponentSpec
  # class, e.g. "SPEC_CLASS = MyComponentSpec").
  SPEC_CLASS = abc_utils.abstract_property()
  # Subclasses must also override the executor spec.
  #
  # Note: EXECUTOR_CLASS has been replaced with EXECUTOR_SPEC. A custom
  # component's existing executor class definition "EXECUTOR_CLASS = MyExecutor"
  # should be replaced with "EXECUTOR_SPEC = ExecutorClassSpec(MyExecutor).
  EXECUTOR_SPEC = abc_utils.abstract_property()

  def __init__(
      self,
      spec: types.ComponentSpec,
      custom_executor_spec: Optional[executor_spec.ExecutorSpec] = None,
      instance_name: Optional[Text] = None):
    """Initialize a component.

    Args:
      spec: types.ComponentSpec object for this component instance.
      custom_executor_spec: Optional custom executor spec overriding the default
        executor specified in the component attribute.
      instance_name: Optional unique identifying name for this instance of the
        component in the pipeline. Required if two instances of the same
        component is used in the pipeline.
    """
    super(BaseComponent, self).__init__(instance_name)
    self.spec = spec
    if custom_executor_spec:
      if not isinstance(custom_executor_spec, executor_spec.ExecutorSpec):
        raise TypeError(
            ('Custom executor spec override %s for %s should be an instance of '
             'ExecutorSpec') % (custom_executor_spec, self.__class__))
    self.executor_spec = (custom_executor_spec or self.__class__.EXECUTOR_SPEC)
    self.driver_class = self.__class__.DRIVER_CLASS
    self._validate_component_class()
    self._validate_spec(spec)

  @classmethod
  def _validate_component_class(cls):
    """Validate that the SPEC_CLASSES property of this class is set properly."""
    if not (inspect.isclass(cls.SPEC_CLASS) and
            issubclass(cls.SPEC_CLASS, types.ComponentSpec)):
      raise TypeError(
          ('Component class %s expects SPEC_CLASS property to be a subclass '
           'of types.ComponentSpec; got %s instead.') % (cls, cls.SPEC_CLASS))
    if not isinstance(cls.EXECUTOR_SPEC, executor_spec.ExecutorSpec):
      raise TypeError((
          'Component class %s expects EXECUTOR_SPEC property to be an instance '
          'of ExecutorSpec; got %s instead.') % (cls, type(cls.EXECUTOR_SPEC)))
    if not (inspect.isclass(cls.DRIVER_CLASS) and
            issubclass(cls.DRIVER_CLASS, base_driver.BaseDriver)):
      raise TypeError(
          ('Component class %s expects DRIVER_CLASS property to be a subclass '
           'of base_driver.BaseDriver; got %s instead.') %
          (cls, cls.DRIVER_CLASS))

  def _validate_spec(self, spec):
    """Verify given spec is valid given the component's SPEC_CLASS."""
    if not isinstance(spec, types.ComponentSpec):
      raise ValueError((
          'BaseComponent (parent class of %s) expects "spec" argument to be an '
          'instance of types.ComponentSpec, got %s instead.') %
                       (self.__class__, spec))
    if not isinstance(spec, self.__class__.SPEC_CLASS):
      raise ValueError(
          ('%s expects the "spec" argument to be an instance of %s; '
           'got %s instead.') %
          (self.__class__, self.__class__.SPEC_CLASS, spec))

  def __repr__(self):
    return ('%s(spec: %s, executor_spec: %s, driver_class: %s, '
            'component_id: %s, inputs: %s, outputs: %s)') % (
                self.__class__.__name__, self.spec, self.executor_spec,
                self.driver_class, self.id, self.inputs, self.outputs)

  @property
  def inputs(self) -> node_common._PropertyDictWrapper:  # pylint: disable=protected-access
    return self.spec.inputs

  @property
  def outputs(self) -> node_common._PropertyDictWrapper:  # pylint: disable=protected-access
    return self.spec.outputs

  @property
  def exec_properties(self) -> Dict[Text, Any]:
    return self.spec.exec_properties

class _SimpleComponentMeta(abc.ABCMeta):
  """Metaclass for SimpleComponent."""

  def __init__(cls, *args):
    super(_SimpleComponentMeta, cls).__init__(*args)

    # Convert SimpleComponent parameter to ComponentSpec ones.
    new_inputs = {}
    for key, artifact_type in cls.INPUTS.items():
      assert issubclass(artifact_type, types.Artifact)
      new_inputs[key] = component_spec.ChannelParameter(type=artifact_type)
    new_outputs = {}
    for key, artifact_type in cls.OUTPUTS.items():
      assert issubclass(artifact_type, types.Artifact)
      new_outputs[key] = component_spec.ChannelParameter(type=artifact_type)


    # TODO
    new_parameters = {}

    cls.SPEC_CLASS = type(
        '%s_ComponentSpec' % cls.__name__,
        (types.ComponentSpec,),
        {
            'INPUTS': new_inputs,
            'OUTPUTS': new_outputs,
            'PARAMETERS': new_parameters,
        })

    class _MetaExecutor(base_executor.BaseExecutor):

      def Do(self, input_dict: Dict[Text, List[types.Artifact]],
             output_dict: Dict[Text, List[types.Artifact]],
             exec_properties: Dict[Text, Any]) -> None:
        cls._run(input_dict, output_dict, exec_properties)  # pylint: disable=protected-access

    cls.EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(
        executor_class=_MetaExecutor)


class ExecutionContext(object):
  def __init__(self, inputs, outputs):
    self.inputs = inputs
    self.outputs = outputs


class SimpleComponent(with_metaclass(_SimpleComponentMeta, BaseComponent)):

  # TODO: introduce frozendict.
  INPUTS = {}
  OUTPUTS = {}
  PARAMETERS = {}

  def __init__(self, **kwargs):
    spec_kwargs = {}
    unseen_args = set(kwargs.keys())
    for key, value in self.INPUTS.items():
      if key not in kwargs:
        raise ValueError(
          '%s expects input %r to be a Channel of type %s (got %s).' % (
            self.__class__.__name__, key, value, kwargs))
      spec_kwargs[key] = kwargs[key]
      unseen_args.remove(key)
    if unseen_args:
      raise ValueError(
        'Unknown arguments to %r: %s.' % (self.__class__.__name__, ', '.join(sorted(unseen_args))))
    for key, value in self.OUTPUTS.items():
      spec_kwargs[key] = channel_utils.as_channel([value()])
    spec = self.SPEC_CLASS(**spec_kwargs)
    super(SimpleComponent, self).__init__(spec)

  @classmethod
  def _run(
      cls: Type['SimpleComponent'],
      input_dict: Dict[Text, List[types.Artifact]],
      output_dict: Dict[Text, List[types.Artifact]],
      exec_properties: Dict[Text, Any]):
    input_mapping = cls._get_input_mapping()
    args = []
    for arg_type, arg_name in input_mapping:
      if arg_type == 'input_value':
        args.append(input_dict[arg_name][0].value)
    inputs = {}
    outputs = {}
    # print('input_dict', input_dict)
    # print('output_dict', output_dict)
    for key in cls.INPUTS:
      inputs[key] = input_dict[key][0]
    for key in cls.OUTPUTS:
      outputs[key] = output_dict[key][0]
    context = ExecutionContext(inputs, outputs)

    if getattr(cls, '_FUNCTION', None):
      return_value = cls._FUNCTION(*args)
    else:
      return_value = cls.execute(None, context, *args)

    if return_value:
      error = False
      if not isinstance(return_value, dict):
        error = True
      if not error:
        for key, value in return_value.items():
          if key in cls.OUTPUTS:
            context.outputs[key].value = value
          else:
            error = True
            break
      if error:
        raise Exception((
          'Return value from %s.execute() (if any) must be a dictionary '
          'with keys that are in the defined ValueArtifact OUTPUTS (got %s instead).') % (cls.__name__, return_value))

  @classmethod
  def _get_input_mapping(cls):
    function_mode = False
    if getattr(cls, '_FUNCTION', None):
      function_mode = True

    if function_mode:
      remaining_args = cls._FUNCTION_ARGS
    else:
      args, varargs, keywords, unused_defaults = inspect.getargspec(cls.execute)
      if varargs or keywords:
        raise ValueError(
            '*args or **kwargs arguments are not supported as '
            'SimpleComponent.execute() parameters.')

      # Validate
      total_count = len(cls.INPUTS) + len(cls.OUTPUTS) + len(cls.PARAMETERS)
      keys = set(itertools.chain(
          cls.INPUTS.keys(), cls.OUTPUTS.keys(), cls.PARAMETERS.keys()))
      if len(keys) != total_count:
        raise ValueError('keys must be distinct')

      # Ignore "self" arg.
      remaining_args = args[1:]
      if remaining_args[0] != 'context':
        raise ValueError(
            'First argument to SimpleComponent.execute() must be a "context" '
            'argument.')
      remaining_args = remaining_args[1:]

    input_mapping = []
    for arg in remaining_args:
      if isinstance(arg, list):
        raise ValueError(
            'Nested input parameters are not supported as '
            'SimpleComponent.execute() parameters.')
      if arg in cls.INPUTS:
        value = cls.INPUTS[arg]
        if not inspect.isclass(value) and issubclass(value, standard_artifacts.ValueArtifact):
          raise ValueError(
              'Parameters must be value artifacts (got %s instead).' % (value,))
        input_mapping.append(('input_value', arg))
      elif arg in cls.OUTPUTS:
        raise ValueError()
        # input_mapping.append(('output',))
      elif arg in cls.PARAMETERS:
        input_mapping.append(('parameter', arg))
      else:
        raise ValueError(
            'Unknown argument to SimpleComponent.execute(): %r.' % arg)

    return input_mapping


class ComponentOutput(object):
  def __init__(self, **kwargs):
    self.kwargs = kwargs

_PRIMITIVE_TO_ARTIFACT = {
  int: standard_artifacts.Integer,
  Text: standard_artifacts.String,
}



def component_from_typehints(func):
  typehints = typing.get_type_hints(func)
  if not isinstance(typehints['return'], ComponentOutput):
    raise ValueError(
      'Function decorated with @component_from_typehints must have '
      'return typehint as a ComponentOutput instance.')

  # print('TYPE_HINTS', typehints)
  argspec = inspect.getfullargspec(func)
  args, varargs, keywords, defaults = argspec.args, argspec.varargs, argspec.varkw, argspec.defaults
  if varargs or keywords:
    raise ValueError(
        '*args or **kwargs arguments are not supported as '
        'SimpleComponent.execute() parameters.')
  if defaults:
    raise ValueError(
        'Currently, optional arguments are not supported as '
        'SimpleComponent.execute() parameters.')
  argtypes = {}
  inputs = {}
  for arg in args:
    if isinstance(arg, list):
      raise ValueError(
          'Nested input parameters are not supported as '
          'SimpleComponent.execute() parameters.')
    if arg not in typehints:
      raise ValueError(
          'All input arguments to a function decorated with '
          '@component_from_typehints must have typehints.')
    arg_typehint = typehints[arg] 
    if isinstance(arg_typehint, types.Artifact):
      # TODO
      raise ValueError()
    elif arg_typehint in _PRIMITIVE_TO_ARTIFACT:
      argtype = _PRIMITIVE_TO_ARTIFACT[arg_typehint]
    else:
      # print('arg_typehint', arg_typehint)
      raise ValueError()
    argtypes[arg] = argtype
    inputs[arg] = argtype

  outputs = {}
  for arg, arg_typehint in typehints['return'].kwargs.items():
    if isinstance(arg_typehint, types.Artifact):
      # TODO
      raise ValueError()
    elif arg_typehint in _PRIMITIVE_TO_ARTIFACT:
      argtype = _PRIMITIVE_TO_ARTIFACT[arg_typehint]
    else:
      # print('arg_typehint', arg_typehint)
      raise ValueError()
    outputs[arg] = argtype

  return type(
    '%s_Component' % func.__name__,
    (SimpleComponent,),
    {
      'INPUTS': inputs,
      'OUTPUTS': outputs,
      '_FUNCTION_ARGS': args,
      '_FUNCTION_ARGTYPES': argtypes,
      '_FUNCTION': func,
    })


