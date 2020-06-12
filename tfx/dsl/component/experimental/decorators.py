# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Decorators for defining components via Python functions.

Experimental: no backwards compatibility guarantees.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import types
from typing import Any, Callable, Dict, List, Text

# Standard Imports

import six

from tfx import types as tfx_types
from tfx.components.base import base_component
from tfx.components.base import base_executor
from tfx.components.base import executor_spec
from tfx.dsl.component.experimental import function_parser
from tfx.types import channel_utils
from tfx.types import component_spec


class _SimpleComponent(base_component.BaseComponent):
  """Component whose constructor generates spec instance from arguments."""

  def __init__(self, *unused_args, **kwargs):
    if unused_args:
      raise ValueError(('%s expects arguments to be passed as keyword '
                        'arguments') % (self.__class__.__name__,))
    spec_kwargs = {}
    unseen_args = set(kwargs.keys())
    for key, channel_parameter in self.SPEC_CLASS.INPUTS.items():
      if key not in kwargs and not channel_parameter.optional:
        raise ValueError('%s expects input %r to be a Channel of type %s.' %
                         (self.__class__.__name__, key, channel_parameter.type))
      if key in kwargs:
        spec_kwargs[key] = kwargs[key]
        unseen_args.remove(key)
    for key, parameter in self.SPEC_CLASS.PARAMETERS.items():
      if key not in kwargs and not parameter.optional:
        raise ValueError('%s expects parameter %r of type %s.' %
                         (self.__class__.__name__, key, parameter.type))
      if key in kwargs:
        spec_kwargs[key] = kwargs[key]
        unseen_args.remove(key)
    instance_name = kwargs.get('instance_name', None)
    unseen_args.discard('instance_name')
    if unseen_args:
      raise ValueError(
          'Unknown arguments to %r: %s.' %
          (self.__class__.__name__, ', '.join(sorted(unseen_args))))
    for key, channel_parameter in self.SPEC_CLASS.OUTPUTS.items():
      spec_kwargs[key] = channel_utils.as_channel([channel_parameter.type()])
    spec = self.SPEC_CLASS(**spec_kwargs)
    super(_SimpleComponent, self).__init__(spec, instance_name=instance_name)


class _FunctionExecutor(base_executor.BaseExecutor):
  """Base class for function-based executors."""

  # Properties that should be overridden by subclass. Defaults are provided to
  # allow pytype to properly type check these properties.

  # Describes the format of each argument passed to the component function, as
  # a dictionary from name to a `function_parser.ArgFormats` enum value.
  _ARG_FORMATS = {}
  # Map from names of optional arguments to their default argument values.
  _ARG_DEFAULTS = {}
  # User-defined component function. Should be wrapped in staticmethod() to
  # avoid being interpreted as a bound method (i.e. one taking `self` as its
  # first argument.
  _FUNCTION = staticmethod(lambda: None)
  # Set of output names that are primitive type values returned from the user
  # function.
  _RETURNED_VALUES = set()

  def Do(self, input_dict: Dict[Text, List[tfx_types.Artifact]],
         output_dict: Dict[Text, List[tfx_types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    function_args = {}
    for name, arg_format in self._ARG_FORMATS.items():
      if arg_format == function_parser.ArgFormats.INPUT_ARTIFACT:
        input_list = input_dict.get(name, [])
        if len(input_list) == 1:
          function_args[name] = input_list[0]
        elif not input_list and name in self._ARG_DEFAULTS:
          # Do not pass the missing optional input.
          pass
        else:
          raise ValueError((
              'Expected input %r to %s to be a singleton ValueArtifact channel '
              '(got %s instead).') % (name, self, input_list))
      elif arg_format == function_parser.ArgFormats.OUTPUT_ARTIFACT:
        output_list = output_dict.get(name, [])
        if len(output_list) == 1:
          function_args[name] = output_list[0]
        else:
          raise ValueError((
              'Expected output %r to %s to be a singleton ValueArtifact channel '
              '(got %s instead).') % (name, self, output_list))
      elif arg_format == function_parser.ArgFormats.ARTIFACT_VALUE:
        input_list = input_dict.get(name, [])
        if len(input_list) == 1:
          function_args[name] = input_list[0].value
        elif not input_list and name in self._ARG_DEFAULTS:
          # Do not pass the missing optional input.
          pass
        else:
          raise ValueError((
              'Expected input %r to %s to be a singleton ValueArtifact channel '
              '(got %s instead).') % (name, self, input_list))
      elif arg_format == function_parser.ArgFormats.PARAMETER:
        if name in exec_properties:
          function_args[name] = exec_properties[name]
        elif name in self._ARG_DEFAULTS:
          # Do not pass the missing optional input.
          pass
        else:
          raise ValueError((
              'Expected non-optional parameter %r of %s to be provided, but no '
              'value was passed.') % (name, self))
      else:
        raise ValueError('Unknown argument format: %r' % (arg_format,))

    # Call function and check returned values.
    outputs = self._FUNCTION(**function_args)
    outputs = outputs or {}
    if not isinstance(outputs, dict):
      raise ValueError(
          ('Expected component executor function %s to return a dict of '
           'outputs (got %r instead).') % (self._FUNCTION, outputs))

    # Assign returned ValueArtifact values.
    for name in self._RETURNED_VALUES:
      if name not in outputs:
        raise ValueError(
            'Did not receive expected output %r as return value from '
            'component executor function %s.' % (name, self._FUNCTION))
      try:
        output_dict[name][0].value = outputs[name]
      except TypeError:
        raise TypeError(
            ('Return value %r for output %r is incompatible with output type '
             '%r.') % (outputs[name], name, output_dict[name][0].__class__))


def component(func: types.FunctionType) -> Callable[..., Any]:
  """Decorator: creates a component from a typehint-annotated Python function.

  This decorator creates a component based on typehint annotations specified for
  the arguments and return value for a Python function. Specifically, function
  arguments can be annotated with the following types and associated semantics:

  * `Parameter[T]` where `T` is `int`, `float`, `str`, or `bytes`: indicates
    that a primitive type execution parameter, whose value is known at pipeline
    construction time, will be passed for this argument. These parameters will
    be recorded in ML Metadata as part of the component's execution record. Can
    be an optional argument.
  * `int`, `float`, `str`, `bytes`: indicates that a primitive type value will
    be passed for this argument. This value is tracked as an `Integer`, `Float`
    `String` or `Bytes` artifact (see `tfx.types.standard_artifacts`) whose
    value is read and passed into the given Python component function. Can be
    an optional argument.
  * `InputArtifact[ArtifactType]`: indicates that an input artifact object of
    type `ArtifactType` (deriving from `tfx.types.Artifact`) will be passed for
    this argument. This artifact is intended to be consumed as an input by this
    component (possibly reading from the path specified by its `.uri`). Can be
    an optional argument by specifying a default value of `None`.
  * `OutputArtifact[ArtifactType]`: indicates that an output artifact object of
    type `ArtifactType` (deriving from `tfx.types.Artifact`) will be passed for
    this argument. This artifact is intended to be emitted as an output by this
    component (and written to the path specified by its `.uri`). Cannot be an
    optional argument.

  The return value typehint should be either empty or `None`, in the case of a
  component function that has no return values, or an instance of
  `OutputDict(key_1=type_1, ...)`, where each key maps to a given type (each
  type is a primitive value type, i.e. `int`, `float`, `str` or `bytes`), to
  indicate that the return value is a dictionary with specified keys and value
  types.

  Note that output artifacts should not be included in the return value
  typehint; they should be included as `OutputArtifact` annotations in the
  function inputs, as described above.

  The function to which this decorator is applied must be at the top level of
  its Python module (it may not be defined within nested classes or function
  closures).

  This is example usage of component definition using this decorator:

      from tfx.components.base.annotations import OutputDict
      from tfx.components.base.annotations import
      InputArtifact
      from tfx.components.base.annotations import
      OutputArtifact
      from tfx.components.base.annotations import
      Parameter
      from tfx.components.base.decorators import component
      from tfx.types.standard_artifacts import Examples
      from tfx.types.standard_artifacts import Model

      @component
      def MyTrainerComponent(
          training_data: InputArtifact[Examples],
          model: OutputArtifact[Model],
          dropout_hyperparameter: float,
          num_iterations: Parameter[int] = 10
          ) -> OutputDict(loss=float, accuracy=float):
        '''My simple trainer component.'''

        records = read_examples(training_data.uri)
        model_obj = train_model(records, num_iterations, dropout_hyperparameter)
        model_obj.write_to(model.uri)

        return {
          'loss': model_obj.loss,
          'accuracy': model_obj.accuracy
        }

      # Example usage in a pipeline graph definition:
      # ...
      trainer = MyTrainerComponent(
          examples=example_gen.outputs['examples'],
          dropout_hyperparameter=other_component.outputs['dropout'],
          num_iterations=1000)
      pusher = Pusher(model=trainer.outputs['model'])
      # ...

  Experimental: no backwards compatibility guarantees.

  Args:
    func: Typehint-annotated component executor function.

  Returns:
    `base_component.BaseComponent` subclass for the given component executor
    function.

  Raises:
    EnvironmentError: if the current Python interpreter is not Python 3.
  """
  if six.PY2:
    raise EnvironmentError('`@component` is only supported in Python 3.')

  # Defining a component within a nested class or function closure causes
  # problems because in this case, the generated component classes can't be
  # referenced via their qualified module path.
  #
  # See https://www.python.org/dev/peps/pep-3155/ for details about the special
  # '<locals>' namespace marker.
  if '<locals>' in func.__qualname__.split('.'):
    raise ValueError(
        'The @component decorator can only be applied to a function defined '
        'at the module level. It cannot be used to construct a component for a '
        'function defined in a nested class or function closure.')

  inputs, outputs, parameters, arg_formats, arg_defaults, returned_values = (
      function_parser.parse_typehint_component_function(func))

  spec_inputs = {}
  spec_outputs = {}
  spec_parameters = {}
  for key, artifact_type in inputs.items():
    spec_inputs[key] = component_spec.ChannelParameter(
        type=artifact_type, optional=(key in arg_defaults))
  for key, artifact_type in outputs.items():
    assert key not in arg_defaults, 'Optional outputs are not supported.'
    spec_outputs[key] = component_spec.ChannelParameter(type=artifact_type)
  for key, primitive_type in parameters.items():
    spec_parameters[key] = component_spec.ExecutionParameter(
        type=primitive_type, optional=(key in arg_defaults))
  component_spec_class = type(
      '%s_Spec' % func.__name__, (tfx_types.ComponentSpec,), {
          'INPUTS': spec_inputs,
          'OUTPUTS': spec_outputs,
          'PARAMETERS': spec_parameters,
      })

  executor_class = type(
      '%s_Executor' % func.__name__,
      (_FunctionExecutor,),
      {
          '_ARG_FORMATS': arg_formats,
          '_ARG_DEFAULTS': arg_defaults,
          # The function needs to be marked with `staticmethod` so that later
          # references of `self._FUNCTION` do not result in a bound method (i.e.
          # one with `self` as its first parameter).
          '_FUNCTION': staticmethod(func),
          '_RETURNED_VALUES': returned_values,
          '__module__': func.__module__,
      })

  # Expose the generated executor class in the same module as the decorated
  # function. This is needed so that the executor class can be accessed at the
  # proper module path. One place this is needed is in the Dill pickler used by
  # Apache Beam serialization.
  module = sys.modules[func.__module__]
  setattr(module, '%s_Executor' % func.__name__, executor_class)

  executor_spec_instance = executor_spec.ExecutorClassSpec(
      executor_class=executor_class)

  return type(
      func.__name__, (_SimpleComponent,), {
          'SPEC_CLASS': component_spec_class,
          'EXECUTOR_SPEC': executor_spec_instance,
          '__module__': func.__module__,
      })
