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

import copy
import functools
import sys
import types
from typing import Any, Dict, List, Optional, Type

from tfx import types as tfx_types
from tfx.dsl.component.experimental import function_parser
from tfx.dsl.component.experimental import json_compat
from tfx.dsl.components.base import base_beam_component
from tfx.dsl.components.base import base_beam_executor
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.types import channel
from tfx.types import component_spec
from tfx.types import system_executions

try:
  import apache_beam as beam  # pytype: disable=import-error  # pylint: disable=g-import-not-at-top
  _BeamPipeline = beam.Pipeline
except ModuleNotFoundError:
  beam = None
  _BeamPipeline = Any


def _extract_func_args(
    obj: str,
    arg_formats: Dict[str, int],
    arg_defaults: Dict[str, Any],
    input_dict: Dict[str, List[tfx_types.Artifact]],
    output_dict: Dict[str, List[tfx_types.Artifact]],
    exec_properties: Dict[str, Any],
    beam_pipeline: Optional[_BeamPipeline] = None,
) -> Dict[str, Any]:
  """Extracts function arguments for the decorated function."""
  result = {}
  for name, arg_format in arg_formats.items():
    if arg_format == function_parser.ArgFormats.INPUT_ARTIFACT:
      input_list = input_dict.get(name, [])
      if len(input_list) == 1:
        result[name] = input_list[0]
      elif not input_list and name in arg_defaults:
        # Do not pass the missing optional input.
        pass
      else:
        raise ValueError(
            ('Expected input %r to %s to be a singleton ValueArtifact channel '
             '(got %s instead).') % (name, obj, input_list))
    elif arg_format == function_parser.ArgFormats.OUTPUT_ARTIFACT:
      output_list = output_dict.get(name, [])
      if len(output_list) == 1:
        result[name] = output_list[0]
      else:
        raise ValueError(
            ('Expected output %r to %s to be a singleton ValueArtifact channel '
             '(got %s instead).') % (name, obj, output_list))
    elif arg_format == function_parser.ArgFormats.ARTIFACT_VALUE:
      input_list = input_dict.get(name, [])
      if len(input_list) == 1:
        result[name] = input_list[0].value
      elif not input_list and name in arg_defaults:
        # Do not pass the missing optional input.
        pass
      else:
        raise ValueError(
            ('Expected input %r to %s to be a singleton ValueArtifact channel '
             '(got %s instead).') % (name, obj, input_list))
    elif arg_format == function_parser.ArgFormats.PARAMETER:
      if name in exec_properties:
        result[name] = exec_properties[name]
      elif name in arg_defaults:
        # Do not pass the missing optional input.
        pass
      else:
        raise ValueError(
            ('Expected non-optional parameter %r of %s to be provided, but no '
             'value was passed.') % (name, obj))
    elif arg_format == function_parser.ArgFormats.BEAM_PARAMETER:
      result[name] = beam_pipeline
      if name in arg_defaults and arg_defaults[name] is not None:
        raise ValueError('beam Pipeline parameter does not allow default ',
                         'value other than None.')
    else:
      raise ValueError('Unknown argument format: %r' % (arg_format,))
  return result


def _assign_returned_values(
    function,
    outputs: Dict[str, Any],
    returned_values: Dict[str, Any],
    output_dict: Dict[str, List[tfx_types.Artifact]],
    json_typehints: Dict[str, Type],  # pylint: disable=g-bare-generic
) -> Dict[str, List[tfx_types.Artifact]]:
  """Validates and assigns the outputs to the output_dict."""
  result = copy.deepcopy(output_dict)
  if not isinstance(outputs, dict):
    raise ValueError(
        ('Expected component executor function %s to return a dict of '
         'outputs (got %r instead).') % (function, outputs))

  # Assign returned ValueArtifact values.
  for name, is_optional in returned_values.items():
    if name not in outputs:
      raise ValueError(
          'Did not receive expected output %r as return value from '
          'component executor function %s.' % (name, function))
    if not is_optional and outputs[name] is None:
      raise ValueError('Non-nullable output %r received None return value from '
                       'component executor function %s.' % (name, function))
    try:
      result[name][0].value = outputs[name]
    except TypeError as e:
      raise TypeError(
          ('Return value %r for output %r is incompatible with output type '
           '%r.') %
          (outputs[name], name, result[name][0].__class__)) from e
    # Handle JsonValue runtime type check.
    if name in json_typehints:
      ret = json_compat.check_strict_json_compat(outputs[name],
                                                 json_typehints[name])
      if not ret:
        raise TypeError(
            ('Return value %r for output %r is incompatible with output type '
             '%r.') % (outputs[name], name, json_typehints[name]))
  return result


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
    if unseen_args:
      raise ValueError(
          'Unknown arguments to %r: %s.' %
          (self.__class__.__name__, ', '.join(sorted(unseen_args))))
    for key, channel_parameter in self.SPEC_CLASS.OUTPUTS.items():
      artifact = channel_parameter.type()
      spec_kwargs[key] = channel.OutputChannel(artifact.type, self,
                                               key).set_artifacts([artifact])
      json_compat_typehint = getattr(channel_parameter, '_JSON_COMPAT_TYPEHINT',
                                     None)
      if json_compat_typehint:
        setattr(spec_kwargs[key], '_JSON_COMPAT_TYPEHINT', json_compat_typehint)
    spec = self.SPEC_CLASS(**spec_kwargs)
    super().__init__(spec)
    # Set class name, which is the decorated function name, as the default id.
    # It can be overwritten by the user.
    self._id = self.__class__.__name__


class _SimpleBeamComponent(_SimpleComponent,
                           base_beam_component.BaseBeamComponent):
  """Component whose constructor generates spec instance from arguments."""
  pass


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
  # Dictionary mapping output names that are primitive type values returned from
  # the user function to whether they are optional (and thus has a nullable
  # return value).
  _RETURNED_VALUES = {}
  # A dictionary mapping output names that are declared
  # as json compatible types to the annotation.
  _RETURN_JSON_COMPAT_TYPEHINT = {}

  def Do(self, input_dict: Dict[str, List[tfx_types.Artifact]],
         output_dict: Dict[str, List[tfx_types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    function_args = _extract_func_args(
        obj=str(self),
        arg_formats=self._ARG_FORMATS,
        arg_defaults=self._ARG_DEFAULTS,
        input_dict=input_dict,
        output_dict=output_dict,
        exec_properties=exec_properties)

    # Call function and check returned values.
    outputs = self._FUNCTION(**function_args)
    outputs = outputs or {}
    output_dict.update(
        _assign_returned_values(
            function=self._FUNCTION,
            outputs=outputs,
            returned_values=self._RETURNED_VALUES,
            output_dict=output_dict,
            json_typehints=self._RETURN_JSON_COMPAT_TYPEHINT,
        ))


class _FunctionBeamExecutor(base_beam_executor.BaseBeamExecutor,
                            _FunctionExecutor):
  """Base class for function-based executors."""

  def Do(self, input_dict: Dict[str, List[tfx_types.Artifact]],
         output_dict: Dict[str, List[tfx_types.Artifact]],
         exec_properties: Dict[str, Any]) -> None:
    function_args = _extract_func_args(
        obj=str(self),
        arg_formats=self._ARG_FORMATS,
        arg_defaults=self._ARG_DEFAULTS,
        input_dict=input_dict,
        output_dict=output_dict,
        exec_properties=exec_properties,
        beam_pipeline=self._make_beam_pipeline())

    # Call function and check returned values.
    outputs = self._FUNCTION(**function_args)
    outputs = outputs or {}
    output_dict.update(
        _assign_returned_values(
            function=self._FUNCTION,
            outputs=outputs,
            returned_values=self._RETURNED_VALUES,
            output_dict=output_dict,
            json_typehints=self._RETURN_JSON_COMPAT_TYPEHINT,
        ))


def component(
    func: Optional[types.FunctionType] = None,
    component_annotation: Optional[Type[
        system_executions.SystemExecution]] = None,
    use_beam: bool = False,
) -> Any:
  """Decorator: creates a component from a typehint-annotated Python function.

  This decorator creates a component based on typehint annotations specified for
  the arguments and return value for a Python function. The decorator can be
  supplied with a parameter `component_annotation` to specify the annotation for
  this component decorator. This annotation hints which system execution type
  this python function-based component belongs to.
  Specifically, function arguments can be annotated with the following types and
  associated semantics:

  * `Parameter[T]` where `T` is `int`, `float`, `str`, or `bool`:
    indicates that a primitive type execution parameter, whose value is known at
    pipeline construction time, will be passed for this argument. These
    parameters will be recorded in ML Metadata as part of the component's
    execution record. Can be an optional argument.
  * `int`, `float`, `str`, `bytes`, `bool`, `Dict`, `List`: indicates that a
    primitive type value will be passed for this argument. This value is tracked
    as an `Integer`, `Float`, `String`, `Bytes`, `Boolean` or `JsonValue`
    artifact (see `tfx.types.standard_artifacts`) whose value is read and passed
    into the given Python component function. Can be an optional argument.
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
  type is a primitive value type, i.e. `int`, `float`, `str`, `bytes`, `bool`
  `Dict` or  `List`; or `Optional[T]`, where T is a primitive type value, in
  which case `None` can be returned), to indicate that the return value is a
  dictionary with specified keys and value types.

  Note that output artifacts should not be included in the return value
  typehint; they should be included as `OutputArtifact` annotations in the
  function inputs, as described above.

  The function to which this decorator is applied must be at the top level of
  its Python module (it may not be defined within nested classes or function
  closures).

  This is example usage of component definition using this decorator:

      from tfx.dsl.components.base.annotations import OutputDict
      from tfx.dsl.components.base.annotations import
      InputArtifact
      from tfx.dsl.components.base.annotations import
      OutputArtifact
      from tfx.dsl.components.base.annotations import
      Parameter
      from tfx.dsl.components.base.decorators import component
      from tfx.types import standard_artifacts
      from tfx.types import system_executions

      @component(component_annotation=system_executions.Train)
      def MyTrainerComponent(
          training_data: InputArtifact[standard_artifacts.Examples],
          model: OutputArtifact[standard_artifacts.Model],
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
          training_data=example_gen.outputs['examples'],
          dropout_hyperparameter=other_component.outputs['dropout'],
          num_iterations=1000)
      pusher = Pusher(model=trainer.outputs['model'])
      # ...

  When the parameter `component_annotation` is not supplied, the default value
  is None. This is another example usage with `component_annotation` = None:

      @component
      def MyTrainerComponent(
          training_data: InputArtifact[standard_artifacts.Examples],
          model: OutputArtifact[standard_artifacts.Model],
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

  When the parameter `use_beam` is True, one of the parameters of the decorated
  function type-annotated by BeamComponentParameter[beam.Pipeline] and the
  default value can only be None. It will be replaced by a beam Pipeline made
  with the tfx pipeline's beam_pipeline_args that's shared with other beam-based
  components:

      @component(use_beam=True)
      def DataProcessingComponent(
          input_examples: InputArtifact[standard_artifacts.Examples],
          output_examples: OutputArtifact[standard_artifacts.Examples],
          beam_pipeline: BeamComponentParameter[beam.Pipeline] = None,
          ) -> None:
        '''My simple trainer component.'''

        records = read_examples(training_data.uri)
        with beam_pipeline as p:
          ...

  Experimental: no backwards compatibility guarantees.

  Args:
    func: Typehint-annotated component executor function.
    component_annotation: used to annotate the python function-based component.
      It is a subclass of SystemExecution from
      third_party/py/tfx/types/system_executions.py; it can be None.
    use_beam: Whether to create a component that is a subclass of
      BaseBeamComponent. This allows a beam.Pipeline to be made with
      tfx-pipeline-wise beam_pipeline_args.

  Returns:
    `base_component.BaseComponent` or `base_component.BaseBeamComponent`
      subclass for the given component executor function.

  Raises:
    EnvironmentError: if the current Python interpreter is not Python 3.
  """
  if func is None:
    return functools.partial(
        component, component_annotation=component_annotation, use_beam=use_beam)

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

  (inputs, outputs, parameters, arg_formats, arg_defaults, returned_values,
   json_typehints, return_json_typehints) = (
       function_parser.parse_typehint_component_function(func))
  if use_beam and list(parameters.values()).count(_BeamPipeline) != 1:
    raise ValueError('The decorated function must have one and only one '
                     'optional parameter of type '
                     'BeamComponentParameter[beam.Pipeline] with '
                     'default value None when use_beam=True.')

  spec_inputs = {}
  spec_outputs = {}
  spec_parameters = {}
  for key, artifact_type in inputs.items():
    spec_inputs[key] = component_spec.ChannelParameter(
        type=artifact_type, optional=(key in arg_defaults))
    if key in json_typehints:
      setattr(spec_inputs[key], '_JSON_COMPAT_TYPEHINT', json_typehints[key])
  for key, artifact_type in outputs.items():
    assert key not in arg_defaults, 'Optional outputs are not supported.'
    spec_outputs[key] = component_spec.ChannelParameter(type=artifact_type)
    if key in return_json_typehints:
      setattr(spec_outputs[key], '_JSON_COMPAT_TYPEHINT',
              return_json_typehints[key])
  for key, primitive_type in parameters.items():
    spec_parameters[key] = component_spec.ExecutionParameter(
        type=primitive_type, optional=(key in arg_defaults))
  component_spec_class = type(
      '%s_Spec' % func.__name__, (tfx_types.ComponentSpec,), {
          'INPUTS': spec_inputs,
          'OUTPUTS': spec_outputs,
          'PARAMETERS': spec_parameters,
          'TYPE_ANNOTATION': component_annotation,
      })

  executor_class = type(
      '%s_Executor' % func.__name__,
      (_FunctionBeamExecutor if use_beam else _FunctionExecutor,),
      {
          '_ARG_FORMATS': arg_formats,
          '_ARG_DEFAULTS': arg_defaults,
          # The function needs to be marked with `staticmethod` so that later
          # references of `self._FUNCTION` do not result in a bound method (i.e.
          # one with `self` as its first parameter).
          '_FUNCTION': staticmethod(func),  # pytype: disable=not-callable
          '_RETURNED_VALUES': returned_values,
          '_RETURN_JSON_COMPAT_TYPEHINT': return_json_typehints,
          '__module__': func.__module__,
      })

  # Expose the generated executor class in the same module as the decorated
  # function. This is needed so that the executor class can be accessed at the
  # proper module path. One place this is needed is in the Dill pickler used by
  # Apache Beam serialization.
  module = sys.modules[func.__module__]
  setattr(module, '%s_Executor' % func.__name__, executor_class)

  executor_spec_class = (
      executor_spec.BeamExecutorSpec
      if use_beam else executor_spec.ExecutorClassSpec)
  executor_spec_instance = executor_spec_class(executor_class=executor_class)

  return type(
      func.__name__, (_SimpleBeamComponent if use_beam else _SimpleComponent,),
      {
          'SPEC_CLASS': component_spec_class,
          'EXECUTOR_SPEC': executor_spec_instance,
          '__module__': func.__module__,
          'test_call': staticmethod(func),  # pytype: disable=not-callable
      })
