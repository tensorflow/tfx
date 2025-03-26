# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Utils for Python function decorator."""

import enum
import inspect
import sys
import types
from typing import Any, Dict, Optional, Type
from tfx import types as tfx_types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import base_executor
from tfx.dsl.components.base import executor_spec
from tfx.types import artifact
from tfx.types import component_spec
from tfx.types import system_executions
from google.protobuf import message


class ArgFormats(enum.Enum):
  INPUT_ARTIFACT = 1
  OUTPUT_ARTIFACT = 2
  ARTIFACT_VALUE = 3
  PARAMETER = 4
  BEAM_PARAMETER = 5
  LIST_INPUT_ARTIFACTS = 6
  PREVIOUS_OUTPUT_ARTIFACTS = 7


def assert_is_functype(func: Any) -> None:
  """Asserts func is an instance of FunctionType.

  Args:
    func: The function to be checked.

  Raises:
    ValueError: if func is not FunctionType.
  """
  if not isinstance(func, types.FunctionType):
    raise ValueError(
        f'Expected a typehint-annotated Python function (got {func} instead).'
    )


def assert_no_varargs_varkw(
    argspec: inspect.FullArgSpec,
    subject_message: Optional[str] = None,
) -> None:
  """Asserts no arbitrary positional arguments (*args) and no arbitrary keyword arguments (**kwargs).

  Args:
    argspec: An `inspect.FullArgSpec` instance describing the function. Usually
      obtained from `inspect.getfullargspec(func)`.
    subject_message: An error message subject.

  Raises:
    ValueError if either arbitrary position arguments or arbitrary keywords
    arguments are found.
  """
  if argspec.varargs or argspec.varkw:
    msg_prefix = subject_message if subject_message else 'The function'
    raise ValueError(
        f'{msg_prefix} does not support *args or **kwargs arguments.'
    )


def extract_arg_defaults(argspec: inspect.FullArgSpec) -> Dict[str, Any]:
  """Extracts arg default values as a dict from arg name to its default value.

  Args:
    argspec: An `inspect.FullArgSpec` instance describing the function. Usually
      obtained from `inspect.getfullargspec(func)`.

  Returns:
    A dict from arg name to its default value. Args without default values are
    omitted in the dict.
  """
  arg_defaults = {}
  if argspec.defaults:
    arg_defaults = dict(
        zip(argspec.args[-len(argspec.defaults) :], argspec.defaults)
    )
  return arg_defaults


def parse_parameter_arg(
    arg: str,
    arg_defaults: Dict[str, Any],
    arg_typehint: Any,
    parameters: Dict[str, Any],
    arg_formats: Dict[str, ArgFormats],
) -> None:
  """Parses and validates a function arg annotated with Parameter[T].

  Args:
    arg: The arg name.
    arg_defaults: A dict from arg name to its default value.
    arg_typehint: The typehint for the arg.
    parameters: An updated dict from arg name to its unwrapped primitive type.
      Note that the parameter arg will be added to the dict if it's in the valid
      format and existing entries won't be changed.
    arg_formats: An updated dict from arg name to its format. The format is the
      value of `ArgFormats` enum). Note that the parameter arg will be added to
      the dict if it's in the valid format and existing entries won't be
      changed.
  """
  if arg in arg_defaults:
    if not (
        arg_defaults[arg] is None
        or isinstance(arg_defaults[arg], arg_typehint.type)
    ):
      raise ValueError(
          f'The default value for optional parameter {arg} on function must be '
          f'an instance of its declared type {arg_typehint.type} or `None`, '
          f'got {arg_defaults[arg]}'
      )
  parameters[arg] = arg_typehint.type
  arg_formats[arg] = ArgFormats.PARAMETER


def assert_is_top_level_func(func: types.FunctionType) -> None:
  """Asserts the func is a top level function.

  This check is needed because defining components within a nested class or
  function closure prevents the generated component classes to be referenced by
  their qualified module path.

  Args:
    func: The function to be checked.

  Raises:
    ValueError if the func is not defined at top level.
  """
  # See https://www.python.org/dev/peps/pep-3155/ for details about the special
  # '<locals>' namespace marker.
  if '<locals>' in func.__qualname__.split('.'):
    raise ValueError(
        'The decorator can only be applied to a function defined at the module'
        ' level. It cannot be used to construct a component for a function'
        ' defined in a nested class or function closure.'
    )


def assert_no_private_func_in_main(func: types.FunctionType) -> None:
  """Asserts the func is not a private function in the main file.


  Args:
    func: The function to be checked.

  Raises:
    ValueError if the func was defined in main and whose name starts with '_'.
  """
  if func.__module__ == '__main__' and func.__name__.startswith('_'):
    raise ValueError(
        'Custom Python functions (both @component and pre/post hooks) declared'
        ' in the main file must be public. Please remove the leading'
        f' underscore from {func.__name__}.'
    )


def _create_component_spec_class(
    func: types.FunctionType,
    arg_defaults: Dict[str, Any],
    inputs: Optional[Dict[str, Type[artifact.Artifact]]] = None,
    outputs: Optional[Dict[str, Type[artifact.Artifact]]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    type_annotation: Optional[Type[system_executions.SystemExecution]] = None,
    json_compatible_inputs: Optional[Dict[str, Any]] = None,
    json_compatible_outputs: Optional[Dict[str, Any]] = None,
) -> Type[tfx_types.ComponentSpec]:
  """Creates the ComponentSpec class for the func-generated component.

  Args:
    func: The function which component will be generated based on.
    arg_defaults: A dict from func arg name to its default value.
    inputs: A dict from input name to its Artifact type.
    outputs: A dict from output name to its Artifact type.
    parameters: A dict from parameter name to its primitive type.
    type_annotation: a subclass to SystemExecution used to annotate the
      component on ComponentSpec.
    json_compatible_inputs: A dict from input names that have json compatible
      types to their typehints. Json compatibility is determined by
      `tfx.dsl.component.experimental.json_compat.is_json_compatible`.
    json_compatible_outputs: A dict from output names that have json compatible
      types to their typehints. Json compatibility is determined by
      `tfx.dsl.component.experimental.json_compat.is_json_compatible`.

  Returns:
    a subclass of ComponentSpec.
  """
  spec_inputs = {}
  spec_outputs = {}
  spec_parameters = {}
  if inputs:
    for key, artifact_type in inputs.items():
      spec_inputs[key] = component_spec.ChannelParameter(
          type=artifact_type, optional=(key in arg_defaults)
      )
      if json_compatible_inputs and key in json_compatible_inputs:
        setattr(
            spec_inputs[key],
            '_JSON_COMPAT_TYPEHINT',
            json_compatible_inputs[key],
        )
  if outputs:
    for key, artifact_type in outputs.items():
      assert key not in arg_defaults, 'Optional outputs are not supported.'
      spec_outputs[key] = component_spec.ChannelParameter(type=artifact_type)
      if json_compatible_outputs and key in json_compatible_outputs:
        setattr(
            spec_outputs[key],
            '_JSON_COMPAT_TYPEHINT',
            json_compatible_outputs[key],
        )
  if parameters:
    for key, param_type in parameters.items():
      if inspect.isclass(param_type) and issubclass(
          param_type, message.Message
      ):
        spec_parameters[key] = component_spec.ExecutionParameter(
            type=param_type, optional=(key in arg_defaults), use_proto=True
        )
      else:
        spec_parameters[key] = component_spec.ExecutionParameter(
            type=param_type, optional=(key in arg_defaults)
        )
  component_spec_class = type(
      '%s_Spec' % func.__name__,
      (tfx_types.ComponentSpec,),
      {
          'INPUTS': spec_inputs,
          'OUTPUTS': spec_outputs,
          'PARAMETERS': spec_parameters,
          'TYPE_ANNOTATION': type_annotation,
      },
  )
  return component_spec_class


def _create_executor_spec_instance(
    func: types.FunctionType,
    base_executor_class: Type[base_executor.BaseExecutor],
    executor_spec_class: Type[executor_spec.ExecutorClassSpec],
    arg_formats: Dict[str, ArgFormats],
    arg_defaults: Dict[str, Any],
    return_values_optionality: Optional[Dict[str, bool]] = None,
    json_compatible_outputs: Optional[Dict[str, Any]] = None,
) -> executor_spec.ExecutorClassSpec:
  """Creates the executor spec instance for the func-generated component.

  Args:
    func: The function which executor spec instance will be generated based on.
    base_executor_class: base class of the generated executor class.
    executor_spec_class: class of the generated executor spec instance.
    arg_formats: An updated dict from arg name to its format. The format is the
      value of `ArgFormats` enum). Note that the parameter arg will be added to
      the dict if it's in the valid format and existing entries won't be
      changed.
    arg_defaults: A dict from func arg name to its default value.
    return_values_optionality: A dict from output names that are primitive type
      values returned from the user function to whether they are `Optional`.
    json_compatible_outputs: A dict from output names that have json compatible
      types to their typehints. Json compatibility is determined by
      `tfx.dsl.component.experimental.json_compat.is_json_compatible`.

  Returns:
    an instance of `executor_spec_class` whose executor_class is a subclass of
    `base_executor_class`.
  """
  assert_no_private_func_in_main(func)
  executor_class_name = f'{func.__name__}_Executor'
  executor_class = type(
      executor_class_name,
      (base_executor_class,),
      {
          '_ARG_FORMATS': arg_formats,
          '_ARG_DEFAULTS': arg_defaults,
          # The function needs to be marked with `staticmethod` so that later
          # references of `self._FUNCTION` do not result in a bound method (i.e.
          # one with `self` as its first parameter).
          '_FUNCTION': staticmethod(func),  # pytype: disable=not-callable
          '_RETURNED_VALUES': return_values_optionality,
          '_RETURN_JSON_COMPAT_TYPEHINT': json_compatible_outputs,
          '__module__': func.__module__,
      },
  )
  # Expose the generated executor class in the same module as the decorated
  # function. This is needed so that the executor class can be accessed at the
  # proper module path. One place this is needed is in the Dill pickler used by
  # Apache Beam serialization.
  module = sys.modules[func.__module__]
  setattr(module, executor_class_name, executor_class)

  executor_spec_instance = executor_spec_class(executor_class=executor_class)
  return executor_spec_instance


def create_component_class(
    func: types.FunctionType,
    arg_defaults: Dict[str, Any],
    arg_formats: Dict[str, ArgFormats],
    base_executor_class: Type[base_executor.BaseExecutor],
    executor_spec_class: Type[executor_spec.ExecutorClassSpec],
    base_component_class: Type[base_component.BaseComponent],
    inputs: Optional[Dict[str, Type[artifact.Artifact]]] = None,
    outputs: Optional[Dict[str, Type[artifact.Artifact]]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    type_annotation: Optional[Type[system_executions.SystemExecution]] = None,
    json_compatible_inputs: Optional[Dict[str, Any]] = None,
    json_compatible_outputs: Optional[Dict[str, Any]] = None,
    return_values_optionality: Optional[Dict[str, bool]] = None,
) -> Type[base_component.BaseComponent]:
  """Creates the component class for the func-generated component.

  Args:
    func: The function which the component class will be generated based on.
    arg_defaults: A dict from func arg name to its default value.
    arg_formats: An updated dict from arg name to its format. The format is the
      value of `ArgFormats` enum). Note that the parameter arg will be added to
      the dict if it's in the valid format and existing entries won't be
      changed.
    base_executor_class: base class of the generated executor class.
    executor_spec_class: class of the generated executor spec instance.
    base_component_class: The base class of the generated component class.
    inputs: A dict from input name to its Artifact type.
    outputs: A dict from output name to its Artifact type.
    parameters: A dict from parameter name to its primitive type.
    type_annotation: a subclass to SystemExecution used to annotate the
      component on ComponentSpec.
    json_compatible_inputs: A dict from input names that have json compatible
      types to their typehints. Json compatibility is determined by
      `tfx.dsl.component.experimental.json_compat.is_json_compatible`.
    json_compatible_outputs: A dict from output names that have json compatible
      types to their typehints. Json compatibility is determined by
      `tfx.dsl.component.experimental.json_compat.is_json_compatible`.
    return_values_optionality: A dict from output names that are primitive type
      values returned from the user function to whether they are `Optional`.

  Returns:
    a subclass of `base_component_class`.
  """

  component_spec_class = _create_component_spec_class(
      func,
      arg_defaults,
      inputs,
      outputs,
      parameters,
      type_annotation,
      json_compatible_inputs,
      json_compatible_outputs,
  )

  executor_spec_instance = _create_executor_spec_instance(
      func,
      base_executor_class,
      executor_spec_class,
      arg_formats,
      arg_defaults,
      return_values_optionality,
      json_compatible_outputs,
  )

  return type(
      func.__name__,
      (base_component_class,),
      {
          'SPEC_CLASS': component_spec_class,
          'EXECUTOR_SPEC': executor_spec_instance,
          '__module__': func.__module__,
          'test_call': staticmethod(func),  # pytype: disable=not-callable
      },
  )
