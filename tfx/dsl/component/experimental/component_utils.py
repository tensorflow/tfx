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
"""Utils for TFX component types. Intended for internal usage only."""

import inspect
from typing import Any, Callable, Mapping, Optional, Type

from tfx import types
from tfx.dsl.component.experimental import json_compat
from tfx.dsl.component.experimental import utils
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec as base_executor_spec
from tfx.orchestration.portable.execution import context
from tfx.proto.orchestration import executable_spec_pb2
from tfx.types import component_spec
from tfx.types import standard_artifacts
from tfx.types.system_executions import SystemExecution
from tfx.utils import name_utils
from tfx.utils import pure_typing_utils


_VALUE_ARTIFACT_TO_TYPE = {
    standard_artifacts.Integer: int,
    standard_artifacts.Float: float,
    standard_artifacts.String: str,
    standard_artifacts.Bytes: bytes,
    standard_artifacts.Boolean: bool,
}


def _convert_function_to_python_executable_spec(
    fn: Optional[Callable[..., Any]] = None
) -> Optional[executable_spec_pb2.PythonClassExecutableSpec]:
  """Validates and converts execution function to PythonClassExecutableSpec."""
  if fn is None:
    return None
  utils.assert_is_top_level_func(fn)
  function_path = name_utils.get_full_name(
      fn, strict_check=True
  )
  return executable_spec_pb2.PythonClassExecutableSpec(
      class_path=function_path
  )


def _type_check_execution_function_params(
    spec: type[component_spec.ComponentSpec],
    fn: Optional[Callable[..., Any]] = None,
) -> None:
  """Validates if the function parameter is aligned with the component spec.

  The function can have the flattened input / output / exec_properties as
  parameter whose name matches the dict key and annotated type matches
  the Artifact type or data type.

  Args:
    spec: A component spec.
    fn: A function that is supposed to be aligned with the given component spec.
  """
  if fn is None:
    return

  channel_parameters = {}
  for parameters in (spec.INPUTS, spec.OUTPUTS):
    channel_parameters.update(parameters)
  signature = inspect.signature(fn)

  # Execution function type check.
  for param_name in signature.parameters:
    param_type = signature.parameters[param_name].annotation
    if param_type is inspect.Signature.empty:
      raise TypeError(
          f'Execution hook function parameter "{param_name}" should be'
          ' annotated.'
      )

    if param_name in channel_parameters:
      channel = channel_parameters[param_name]
      allowed_param_types = []

      if not issubclass(channel.type, standard_artifacts.ValueArtifact):
        allowed_param_types = [
            list[channel.type],
            Optional[channel.type] if channel.optional else channel.type,
        ]
      elif param_name in spec.INPUTS:
        if channel.type in _VALUE_ARTIFACT_TO_TYPE:
          # Primitvie ValueArtifact input can be annotated as a primitive type.
          primitive_type = _VALUE_ARTIFACT_TO_TYPE[channel.type]
          allowed_param_types.append(
              Optional[primitive_type] if channel.optional else primitive_type
          )
        else:
          # JsonValue artifact type.
          is_param_optional, inner_param_type = (
              pure_typing_utils.maybe_unwrap_optional(param_type)
          )
          if (
              json_compat.is_json_compatible(inner_param_type)
              and is_param_optional == channel.optional
          ):
            continue  # Type check okay
          else:
            allowed_param_types.append(
                'Optional[JsonableType]' if channel.optional else 'JsonableType'
            )
            raise TypeError(
                f'Parameter type mismatched {param_name}: {param_type} from the'
                f' executable function {fn.__name__}. The allowed'
                f' types are {allowed_param_types}.'
            )

      # TODO(wssong): We should care for AsyncOutputArtifact type annotation for
      # channels with is_async=True (go/tflex-list-output).

      if param_type not in allowed_param_types:
        raise TypeError(
            f'Parameter type mismatched {param_name}: {param_type} from the'
            f' executable function {fn.__name__}. The allowed'
            f' types are {allowed_param_types}.'
        )
    elif param_name in spec.PARAMETERS:
      exec_prop = spec.PARAMETERS[param_name]

      allowed_param_types = [
          Optional[exec_prop.type] if exec_prop.optional else exec_prop.type
      ]

      if param_type not in allowed_param_types:
        raise TypeError(
            f'Parameter type mismatched {param_name}: {param_type} from the'
            f' executable function {fn.__name__}. The allowed'
            f' types are {allowed_param_types}.'
        )
    elif param_type is context.ExecutionContext:
      # Does not validate name to be matched for the ExecutionContext type_hint.
      pass
    else:
      raise AttributeError(
          f'Unsupported parameter {param_name}: {param_type} from the'
          f' executable function {fn.__name__}. Parameter should'
          ' be inputs or outputs of the component, or the ExecutionContext'
          ' variable.'
      )


def create_tfx_component_class(
    name: str,
    tfx_executor_spec: base_executor_spec.ExecutorSpec,
    input_channel_parameters: Optional[
        Mapping[str, component_spec.ChannelParameter]
    ] = None,
    output_channel_parameters: Optional[
        Mapping[str, component_spec.ChannelParameter]
    ] = None,
    execution_parameters: Optional[
        Mapping[str, component_spec.ExecutionParameter]
    ] = None,
    type_annotation: Optional[Type[SystemExecution]] = None,
    default_init_args: Optional[Mapping[str, Any]] = None,
    # pre_execution and post_execution are not supported in OSS.
    pre_execution: Optional[Callable[..., Any]] = None,
    post_execution: Optional[Callable[..., Any]] = None,
    base_class: Type[
        base_component.BaseComponent
    ] = base_component.BaseComponent,
) -> Callable[..., base_component.BaseComponent]:
  """Creates a TFX component class dynamically."""
  tfx_component_spec_class = type(
      str(name) + 'Spec',
      (component_spec.ComponentSpec,),
      dict(
          PARAMETERS=execution_parameters or {},
          INPUTS=input_channel_parameters or {},
          OUTPUTS=output_channel_parameters or {},
          TYPE_ANNOTATION=type_annotation,
      ),
  )

  for fn in (pre_execution, post_execution):
    if fn is not None:
      _type_check_execution_function_params(tfx_component_spec_class, fn)
      utils.assert_no_private_func_in_main(fn)
  try:
    pre_execution_spec, post_execution_spec = [
        _convert_function_to_python_executable_spec(fn)
        for fn in (pre_execution, post_execution)
    ]
  except ValueError as e:
    raise ValueError(f'Invalid execution hook function of {name}') from e

  def tfx_component_class_init(self, **kwargs):
    arguments = {}
    arguments.update(kwargs)
    arguments.update(default_init_args or {})

    # Provide default values for output channels.
    output_channel_params = output_channel_parameters or {}
    for output_key, output_channel_param in output_channel_params.items():
      if output_key not in arguments:
        output_channel = types.OutputChannel(
            artifact_type=output_channel_param.type,
            producer_component=self,
            output_key=output_key,
            is_async=output_channel_param.is_async,
        )
        arguments[output_key] = output_channel

    base_class.__init__(
        self,
        # Generate spec by wiring up the input/output channel.
        spec=self.__class__.SPEC_CLASS(**arguments))
    # Set class name as the default id. It can be overwritten by the user.
    if not self.id:
      base_class.with_id(self, self.__class__.__name__)

  tfx_component_class = type(
      str(name),
      (base_class,),
      dict(
          SPEC_CLASS=tfx_component_spec_class,
          EXECUTOR_SPEC=tfx_executor_spec,
          PRE_EXECUTABLE_SPEC=pre_execution_spec,
          POST_EXECUTABLE_SPEC=post_execution_spec,
          __init__=tfx_component_class_init,
      ),
  )
  return tfx_component_class
