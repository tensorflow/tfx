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

from typing import Any, Callable, Mapping, Optional, Type

from tfx import types
from tfx.dsl.component.experimental import utils
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec as base_executor_spec
from tfx.proto.orchestration import executable_spec_pb2
from tfx.types import component_spec
from tfx.types.system_executions import SystemExecution
from tfx.utils import name_utils


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
