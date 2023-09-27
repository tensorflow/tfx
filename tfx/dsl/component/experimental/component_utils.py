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

from typing import Any, Callable, Dict, Optional, Type

from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec as base_executor_spec
from tfx.types import component_spec
from tfx.types.system_executions import SystemExecution


def create_tfx_component_class(
    name: str,
    tfx_executor_spec: base_executor_spec.ExecutorSpec,
    input_channel_parameters: Optional[Dict[
        str, component_spec.ChannelParameter]] = None,
    output_channel_parameters: Optional[Dict[
        str, component_spec.ChannelParameter]] = None,
    execution_parameters: Optional[Dict[
        str, component_spec.ExecutionParameter]] = None,
    type_annotation: Optional[Type[SystemExecution]] = None,
    default_init_args: Optional[Dict[str, Any]] = None,
    base_class: Type[
        base_component.BaseComponent] = base_component.BaseComponent,
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

  def tfx_component_class_init(self, **kwargs):
    arguments = {}
    arguments.update(kwargs)
    arguments.update(default_init_args or {})

    # Provide default values for output channels.
    output_channel_params = output_channel_parameters or {}
    for output_key, output_channel_param in output_channel_params.items():
      if output_key not in arguments:
        arguments[output_key] = types.Channel(type=output_channel_param.type)

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
          __init__=tfx_component_class_init,
      ),
  )
  return tfx_component_class
