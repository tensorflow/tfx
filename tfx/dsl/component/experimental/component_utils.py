# Lint as: python3
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
from typing import Any, Callable, Dict, Optional, Text

from tfx.components.base import base_component
from tfx.components.base import executor_spec as base_executor_spec
from tfx.types import component_spec


def create_tfx_component_class(
    name: Text,
    tfx_executor_spec: base_executor_spec.ExecutorSpec,
    input_channel_parameters: Dict[Text,
                                   component_spec.ChannelParameter] = None,
    output_channel_parameters: Dict[Text,
                                    component_spec.ChannelParameter] = None,
    execution_parameters: Dict[Text, component_spec.ExecutionParameter] = None,
    default_init_args: Optional[Dict[Text, Any]] = None
) -> Callable[..., base_component.BaseComponent]:
  """Creates a TFX component class dynamically."""
  tfx_component_spec_class = type(
      str(name) + 'Spec',
      (component_spec.ComponentSpec,),
      dict(
          PARAMETERS=execution_parameters,
          INPUTS=input_channel_parameters,
          OUTPUTS=output_channel_parameters,
      ),
  )

  def tfx_component_class_init(self, **kwargs):
    instance_name = kwargs.pop('instance_name', None)
    arguments = {}
    arguments.update(kwargs)
    arguments.update(default_init_args)

    base_component.BaseComponent.__init__(
        self,
        # Generate spec by wiring up the input/output channel.
        spec=self.__class__.SPEC_CLASS(**arguments),
        instance_name=instance_name,
    )

  tfx_component_class = type(
      str(name),
      (base_component.BaseComponent,),
      dict(
          SPEC_CLASS=tfx_component_spec_class,
          EXECUTOR_SPEC=tfx_executor_spec,
          __init__=tfx_component_class_init,
      ),
  )
  return tfx_component_class
