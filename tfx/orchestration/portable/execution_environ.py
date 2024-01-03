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
"""Environment for component execution."""

import contextlib
from typing import Optional, Type, TypeVar

from tfx.orchestration.portable import data_types
from tfx.orchestration.portable.execution import di_providers
from tfx.orchestration.portable.execution import context
from tfx.utils.di import module

from google.protobuf import message


_TAny = TypeVar('_TAny')


class Environ(contextlib.ExitStack):
  """Tflex component execution environment."""

  def __init__(
      self,
      *,
      execution_info: data_types.ExecutionInfo,
      executor_spec: Optional[message.Message] = None,
      platform_config: Optional[message.Message] = None,
      pipeline_platform_config: Optional[message.Message] = None,
  ):
    super().__init__()

    self._module = module.DependencyModule()

    self._module.provide_value(value=execution_info)
    names = {
        *execution_info.input_dict,
        *execution_info.output_dict,
        *execution_info.exec_properties,
    }
    self._module.add_provider(di_providers.FlatExecutionInfoProvider(names))

    # TODO(wssong): Change this to provide_class(context.ExecutionContext)
    # after wiring executor_spec, platform_config, and pipeline_platform_config
    # with concrete types (not message.Message) to be used for the
    # module.match() function.
    execution_context = context.ExecutionContext(
        exec_info=execution_info,
        executor_spec=executor_spec,
        platform_config=platform_config,
        pipeline_platform_config=pipeline_platform_config,
    )
    self._module.provide_value(execution_context)

  def strict_get(self, name: str, type_hint: Type[_TAny]) -> _TAny:
    """Get environment value with name and type hint."""
    return self._module.get(name, type_hint)
