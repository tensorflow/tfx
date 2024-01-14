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
"""DependencyModule for the node execution environment."""


from tfx.orchestration.portable import data_types
from tfx.orchestration.portable.execution import context
from tfx.orchestration.portable.execution import di_providers
from tfx.utils.di import module


def create_module_from_static_environ(
    exec_info: data_types.ExecutionInfo,
) -> module.DependencyModule:
  """Create DependencyModule for node execution."""
  # Order of the provider (precedence) matters.
  result = module.DependencyModule()
  result.provide_named_class('context', context.ExecutionContext)
  result.add_provider(
      di_providers.FlatExecutionInfoProvider(
          {
              *exec_info.input_dict.keys(),
              *exec_info.output_dict.keys(),
              *exec_info.exec_properties.keys(),
          },
          strict=True,
      )
  )
  result.provide_named_value('exec_info', exec_info)
  return result
