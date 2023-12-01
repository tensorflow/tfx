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
"""Node for cleaning up the job."""
from tfx.dsl.compiler import constants as compiler_constants
from tfx.dsl.components.base import base_node


class JobCleanup(base_node.BaseNode):
  """Node for bring down jobs. Meant for use with LaunchOnly components."""

  def __init__(self, launch_only_component: base_node.BaseNode):
    # TODO: b/311216280 - Verify artifact type and that upstream is launch only.
    if len(launch_only_component.outputs) != 1:
      raise ValueError(
          'Launch only component must have exactly one output, found'
          f' {launch_only_component.outputs}'
      )
    super().__init__()

    launch_only_channel = launch_only_component.outputs[
        compiler_constants.LAUNCH_ONLY_CHANNEL_NAME
    ]
    self._inputs = {
        compiler_constants.LAUNCH_ONLY_CHANNEL_NAME: launch_only_channel
    }
    self._launcher_component = launch_only_component

  @property
  def inputs(self):
    return self._inputs

  # TODO: b/314154775 - Setup user defined outputs.
  @property
  def outputs(self):
    return {}

  @property
  def exec_properties(self):
    return {'launcher_component': self._launcher_component.id}
