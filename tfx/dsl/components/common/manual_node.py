# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Manual Node definition. Experimental."""

from typing import Any, Dict

from tfx.dsl.components.base import base_node

# Constant to access a manual node's description property.
DESCRIPTION_KEY = 'description'


class ManualNode(base_node.BaseNode):
  """Definition for Manual Node.

  Important: This feature is currently experimental and only available in the
  TFX experimental orchestrator.

  Manual Node is a special TFX node which requires manual operation to complete.
  Example usage:

  ```
  trainer = Trainer(...)
  manual_node = ManualNode(
      description='Please manually inspect the generated model before pushing.'
  )
  pusher = Pusher(...)

  manual_node.add_upstream_node(trainer)
  manual_node.add_downstream_node(pusher)
  ```
  """

  def __init__(self, description: str):
    self._description = description
    super().__init__()

  @property
  def inputs(self) -> Dict[str, Any]:
    return {}

  @property
  def outputs(self) -> Dict[str, Any]:
    return {}

  @property
  def exec_properties(self) -> Dict[str, Any]:
    return {
        DESCRIPTION_KEY: self._description
    }
