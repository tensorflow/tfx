"""Creates a manager node from an existing component."""
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

import copy

from tfx.dsl.components.base import base_component
from tfx.dsl.experimental.node_execution_options import utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import channel
from tfx.types import component_spec
from tfx.types import standard_artifacts

from google.protobuf import message


MANAGER_ARTIFACT_OUTPUT_CHANNEL = "manager"
MANAGER_ARTIFACT_READY_KEY = "manager_artifact_ready"


class _BaseManagerNode(base_component.BaseComponent):
  """Base class for manager nodes. Should never be directly instantiated."""

  def __init__(
      self,
      *args,
      node_execution_options: utils.NodeExecutionOptions,
      platform_config: message.Message,
      **kwargs,
  ):
    super().__init__(*args, **kwargs)
    self.node_execution_options = node_execution_options
    self.platform_config = platform_config

  @property
  def is_manager_node(self) -> bool:
    return True


def create_manager_node_from(
    component: base_component.BaseComponent,
) -> _BaseManagerNode:
  """Converts a component into a ManagerNode."""
  spec_class = copy.deepcopy(component.SPEC_CLASS)
  spec_kwargs = component.spec.to_json_dict()
  if spec_class.OUTPUTS is not None:
    spec_class.OUTPUTS[MANAGER_ARTIFACT_OUTPUT_CHANNEL] = (
        component_spec.ChannelParameter(type=standard_artifacts.ManagerArtifact)
    )
    manager_channel = channel.OutputChannel(
        artifact_type=standard_artifacts.ManagerArtifact,
        producer_component=component.id,
        output_key=MANAGER_ARTIFACT_OUTPUT_CHANNEL,
    )

    spec_kwargs = component.spec.to_json_dict()
    spec_kwargs.update({MANAGER_ARTIFACT_OUTPUT_CHANNEL: manager_channel})
  spec = spec_class(**spec_kwargs)
  node_execution_options = (
      utils.NodeExecutionOptions()
      if component.node_execution_options is None
      else copy.deepcopy(component.node_execution_options)
  )
  node_execution_options.lazily_execute = True

  manager_node_class = type(
      component.id,
      (_BaseManagerNode,),
      {
          "SPEC_CLASS": spec_class,
          "EXECUTOR_SPEC": copy.deepcopy(component.EXECUTOR_SPEC),
      },
  )
  return manager_node_class(
      spec=spec,
      node_execution_options=node_execution_options,
      platform_config=copy.deepcopy(component.platform_config),
  )


def is_manager_node(node: pipeline_pb2.PipelineNode) -> bool:
  """Determines if a node proto is a ManagerNode."""
  return node.execution_options.HasField("manager_node_execution_options")
