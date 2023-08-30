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


from tfx.dsl.components.base import base_node
from tfx.orchestration import node_proto_view
from tfx.proto.orchestration import pipeline_pb2


MANAGER_ARTIFACT_OUTPUT_CHANNEL = "manager"
MANAGER_ARTIFACT_READY_KEY = "manager_artifact_ready"


def is_manager_node(
    node: pipeline_pb2.PipelineNode
    | node_proto_view.NodeProtoView
    | base_node.BaseNode,
) -> bool:
  """Determines if a node is a ManagerNode."""
  if isinstance(
      node, (pipeline_pb2.PipelineNode, node_proto_view.NodeProtoView)
  ):
    return node.execution_options and node.execution_options.HasField(
        "manager_node_execution_options"
    )
  elif isinstance(node, base_node.BaseNode):
    return (
        node.node_execution_options is not None
        and getattr(
            node.node_execution_options,
            "manager_node_execution_options",
            None,
        )
        is not None
    )
