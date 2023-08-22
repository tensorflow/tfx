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
"""Functions to unpack IntermediateDeploymentConfig and its children."""

from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import message


def get_platform_config(
    deployment_config: pipeline_pb2.IntermediateDeploymentConfig,
    node_id: str,
) -> message.Message | None:
  """Returns the platform config for the given node if it exists."""
  return deployment_config.node_level_platform_configs.get(node_id)


def get_executor_spec(
    deployment_config: pipeline_pb2.IntermediateDeploymentConfig,
    node_id: str,
) -> message.Message | None:
  """Returns the executor spec for the given node if it exists."""
  return deployment_config.executor_specs.get(node_id)
