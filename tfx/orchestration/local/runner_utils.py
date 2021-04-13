"""Utils for pipeline runners for TFleX pipelines."""
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
from typing import Optional

from tfx.proto.orchestration import local_deployment_config_pb2
from tfx.proto.orchestration import pipeline_pb2

from google.protobuf import any_pb2
from google.protobuf import message


def extract_local_deployment_config(
    pipeline: pipeline_pb2.Pipeline
) -> local_deployment_config_pb2.LocalDeploymentConfig:
  """Extracts the proto.Any pipeline.deployment_config to LocalDeploymentConfig."""

  if not pipeline.deployment_config:
    raise ValueError('deployment_config is not available in the pipeline.')

  result = local_deployment_config_pb2.LocalDeploymentConfig()
  if pipeline.deployment_config.Unpack(result):
    return result

  result = pipeline_pb2.IntermediateDeploymentConfig()
  if pipeline.deployment_config.Unpack(result):
    return _to_local_deployment(result)

  raise ValueError('deployment_config {} of type {} is not supported'.format(
      pipeline.deployment_config, type(pipeline.deployment_config)))


def _build_executable_spec(
    node_id: str,
    spec: any_pb2.Any) -> local_deployment_config_pb2.ExecutableSpec:
  """Builds ExecutableSpec given the any proto from IntermediateDeploymentConfig."""
  result = local_deployment_config_pb2.ExecutableSpec()
  if spec.Is(result.python_class_executable_spec.DESCRIPTOR):
    spec.Unpack(result.python_class_executable_spec)
  elif spec.Is(result.container_executable_spec.DESCRIPTOR):
    spec.Unpack(result.container_executable_spec)
  elif spec.Is(result.beam_executable_spec.DESCRIPTOR):
    spec.Unpack(result.beam_executable_spec)
  else:
    raise ValueError(
        'Executor spec of {} is expected to be of one of the '
        'types of tfx.orchestration.deployment_config.ExecutableSpec.spec '
        'but got type {}'.format(node_id, spec.type_url))
  return result


def _build_local_platform_config(
    node_id: str,
    spec: any_pb2.Any) -> local_deployment_config_pb2.LocalPlatformConfig:
  """Builds LocalPlatformConfig given the any proto from IntermediateDeploymentConfig."""
  result = local_deployment_config_pb2.LocalPlatformConfig()
  if spec.Is(result.docker_platform_config.DESCRIPTOR):
    spec.Unpack(result.docker_platform_config)
  else:
    raise ValueError(
        'Platform config of {} is expected to be of one of the types of '
        'tfx.orchestration.deployment_config.LocalPlatformConfig.config '
        'but got type {}'.format(node_id, spec.type_url))
  return result


def _to_local_deployment(
    input_config: pipeline_pb2.IntermediateDeploymentConfig
) -> local_deployment_config_pb2.LocalDeploymentConfig:
  """Turns IntermediateDeploymentConfig to LocalDeploymentConfig."""
  result = local_deployment_config_pb2.LocalDeploymentConfig()
  for k, v in input_config.executor_specs.items():
    result.executor_specs[k].CopyFrom(_build_executable_spec(k, v))

  for k, v in input_config.custom_driver_specs.items():
    result.custom_driver_specs[k].CopyFrom(_build_executable_spec(k, v))

  for k, v in input_config.node_level_platform_configs.items():
    result.node_level_platform_configs[k].CopyFrom(
        _build_local_platform_config(k, v))

  if not input_config.metadata_connection_config.Unpack(
      result.metadata_connection_config):
    raise ValueError('metadata_connection_config is expected to be in type '
                     'ml_metadata.ConnectionConfig, but got type {}'.format(
                         input_config.metadata_connection_config.type_url))
  return result


def extract_executor_spec(
    deployment_config: local_deployment_config_pb2.LocalDeploymentConfig,
    node_id: str
) -> Optional[message.Message]:
  return _unwrap_executable_spec(
      deployment_config.executor_specs.get(node_id))


def extract_custom_driver_spec(
    deployment_config: local_deployment_config_pb2.LocalDeploymentConfig,
    node_id: str
) -> Optional[message.Message]:
  return _unwrap_executable_spec(
      deployment_config.custom_driver_specs.get(node_id))


def _unwrap_executable_spec(
    executable_spec: Optional[local_deployment_config_pb2.ExecutableSpec]
) -> Optional[message.Message]:
  """Unwraps the one of spec from ExecutableSpec."""
  return (getattr(executable_spec, executable_spec.WhichOneof('spec'))
          if executable_spec else None)
