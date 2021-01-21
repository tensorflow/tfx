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
"""Service job runner interface and registry."""

import abc
from typing import Text, Type, TypeVar

from tfx.orchestration import metadata
from tfx.proto.orchestration import pipeline_pb2

from ml_metadata.proto import metadata_store_pb2


class ServiceJobRunner(abc.ABC):
  """Interface for service job runners."""

  def __init__(self, mlmd_handle: metadata.Metadata,
               pipeline: pipeline_pb2.Pipeline, node_id: Text,
               execution: metadata_store_pb2.Execution):
    """Constructor.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline: The pipeline IR proto.
      node_id: The node that contains the servie job to run.
      execution: The pipeline execution in MLMD.
    """
    self.mlmd_handle = mlmd_handle
    self.pipeline = pipeline
    self.node_id = node_id
    self.execution = execution

  @abc.abstractmethod
  def run(self) -> None:
    """Runs the given service jobs.

    This method blocks until the pipeline is stopped or until explicitly
    cancelled by a call to `cancel`. When cancelled, `run` is expected to stop
    any ongoing work, clean up and return as soon as possible. Note that
    `cancel` will be invoked from a different thread than `run` and hence the
    concrete implementations must be thread safe. It's technically possible for
    `cancel` to be invoked before `run`; job runner implementations should
    handle this case by returning from `run` immediately.
    """

  @abc.abstractmethod
  def cancel(self) -> None:
    """Cancels service job runner.

    This method will be invoked from a different thread than the thread that's
    blocked on call to `run`. `cancel` must return immediately when called.
    Upon cancellation, `run` method is expected to stop any ongoing work,
    clean up and return as soon as possible. It's technically possible for
    `cancel` to be invoked before `run`; job runner implementations should
    handle this case by returning from `run` immediately.
    """


T = TypeVar('T', bound='ServiceJobRunnerRegistry')


class ServiceJobRunnerRegistry:
  """A registry for service job runner."""

  _service_job_runner_registry = {}

  @classmethod
  def register(cls: Type[T], executor_spec_type_url: Text,
               runner_class: Type[ServiceJobRunner]) -> None:
    """Registers a new service job runner for the given executor spec type url.

    Args:
      executor_spec_type_url: The URL of the executor spec type.
      runner_class: The class that will be instantiated for a matching job.

    Raises:
      ValueError: If `executor_spec_type_url` is already in the registry.
    """
    if executor_spec_type_url in cls._service_job_runner_registry:
      raise ValueError(
          'A service job runner already exists for the executor spec type url: '
          '{}'.format(executor_spec_type_url))
    cls._service_job_runner_registry[executor_spec_type_url] = runner_class

  @classmethod
  def clear(cls: Type[T]) -> None:
    cls._service_job_runner_registry.clear()

  @classmethod
  def create_service_job_runner(
      cls: Type[T], mlmd_handle: metadata.Metadata,
      pipeline: pipeline_pb2.Pipeline, node_id: Text,
      execution: metadata_store_pb2.Execution) -> ServiceJobRunner:
    """Creates a service job runner for the given job.

    Note that this assumes deployment_config packed in the pipeline IR is of
    type `IntermediateDeploymentConfig`. This detail may change in the future.

    Args:
      mlmd_handle: A handle to the MLMD db.
      pipeline: The pipeline IR.
      node_id: The pipeline node that contains the service job to run.
      execution: The pipeline execution in MLMD.

    Returns:
      An instance of `ServiceJobRunner` for the given node.

    Raises:
      ValueError: Deployment config not present in the IR proto or if executor
        spec for the node not configured in the IR.
    """
    if not pipeline.deployment_config.Is(
        pipeline_pb2.IntermediateDeploymentConfig.DESCRIPTOR):
      raise ValueError('No deployment config found in pipeline IR.')
    depl_config = pipeline_pb2.IntermediateDeploymentConfig()
    pipeline.deployment_config.Unpack(depl_config)
    if node_id not in depl_config.executor_specs:
      raise ValueError(
          'Executor spec for node id `{}` not found in pipeline IR.'.format(
              node_id))
    executor_spec_type_url = depl_config.executor_specs[node_id].type_url
    return cls._service_job_runner_registry[executor_spec_type_url](
        mlmd_handle=mlmd_handle,
        pipeline=pipeline,
        node_id=node_id,
        execution=execution)
