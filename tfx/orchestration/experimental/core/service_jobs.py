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
"""Interfaces and functionality for dealing with service jobs."""

import abc
import enum

from tfx.orchestration.experimental.core import pipeline_state as pstate


@enum.unique
class ServiceStatus(enum.Enum):
  UNKNOWN = 0
  RUNNING = 1
  SUCCESS = 2
  FAILED = 3


class ServiceJobManager(abc.ABC):
  """Interface for service job manager.

  Service jobs are long-running jobs associated with a node or a pipeline that
  persist across executions (eg: worker pools, Tensorboard, etc). Service jobs
  should be started before the nodes that depend on them can be run.
  """

  @abc.abstractmethod
  def ensure_node_services(self, pipeline_state: pstate.PipelineState,
                           node_id: str) -> ServiceStatus:
    """Ensures necessary service jobs are started and healthy for the node.

    `ensure_node_services` will be called in the orchestration loop periodically
    and is expected to:

    * Start any service jobs required by the pipeline node.
    * Probe job health, handle failure and return appropriate status.

    Note that this method will only be called if either `is_pure_service_node`
    or `is_mixed_service_node` return `True` for the node.

    Args:
      pipeline_state: A `PipelineState` object for an active pipeline.
      node_id: Id of the node to ensure services.

    Returns:
      Status of the service job(s) for the node.
    """

  @abc.abstractmethod
  def stop_node_services(self, pipeline_state: pstate.PipelineState,
                         node_id: str) -> None:
    """Stops service jobs (if any) associated with the node.

    Note that this method will only be called if either `is_pure_service_node`
    or `is_mixed_service_node` return `True` for the node.

    Args:
      pipeline_state: A `PipelineState` object for an active pipeline.
      node_id: Id of the node to stop services.
    """

  @abc.abstractmethod
  def is_pure_service_node(self, pipeline_state: pstate.PipelineState,
                           node_id: str) -> bool:
    """Returns `True` if the given node only has service job(s).

    Args:
      pipeline_state: A `PipelineState` object for an active pipeline.
      node_id: Id of the node in the pipeline to be checked.

    Returns:
      `True` if the node only has service job(s).
    """

  @abc.abstractmethod
  def is_mixed_service_node(self, pipeline_state: pstate.PipelineState,
                            node_id: str) -> bool:
    """Returns `True` if the given node has a mix of executor and service jobs.

    Args:
      pipeline_state: A `PipelineState` object for an active pipeline.
      node_id: Id of the node in the pipeline to be checked.

    Returns:
      `True` if the node has a mix of executor and service jobs.
    """


class DummyServiceJobManager(ServiceJobManager):
  """A service job manager for environments without service jobs support."""

  def ensure_node_services(self, pipeline_state: pstate.PipelineState,
                           node_id: str) -> ServiceStatus:
    del pipeline_state, node_id
    raise NotImplementedError('Service jobs not supported.')

  def stop_node_services(self, pipeline_state: pstate.PipelineState,
                         node_id: str) -> None:
    del pipeline_state, node_id
    raise NotImplementedError('Service jobs not supported.')

  def is_pure_service_node(self, pipeline_state: pstate.PipelineState,
                           node_id: str) -> bool:
    del pipeline_state, node_id
    return False

  def is_mixed_service_node(self, pipeline_state: pstate.PipelineState,
                            node_id: str) -> bool:
    del pipeline_state, node_id
    return False
