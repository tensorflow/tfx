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
from typing import Set

from tfx.orchestration.experimental.core import pipeline_state as pstate
from tfx.orchestration.experimental.core import task as task_lib


class ServiceJobManager(abc.ABC):
  """Interface for service job manager."""

  @abc.abstractmethod
  def ensure_services(
      self, pipeline_state: pstate.PipelineState) -> Set[task_lib.NodeUid]:
    """Ensures necessary service jobs are started and healthy for the pipeline.

    Service jobs are long-running jobs associated with a node or the pipeline
    that persist across executions (eg: worker pools, Tensorboard, etc). Service
    jobs are started before the nodes that depend on them are started.

    `ensure_services` will be called in the orchestration loop periodically and
    is expected to:

    1. Start any service jobs required by the pipeline nodes.
    2. Probe job health and handle failures. If a service job fails, the
       corresponding node uids should be returned.
    3. Optionally stop service jobs that are no longer needed. Whether or not a
       service job is needed is context dependent, for eg: in a typical sync
       pipeline, one may want Tensorboard job to continue running even after the
       corresponding trainer has completed but others like worker pool services
       may be shutdown.

    Args:
      pipeline_state: A `PipelineState` object for an active pipeline.

    Returns:
      List of NodeUids of nodes whose service jobs are in a state of permanent
      failure.
    """

  @abc.abstractmethod
  def stop_services(self, pipeline_state: pstate.PipelineState) -> None:
    """Stops all service jobs associated with the pipeline.

    Args:
      pipeline_state: A `PipelineState` object for an active pipeline.
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
