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
"""For environment specific extensions."""

import abc
from typing import Optional, Sequence

from tfx.orchestration.experimental.core import orchestration_options
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib

from ml_metadata.proto import metadata_store_pb2

_ENV = None


class Env(abc.ABC):
  """Base class for environment specific extensions."""

  def __enter__(self) -> None:
    global _ENV
    self._old_env = _ENV
    _ENV = self

  def __exit__(self, exc_type, exc_val, exc_tb):
    global _ENV
    _ENV = self._old_env

  @abc.abstractmethod
  def get_orchestration_options(
      self, pipeline: pipeline_pb2.Pipeline
  ) -> orchestration_options.OrchestrationOptions:
    """Gets orchestration options for the pipeline."""

  @abc.abstractmethod
  def get_base_dir(self) -> Optional[str]:
    """Returns the base directory for the pipeline."""

  @abc.abstractmethod
  def label_and_tag_pipeline_run(
      self, mlmd_handle, pipeline_id, pipeline_run_id, labels, tags
  ) -> None:
    """Labels and tags the pipeline run after it starts."""

  @abc.abstractmethod
  def max_mlmd_str_value_length(self) -> Optional[int]:
    """Returns max size of a string value in MLMD db, `None` if unlimited."""

  @abc.abstractmethod
  def concurrent_pipeline_runs_enabled(
      self, pipeline: pipeline_pb2.Pipeline
  ) -> bool:
    """Returns whether concurrent pipeline runs are enabled."""

  @abc.abstractmethod
  def is_pure_service_node(
      self, pipeline: pipeline_pb2.Pipeline, node_id: str
  ) -> bool:
    """Returns whether the given node is a pure service node."""

  @abc.abstractmethod
  def health_status(self) -> status_lib.Status:
    """Returns the orchestrator's overall health status."""

  @abc.abstractmethod
  def set_health_status(self, status: status_lib.Status) -> None:
    """Sets orchestrator's overall health status."""

  @abc.abstractmethod
  def check_if_can_orchestrate(self, pipeline: pipeline_pb2.Pipeline) -> None:
    """Check if this orchestrator is capable of orchestrating the pipeline."""

  @abc.abstractmethod
  def prepare_orchestrator_for_pipeline_run(
      self, pipeline: pipeline_pb2.Pipeline
  ):
    """Prepares the orchestrator to execute the provided pipeline.

    This *can* mutate the provided IR in-place.

    Args:
      pipeline: The pipeline IR to prepare for.
    """

  @abc.abstractmethod
  def create_sync_or_upsert_async_pipeline_run(
      self,
      owner: str,
      pipeline_name: str,
      execution: metadata_store_pb2.Execution,
      pipeline: pipeline_pb2.Pipeline,
      pipeline_run_metadata: Optional[str] = None,
      base_pipeline_run_id: Optional[str] = None,
  ) -> None:
    """Creates or updates a (sub-)pipeline run in the storage backend."""

  @abc.abstractmethod
  def update_pipeline_run_status(
      self,
      owner: str,
      pipeline_name: str,
      pipeline: pipeline_pb2.Pipeline,
      original_execution: metadata_store_pb2.Execution,
      modified_execution: metadata_store_pb2.Execution,
      sub_pipeline_ids: Optional[Sequence[str]] = None,
  ) -> None:
    """Updates orchestrator storage backends with pipeline run status."""

  @abc.abstractmethod
  def should_orchestrate(self, pipeline: pipeline_pb2.Pipeline) -> bool:
    """Environment specific definition of orchestratable pipeline.

    `pipeline_state.PipelineState.load_all_active` will only load the
    orchestratable pipeline states according to this definition. For example,
    sharded orchestrator will only filter the pipeline_run_id that belongs to
    its own shard index.

    Args:
      pipeline: The Pipeline IR.

    Returns:
      Whether the env should orchestrate the pipeline.
    """


class _DefaultEnv(Env):
  """Default environment."""

  def get_orchestration_options(
      self, pipeline: pipeline_pb2.Pipeline
  ) -> orchestration_options.OrchestrationOptions:
    del pipeline
    return orchestration_options.OrchestrationOptions()

  def get_base_dir(self) -> Optional[str]:
    return None

  def label_and_tag_pipeline_run(
      self, mlmd_handle, pipeline_id, pipeline_run_id, labels, tags
  ) -> None:
    return None

  def max_mlmd_str_value_length(self) -> Optional[int]:
    return None

  def concurrent_pipeline_runs_enabled(
      self, pipeline: pipeline_pb2.Pipeline
  ) -> bool:
    return False

  def is_pure_service_node(
      self, pipeline: pipeline_pb2.Pipeline, node_id: str
  ) -> bool:
    return False

  def health_status(self) -> status_lib.Status:
    return status_lib.Status(code=status_lib.Code.OK)

  def set_health_status(self, status: status_lib.Status) -> None:
    pass

  def check_if_can_orchestrate(self, pipeline: pipeline_pb2.Pipeline) -> None:
    pass

  def prepare_orchestrator_for_pipeline_run(
      self, pipeline: pipeline_pb2.Pipeline
  ):
    pass

  def create_sync_or_upsert_async_pipeline_run(
      self,
      owner: str,
      pipeline_name: str,
      execution: metadata_store_pb2.Execution,
      pipeline: pipeline_pb2.Pipeline,
      pipeline_run_metadata: Optional[str] = None,
      base_pipeline_run_id: Optional[str] = None,
  ) -> None:
    pass

  def update_pipeline_run_status(
      self,
      owner: str,
      pipeline_name: str,
      pipeline: pipeline_pb2.Pipeline,
      original_execution: metadata_store_pb2.Execution,
      modified_execution: metadata_store_pb2.Execution,
      sub_pipeline_ids: Optional[Sequence[str]] = None,
  ) -> None:
    pass

  def should_orchestrate(self, pipeline: pipeline_pb2.Pipeline) -> bool:
    # By default, all pipeline runs should be orchestrated.
    return True


_ENV = _DefaultEnv()


def get_env() -> Env:
  return _ENV
