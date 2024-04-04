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
from typing import Optional

from tfx.orchestration.experimental.core import orchestration_options
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import status as status_lib

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
  def max_mlmd_str_value_length(self) -> Optional[int]:
    """Returns max size of a string value in MLMD db, `None` if unlimited."""

  @abc.abstractmethod
  def concurrent_pipeline_runs_enabled(self) -> bool:
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


class _DefaultEnv(Env):
  """Default environment."""

  def get_orchestration_options(
      self, pipeline: pipeline_pb2.Pipeline
  ) -> orchestration_options.OrchestrationOptions:
    del pipeline
    return orchestration_options.OrchestrationOptions()

  def get_base_dir(self) -> Optional[str]:
    return None

  def max_mlmd_str_value_length(self) -> Optional[int]:
    return None

  def concurrent_pipeline_runs_enabled(self) -> bool:
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


_ENV = _DefaultEnv()


def get_env() -> Env:
  return _ENV
