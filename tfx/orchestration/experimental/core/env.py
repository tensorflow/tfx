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

from tfx.orchestration.experimental.core import orchestration_options
from tfx.proto.orchestration import pipeline_pb2

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


class _DefaultEnv(Env):

  def get_orchestration_options(
      self, pipeline: pipeline_pb2.Pipeline
  ) -> orchestration_options.OrchestrationOptions:
    del pipeline
    return orchestration_options.OrchestrationOptions()


_ENV = _DefaultEnv()


def get_env() -> Env:
  return _ENV
