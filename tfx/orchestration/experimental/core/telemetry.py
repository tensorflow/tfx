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
"""Telemetry logging functionality."""

import abc

from tfx.proto.orchestration import pipeline_pb2


class TelemetryLoggerBase(abc.ABC):
  """Base class for environment specific telemetry logging functionality."""

  @abc.abstractmethod
  def log_component_run(self, pipeline: pipeline_pb2.Pipeline) -> None:
    """Logs component run. Should be invoked after successful component run."""


class DummyTelemetryLogger(TelemetryLoggerBase):
  """Dummy telemetry logger does nothing."""

  def log_component_run(self, pipeline: pipeline_pb2.Pipeline) -> None:
    pass


TELEMETRY_LOGGER = DummyTelemetryLogger()
