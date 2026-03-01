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
"""Utilities for writing Artifacts to disk."""

from tfx.utils import io_utils
from tensorflow_metadata.proto.v0 import anomalies_pb2


def write_anomalies(
    filepath: str,
    anomalies: anomalies_pb2.Anomalies,
) -> None:
  """Writes Anomalies to a binary proto file."""
  io_utils.write_bytes_file(
      filepath, anomalies.SerializeToString()
  )
