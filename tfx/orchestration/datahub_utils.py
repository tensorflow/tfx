# Copyright 2024 Google LLC. All Rights Reserved.
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
"""Utils to log Tflex/MLMD entities."""
from typing import Optional

from tfx.orchestration.experimental.core import task as task_lib
from tfx.utils import typing_utils

from ml_metadata.proto import metadata_store_pb2


def log_node_execution(
    execution: metadata_store_pb2.Execution,
    task: Optional[task_lib.ExecNodeTask] = None,
    output_artifacts: Optional[typing_utils.ArtifactMultiMap] = None,
):
  """Logs a Tflex node execution and its input/output artifacts."""
  del execution, task, output_artifacts
  return
