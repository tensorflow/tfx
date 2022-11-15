# Copyright 2022 Google LLC. All Rights Reserved.
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
"""Portable APIs for managing artifacts in MLMD."""
from typing import Iterable, Optional

from tfx.orchestration import metadata
from tfx.types import artifact as artifact_lib


def update_artifacts(metadata_handler: metadata.Metadata,
                     tfx_artifacts: Iterable[artifact_lib.Artifact],
                     new_artifact_state: Optional[str] = None) -> None:
  """Updates existing TFX artifacts in MLMD."""
  mlmd_artifacts_to_update = []
  for tfx_artifact in tfx_artifacts:
    if not tfx_artifact.mlmd_artifact.HasField('id'):
      raise ValueError('Artifact must have an MLMD ID in order to be updated.')
    if new_artifact_state:
      tfx_artifact.state = new_artifact_state
    mlmd_artifacts_to_update.append(tfx_artifact.mlmd_artifact)
  if mlmd_artifacts_to_update:
    metadata_handler.store.put_artifacts(mlmd_artifacts_to_update)
