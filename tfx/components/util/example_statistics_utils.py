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
"""Utilities for ExamplesStatistics aritfact type."""

from typing import Optional
from tfx.types import artifact as _artifact
from tfx.types import artifact_utils


def get_stats_path(artifact: _artifact.Artifact,
                   split_name: Optional[str] = None) -> str:
  """Returns the path that actual statistics files are located."""
  if split_name is None:
    return artifact.uri
  return artifact_utils.get_split_uri([artifact], split_name)
