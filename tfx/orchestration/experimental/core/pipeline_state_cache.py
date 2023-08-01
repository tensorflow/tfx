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
"""Cache to store status of pipeline state."""

import threading


class _LivePipelineCache:
  """Read-through/ write-through cache for whether to check active pipelines.

  If no live pipelines, value is false.
  If there is active pipeline or not sure, value is true.

  """

  def __init__(self):
    self._live_pipeline_or_check = True
    self._lock = threading.Lock()

  def get_check_signal(self) -> bool:
    with self._lock:
      return self._live_pipeline_or_check

  def update_check_signal(self, update_value: bool) -> None:
    with self._lock:
      self._live_pipeline_or_check = update_value


live_pipeline_cache = _LivePipelineCache()
pipeline_cache_update_lock = threading.Lock()
