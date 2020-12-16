# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Utilities to be used by user code when invoking Transform."""

import enum


# Enum used in stats_options_updater_fn to specify which stats are being
# updated.
class StatsType(enum.Enum):
  UNKNOWN = 0
  PRE_TRANSFORM = 1
  POST_TRANSFORM = 2
