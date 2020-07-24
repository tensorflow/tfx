# Lint as: python3
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
"""Stats Options for customizing TFDV in TFT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_data_validation as tfdv


# An instance of `tfdv.StatsOptions()` used when computing pre-transform
# statistics. If not specified, default options are used.
_PRE_TRANSFORM_STATS_OPTIONS = None

# An instance of `tfdv.StatsOptions()` used when computing post-transform
# statistics. If not specified, default options are used.
_POST_TRANSFORM_STATS_OPTIONS = None


def set_pre_transform_stats_options(stats_options: tfdv.StatsOptions):
  global _PRE_TRANSFORM_STATS_OPTIONS
  _PRE_TRANSFORM_STATS_OPTIONS = stats_options


def set_post_transform_stats_options(stats_options: tfdv.StatsOptions):
  global _POST_TRANSFORM_STATS_OPTIONS
  _POST_TRANSFORM_STATS_OPTIONS = stats_options


def get_pre_transform_stats_options() -> tfdv.StatsOptions:
  return (tfdv.StatsOptions() if _PRE_TRANSFORM_STATS_OPTIONS is None
          else _PRE_TRANSFORM_STATS_OPTIONS)


def get_post_transform_stats_options() -> tfdv.StatsOptions:
  return (tfdv.StatsOptions() if _POST_TRANSFORM_STATS_OPTIONS is None
          else _POST_TRANSFORM_STATS_OPTIONS)
