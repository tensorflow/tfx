# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Experimental Resolver for getting the latest blessed model."""

from tfx.dsl.input_resolution.strategies import latest_blessed_model_strategy
from tfx.utils import deprecation_utils

LatestBlessedModelResolver = deprecation_utils.deprecated_alias(
    'tfx.dsl.experimental.latest_blessed_model_resolver.LatestBlessedModelResolver',
    'tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy.LatestBlessedModelStrategy',
    latest_blessed_model_strategy.LatestBlessedModelStrategy)
