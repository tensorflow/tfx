# Lint as: python2, python3
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
"""Subpackage for TFX pipeline types."""

from typing import Text, Union

from tfx.types.artifact import Artifact
from tfx.types.artifact import ValueArtifact
from tfx.types.channel import Channel
from tfx.types.component_spec import ComponentSpec

# Property type for artifacts, executions and contexts.
Property = Union[int, float, Text]
