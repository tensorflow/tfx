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
"""Subpackage for TFX pipeline types.

DO NOT USE THIS MODULE DIRECTLY. This module is a private module, please use the
public symbols from `tfx.v1`:

- `tfx.v1.dsl.Artifact`
- `tfx.v1.types.BaseChannel`
- `Channel` is now deprecated and should not be used directly.
- `OutputChannel`, `ExecPropertyTypes`, `Property`, `ComponentSpec`,
  `ValueArtifact` is not meant to be public.
"""

from tfx.types.artifact import Artifact
from tfx.types.channel import (
    BaseChannel,
    Channel,
    ExecPropertyTypes,
    OutputChannel,
    Property,
)
from tfx.types.component_spec import ComponentSpec
from tfx.types.value_artifact import ValueArtifact

__all__ = [
    "Artifact",
    "BaseChannel",
    "Channel",
    "ComponentSpec",
    "ExecPropertyTypes",
    "OutputChannel",
    "Property",
    "ValueArtifact",
]
