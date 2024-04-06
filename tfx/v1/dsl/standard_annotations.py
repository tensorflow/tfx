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
"""Public API for base type annotations."""

from tfx.types import system_artifacts as _system_artifacts
from tfx.types import system_executions as _system_executions

# List of MLMD base artifact type annotations.
Dataset = _system_artifacts.Dataset
Model = _system_artifacts.Model
Statistics = _system_artifacts.Statistics
Metrics = _system_artifacts.Metrics

# List of MLMD base execution type annotations.
Train = _system_executions.Train
Transform = _system_executions.Transform
Process = _system_executions.Process
Evaluate = _system_executions.Evaluate
Deploy = _system_executions.Deploy

del _system_artifacts
del _system_executions
