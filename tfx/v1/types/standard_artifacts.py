# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Public API for standard_artifacts."""

from tfx.types import standard_artifacts

Examples = standard_artifacts.Examples
ExampleAnomalies = standard_artifacts.ExampleAnomalies
ExampleStatistics = standard_artifacts.ExampleStatistics
InferenceResult = standard_artifacts.InferenceResult
InfraBlessing = standard_artifacts.InfraBlessing
Model = standard_artifacts.Model
ModelRun = standard_artifacts.ModelRun
ModelBlessing = standard_artifacts.ModelBlessing
ModelEvaluation = standard_artifacts.ModelEvaluation
PushedModel = standard_artifacts.PushedModel
Schema = standard_artifacts.Schema
TransformCache = standard_artifacts.TransformCache
TransformGraph = standard_artifacts.TransformGraph
HyperParameters = standard_artifacts.HyperParameters

# Artifacts of small scalar-values.
Bytes = standard_artifacts.Bytes
Float = standard_artifacts.Float
Integer = standard_artifacts.Integer
String = standard_artifacts.String
