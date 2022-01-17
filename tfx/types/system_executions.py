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
"""A set of TFX System Executions.

It matches the MLMD system execution types from:
third_party/ml_metadata/metadata_store/mlmd_types.py
"""
import abc

from ml_metadata.metadata_store import mlmd_types


class SystemExecution(abc.ABC):
  """TFX system execution base class.

  A user may create a subclass of SystemExecution and override the
  MLMD_SYSTEM_BASE_TYPE property with the MLMD system type enum.

  The subclasses, e.g, Train, Transform, Process, e.t.c, match the MLMD types
  from third_party/ml_metadata/metadata_store/mlmd_types.py.
  """

  # MLMD system base type enum. Override it when creating subclasses.
  MLMD_SYSTEM_BASE_TYPE = None


class Train(SystemExecution):
  """Train is a TFX pre-defined system execution.

  Train is one of the key executions that performs the actual model training.
  """
  MLMD_SYSTEM_BASE_TYPE = mlmd_types.Train().system_type


class Transform(SystemExecution):
  """Transform is a TFX pre-defined system execution.

  It performs transformations and feature engineering in training and serving.
  """
  MLMD_SYSTEM_BASE_TYPE = mlmd_types.Transform().system_type


class Process(SystemExecution):
  """Process is a TFX pre-defined system execution.

  It includes various executions such as ExampleGen, SchemaGen, SkewDetection,
  e.t.c., which performs data/model/statistics processing.
  """
  MLMD_SYSTEM_BASE_TYPE = mlmd_types.Process().system_type


class Evaluate(SystemExecution):
  """Evaluate is a TFX pre-defined system execution.

  It computes a model’s evaluation statistics over (slices of) features.
  """
  MLMD_SYSTEM_BASE_TYPE = mlmd_types.Evaluate().system_type


class Deploy(SystemExecution):
  """Deploy is a TFX pre-defined system execution.

  This execution performs model deployment. For example, Pusher component can be
  annotated as Deploy execution, which checks whether the model passed the
  validation steps and pushes fully validated models to Servomatic, CNS/Placer,
  TF-Hub, and other destinations.
  """
  MLMD_SYSTEM_BASE_TYPE = mlmd_types.Deploy().system_type
