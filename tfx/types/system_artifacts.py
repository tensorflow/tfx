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
"""A set of TFX System Artifacts.

It matches the MLMD system artifacts types from:
third_party/ml_metadata/metadata_store/mlmd_types.py
"""
import abc

from ml_metadata.metadata_store import mlmd_types


class SystemArtifact(abc.ABC):
  """TFX system artifact base class.

  A user may create a subclass of SystemArtifact and override the
  MLMD_SYSTEM_BASE_TYPE property with the MLMD system type enum.

  The subclasses, e.g, Dataset, Model, Statistics, e.t.c, match the MLMD types
  from third_party/ml_metadata/metadata_store/mlmd_types.py.
  """

  # MLMD system base type enum. Override it when creating subclasses.
  MLMD_SYSTEM_BASE_TYPE = None


class Dataset(SystemArtifact):
  """Dataset is a TFX pre-defined system artifact."""
  MLMD_SYSTEM_BASE_TYPE = mlmd_types.Dataset().system_type


class Model(SystemArtifact):
  """Model is a TFX pre-defined system artifact."""
  MLMD_SYSTEM_BASE_TYPE = mlmd_types.Model().system_type


class Statistics(SystemArtifact):
  """Statistics is a TFX pre-defined system artifact."""
  MLMD_SYSTEM_BASE_TYPE = mlmd_types.Statistics().system_type


class Metrics(SystemArtifact):
  """Metrics is a TFX pre-defined system artifact."""
  MLMD_SYSTEM_BASE_TYPE = mlmd_types.Metrics().system_type
