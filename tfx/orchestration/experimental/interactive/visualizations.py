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
"""TFX notebook visualizations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Text, Type

# Standard Imports

from six import with_metaclass

from tfx import types
from tfx.utils import abc_utils


class ArtifactVisualization(with_metaclass(abc.ABCMeta)):
  """Visualization for a certain type of Artifact."""

  # Artifact type (of type `Type[types.Artifact]`) to which the visualization
  # applies.
  ARTIFACT_TYPE = abc_utils.abstract_property()

  @abc.abstractmethod
  def display(self, artifact: types.Artifact) -> Text:
    """Returns HTML string rendering artifact, in a notebook environment."""
    raise NotImplementedError()


class ArtifactVisualizationRegistry(object):
  """Registry of artifact visualizations."""

  def __init__(self):
    self.visualizations = {}

  def register(self, visualization_class: Type[ArtifactVisualization]):
    artifact_type = visualization_class.ARTIFACT_TYPE
    if not (issubclass(artifact_type, types.Artifact) and
            artifact_type.TYPE_NAME is not None):
      raise TypeError(
          'Visualization class must provide subclass of types.Artifact in its '
          'ARTIFACT_TYPE attribute. This subclass must have non-None TYPE_NAME '
          'attribute.')
    self.visualizations[artifact_type.TYPE_NAME] = visualization_class()  # pytype: disable=not-instantiable

  def get_visualization(self, artifact_type_name):
    return self.visualizations.get(artifact_type_name, None)


_REGISTRY = ArtifactVisualizationRegistry()


def get_registry():
  return _REGISTRY
