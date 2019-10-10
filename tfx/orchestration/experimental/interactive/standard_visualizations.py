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
"""TFX notebook visualizations for standard TFX artifacts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Standard Imports

import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma

from tfx import types
from tfx.orchestration.experimental.interactive import visualizations
from tfx.types import standard_artifacts


class ExampleAnomaliesVisualization(visualizations.ArtifactVisualization):

  ARTIFACT_TYPE = standard_artifacts.ExampleAnomalies

  def display(self, artifact: types.Artifact):
    anomalies_path = os.path.join(artifact.uri, 'anomalies.pbtxt')
    anomalies = tfdv.load_anomalies_text(anomalies_path)
    tfdv.display_anomalies(anomalies)


class ExampleStatisticsVisualization(visualizations.ArtifactVisualization):

  ARTIFACT_TYPE = standard_artifacts.ExampleStatistics

  def display(self, artifact: types.Artifact):
    stats_path = os.path.join(artifact.uri, 'stats_tfrecord')
    stats = tfdv.load_statistics(stats_path)
    tfdv.visualize_statistics(stats)


class ModelEvaluationVisualization(visualizations.ArtifactVisualization):

  ARTIFACT_TYPE = standard_artifacts.ModelEvaluation

  def display(self, artifact: types.Artifact):
    tfma_result = tfma.load_eval_result(artifact.uri)
    # TODO(ccy): add comment instructing user to use the TFMA library directly
    # in order to render non-default slicing metric views.
    tfma.view.render_slicing_metrics(tfma_result)


class SchemaVisualization(visualizations.ArtifactVisualization):

  ARTIFACT_TYPE = standard_artifacts.Schema

  def display(self, artifact: types.Artifact):
    schema_path = os.path.join(artifact.uri, 'schema.pbtxt')
    schema = tfdv.load_schema_text(schema_path)
    tfdv.display_schema(schema)


STANDARD_VISUALIZATIONS = frozenset([
    ExampleAnomaliesVisualization,
    ExampleStatisticsVisualization,
    ModelEvaluationVisualization,
    SchemaVisualization,
])


def register_standard_visualizations():
  for visualization in STANDARD_VISUALIZATIONS:
    visualizations.get_registry().register(visualization)
