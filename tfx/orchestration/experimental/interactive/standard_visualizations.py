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

import os


import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma

from tfx import types
from tfx.orchestration.experimental.interactive import visualizations
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import io_utils
from tensorflow_metadata.proto.v0 import anomalies_pb2


class ExampleAnomaliesVisualization(visualizations.ArtifactVisualization):
  """Visualization for standard_artifacts.ExampleAnomalies."""

  ARTIFACT_TYPE = standard_artifacts.ExampleAnomalies

  def display(self, artifact: types.Artifact):
    from IPython.core.display import display  # pylint: disable=g-import-not-at-top
    from IPython.core.display import HTML  # pylint: disable=g-import-not-at-top
    for split in artifact_utils.decode_split_names(artifact.split_names):
      display(HTML('<div><b>%r split:</b></div><br/>' % split))
      anomalies_path = io_utils.get_only_uri_in_dir(
          artifact_utils.get_split_uri([artifact], split))
      if artifact_utils.is_artifact_version_older_than(
          artifact, artifact_utils._ARTIFACT_VERSION_FOR_ANOMALIES_UPDATE):  # pylint: disable=protected-access
        anomalies = tfdv.load_anomalies_text(anomalies_path)
      else:
        anomalies = anomalies_pb2.Anomalies()
        anomalies_bytes = io_utils.read_bytes_file(anomalies_path)
        anomalies.ParseFromString(anomalies_bytes)
      tfdv.display_anomalies(anomalies)


class ExampleStatisticsVisualization(visualizations.ArtifactVisualization):
  """Visualization for standard_artifacts.Statistics."""

  ARTIFACT_TYPE = standard_artifacts.ExampleStatistics

  def display(self, artifact: types.Artifact):
    from IPython.core.display import display  # pylint: disable=g-import-not-at-top
    from IPython.core.display import HTML  # pylint: disable=g-import-not-at-top
    for split in artifact_utils.decode_split_names(artifact.split_names):
      display(HTML('<div><b>%r split:</b></div><br/>' % split))
      stats_path = io_utils.get_only_uri_in_dir(
          artifact_utils.get_split_uri([artifact], split))
      if artifact_utils.is_artifact_version_older_than(
          artifact, artifact_utils._ARTIFACT_VERSION_FOR_STATS_UPDATE):  # pylint: disable=protected-access
        stats = tfdv.load_statistics(stats_path)
      else:
        stats = tfdv.load_stats_binary(stats_path)
      tfdv.visualize_statistics(stats)


class ModelEvaluationVisualization(visualizations.ArtifactVisualization):
  """Visualization for standard_artifacts.ModelEvaluation."""

  ARTIFACT_TYPE = standard_artifacts.ModelEvaluation

  def display(self, artifact: types.Artifact):
    tfma_result = tfma.load_eval_result(artifact.uri)
    # TODO(ccy): add comment instructing user to use the TFMA library directly
    # in order to render non-default slicing metric views.
    tfma.view.render_slicing_metrics(tfma_result)


class SchemaVisualization(visualizations.ArtifactVisualization):
  """Visualization for standard_artifacts.Schema."""

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
