# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Helper utils for executor verifier."""

import os
import absl
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tensorflow_metadata.proto.v0 import anomalies_pb2
from typing import Dict, List, Text, Optional

from tensorflow_model_analysis.view import SlicedMetrics
from tensorflow_model_analysis.view import view_types
from ml_metadata.proto import metadata_store_pb2
from tfx.components.trainer import constants
from tfx import types
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import artifact_utils
from tfx.utils import io_utils
def _compare_relative_difference(value: float,
                                expected_value: float,
                                threshold: float) -> bool:
  """Compares relative difference between value and expected_value to 
    a specified threshold.

  Args:
    value:
    expected_value:
    threshold: a float between 0 and 1

  Returns:
    a boolean indicating whether the relative difference is within the
    threshold
  """
  if value != expected_value:
    if expected_value:
      relative_diff = abs(value - expected_value)/abs(expected_value)
      if not (expected_value and relative_diff <= threshold):
        absl.logging.warning(
            "Relative difference {} exceeded threshold {}"\
                                        .format(relative_diff, threshold))
        return False
  return True

def get_pipeline_outputs(
    metadata_connection_config:
        Optional[metadata_store_pb2.ConnectionConfig],
    pipeline_info: data_types.PipelineInfo
    ) -> Dict[Text, Dict[Text, List[types.Artifact]]]:
  """Returns a dictionary of component_id: output.

  Args:
    metadata_connection_config:
    pipeline_info: pipeline info of 

  Returns:
    a dictionary of component_id: output.
  """
  output_map = {}
  with metadata.Metadata(metadata_connection_config) as m:
    context = m.get_pipeline_run_context(pipeline_info)
    for execution in m.store.get_executions_by_context(context.id):
      component_id = execution.properties['component_id'].string_value
      output_dict = {}
      for event in m.store.get_events_by_execution_ids([execution.id]):
        if event.type == metadata_store_pb2.Event.OUTPUT:
          artifacts = m.store.get_artifacts_by_id([event.artifact_id])
          for step in event.path.steps:
            if step.HasField("key"):
              output_dict[step.key] = artifact_utils.get_single_instance(
                  artifacts)
      output_map[component_id] = output_dict
  return output_map

def _verify_file_path(output_uri, expected_uri, check_file=False):
  """Verify the file directory paths."""
  for dir_name, sub_dirs, leaf_files in tf.io.gfile.walk(expected_uri):
    for sub_dir in sub_dirs:
      file_path = os.path.join(dir_name, sub_dir)
      new_file_path = os.path.join(dir_name.replace(expected_uri, output_uri, 1), sub_dir)
      if not tf.io.gfile.exists(new_file_path):
        return False
    if check_file:
      for leaf_file in leaf_files:
        leaf_file_path = os.path.join(dir_name, leaf_file)
        new_file_path = os.path.join(dir_name.replace(expected_uri, output_uri, 1), leaf_file)
        if not tf.io.gfile.exists(new_file_path):
          return False
  return True

def _group_metric_by_slice(eval_result_metric: List[SlicedMetrics]
                          ) -> Dict[Text, Dict[Text, float]]:
  """Returns a slice map.

  Args:
    eval_result_metric: list of sliced metrics.

  Returns:
    a slice map that holds a dictionary of metric and value for slices
  """
  slice_map = {}
  for metric in eval_result_metric:
    slice_map[metric[0]] = {k: v['doubleValue'] \
                              for k, v in metric[1][''][''].items()}
  return slice_map

def _compare_eval_results(output_uri: Text, 
                         expected_uri: Text,
                         threshold: float) -> bool:
  """Compares accuracy on overall dataset using two EvalResult.

  Args:
    eval_result: Result of a model analysis run.
    expected_eval_result: Result of a model analysis run.
    threshold: a float between 0 and 1

  Returns:
    boolean whether the eval result values are similar within a threshold.
  """
  eval_result = tfma.load_eval_result(output_uri)
  expected_eval_result = tfma.load_eval_result(expected_uri)
  eval_slicing_metrics = eval_result.slicing_metrics
  expected_slicing_metrics = expected_eval_result.slicing_metrics
  slice_map = _group_metric_by_slice(eval_slicing_metrics)
  expected_slice_map = _group_metric_by_slice(expected_slicing_metrics)

  for metric_name, value in slice_map[()].items():
    expected_value = expected_slice_map[()][metric_name]
    if not _compare_relative_difference(value, expected_value, threshold):
      return False
  return True

def _compare_file_sizes(output_uri: Text,
                        expected_uri: Text, 
                        threshold: float) -> bool:
  for leaf_file in tf.io.gfile.listdir(expected_uri):
    expected_file_name = os.path.join(expected_uri, leaf_file)
    file_name = expected_file_name.replace(expected_uri, output_uri)
    if not _compare_relative_difference(
        tf.io.gfile.GFile(file_name).size(),
        tf.io.gfile.GFile(expected_file_name).size(),
        threshold):
      return False
  return True

def verify(output_uri: Text, key: Text, artifact: Text, threshold: float) -> bool:
  """Default artifact verifier.

  Args: ...
  """
  artifact_name = artifact.custom_properties['name'].string_value

  # if artifact_name == constants.EXAMPLES_KEY:
  #   eval_output = standard_artifacts.ModelEvaluation()
  #   eval_output.uri = os.path.join(output_data_dir, 'eval_output')
  #   blessing_output = standard_artifacts.ModelBlessing()
  #   blessing_output.uri = os.path.join(output_data_dir, 'blessing_output')
  if artifact_name == constants.SCHEMA_KEY:
    if not _compare_file_sizes(artifact.uri, output_uri, threshold):
      return False

  if artifact_name == 'evaluation':
    if not _compare_eval_results(
        artifact.uri,
        output_uri,
        threshold):
      return False

  elif artifact_name in [constants.MODEL_KEY, constants.TRANSFORM_GRAPH_KEY]:
    if not _compare_file_sizes(artifact.uri, output_uri, threshold):
      return False

  return _verify_file_path(output_uri, artifact.uri)
