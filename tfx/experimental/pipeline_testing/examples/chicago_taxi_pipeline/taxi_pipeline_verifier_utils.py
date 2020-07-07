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
"""Helper utils for verifier"""
import os
import absl
import tensorflow as tf

from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration import metadata
from tfx.types import artifact_utils

def get_component_output_map(metadata_connection_config, pipeline_info):
  """returns a dictionary of component_id: output"""
  output_map = {}
  with metadata.Metadata(metadata_connection_config) as m:
    context = m.get_pipeline_run_context(pipeline_info)
    for execution in m.store.get_executions_by_context(context.id):
      component_id = execution.properties['component_id'].string_value
      # output_config = execution.properties['output_config'].string_value
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

def _group_metric_by_slice(eval_result_metric):
  slice_map = {}
  for metric in eval_result_metric:
    slice_map[metric[0]] = {k: v['doubleValue'] \
                              for k, v in metric[1][''][''].items()}
  return slice_map

def compare_relative_difference(value, expected_value, threshold):
  """comparing relative difference to threshold"""
  if value != expected_value:
    if expected_value:
      relative_diff = abs(value - expected_value)/abs(expected_value)
      if not (expected_value and relative_diff <= threshold):
        absl.logging.warning(
            "Relative difference {} exceeded threshold {}"\
                                        .format(relative_diff, threshold))
        return False
  return True

def compare_eval_results(eval_result, expected_eval_result, threshold):
  """comparing eval_results"""
  eval_slicing_metrics = eval_result.slicing_metrics
  expected_slicing_metrics = expected_eval_result.slicing_metrics
  slice_map = _group_metric_by_slice(eval_slicing_metrics)
  expected_slice_map = _group_metric_by_slice(expected_slicing_metrics)
  for slice_item, metric_dict in slice_map.items():
    for metric_name, value in metric_dict.items():
      if (slice_item not in expected_slice_map) \
          or metric_name not in expected_slice_map[slice_item]:
        print("metric_name", metric_name) # _diff?
        continue
      expected_value = expected_slice_map[slice_item][metric_name]
      if not compare_relative_difference(value, expected_value, threshold):
        return False
  return True

def compare_model_file_sizes(model_dir, expected_model_dir, threshold):
  """comparing sizes of saved models"""
  serving_model_dir = os.path.join(model_dir, 'serving_model_dir')

  for leaf_file in tf.io.gfile.listdir(serving_model_dir):
    if leaf_file.startswith('model.ckpt') or leaf_file == 'graph.pbtxt':
      file_name = os.path.join(serving_model_dir, leaf_file)
      expected_file_name = file_name.replace(model_dir, expected_model_dir)
      if not compare_relative_difference(tf.io.gfile.GFile(file_name).size(),
                                  tf.io.gfile.GFile(expected_file_name).size(),
                                  threshold):
        return False

  eval_model_dir = os.path.join(model_dir, 'eval_model_dir')
  subdirs = tf.io.gfile.listdir(eval_model_dir)
  assert len(subdirs) == 1
  eval_path = os.path.join(serving_model_dir, subdirs[0])
  for dir_name, _, leaf_files in tf.io.gfile.walk(eval_path):
    for leaf_file in leaf_files:
      file_name = os.path.join(dir_name, leaf_file)
      expected_file_name = file_name.replace(model_dir, expected_model_dir)
      if not compare_relative_difference(
          tf.io.gfile.GFile(file_name).size(),
          tf.io.gfile.GFile(expected_file_name).size(),
          threshold):
        return False
  return True
