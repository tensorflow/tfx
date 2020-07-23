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
from typing import Dict, List, Text, Optional

from tensorflow_model_analysis.view import SlicedMetrics
from tensorflow_model_analysis.view import view_types
from ml_metadata.proto import metadata_store_pb2

from tfx import types
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.types import artifact_utils

def get_component_output_map(
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

def compare_relative_difference(value: float,
                                expected_value: float,
                                threshold: float) -> bool:
  """Compares relative difference between value and expected_value to 
    a threshold.
  Args:
    value:
    expected_value:
    threshold:

  Returns:
    a boolean whether 
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

def compare_eval_results(eval_result: view_types.EvalResult,
                         expected_eval_result: view_types.EvalResult,
                         threshold: float) -> bool:
  """Compares two EvalResult.
  Args:
    eval_result:
    expected_eval_result:
    threshold:

  Returns:
    boolean whether the eval result values are similar within a threshold.
  """
  eval_slicing_metrics = eval_result.slicing_metrics
  expected_slicing_metrics = expected_eval_result.slicing_metrics
  slice_map = _group_metric_by_slice(eval_slicing_metrics)
  expected_slice_map = _group_metric_by_slice(expected_slicing_metrics)
  for slice_item, metric_dict in slice_map.items():
    for metric_name, value in metric_dict.items():
      if (slice_item not in expected_slice_map) \
          or metric_name not in expected_slice_map[slice_item]:
        print("metric_name", metric_name)
        continue
      expected_value = expected_slice_map[slice_item][metric_name]
      if not compare_relative_difference(value, expected_value, threshold):
        return False
  return True

def compare_model_file_sizes(model_dir: Text,
                             expected_model_dir: Text,
                             threshold: float) -> bool:
  """Comparing sizes of output and recorded model.
  
  Args:
    model_dir: directory to model
    expected_model_dir: directory to saved model
    threshold: a float between 0 and 1

  Returns:
    returns whether the sizes of models are within the threshold
  """
  serving_model_dir = os.path.join(model_dir, 'serving_model_dir')

  for leaf_file in tf.io.gfile.listdir(serving_model_dir):
    if leaf_file.startswith('model.ckpt') or leaf_file == 'graph.pbtxt':
      file_name = os.path.join(serving_model_dir, leaf_file)
      expected_file_name = file_name.replace(model_dir, expected_model_dir)
      if not compare_relative_difference(
          tf.io.gfile.GFile(file_name).size(),
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

def verify_example_gen(self):
  self.assertTrue(tf.io.gfile.exists(os.path.join(self._tmp_dir)))
  self.assertTrue(
      tf.io.gfile.exists(
          os.path.join(self._blessing.uri, constants.BLESSED_FILE_NAME)))

def verify_statistics_gen(self):
  for record_dir.

def verify_schema_gen(self):
  tf.io.gfile.exists()

def verify_validator(self):
  self.assertEqual(['anomalies.pbtxt'],
                   tf.io.gfile.listdir(validation_output.uri))
  anomalies = io_utils.parse_pbtxt_file(
      os.path.join(validation_output.uri, 'anomalies.pbtxt'),
      anomalies_pb2.Anomalies())
  self.assertNotEqual(0, len(anomalies.anomaly_info))

def verify_trainer(self, output_dict: Dict[Text, List[types.Artifact]]):
  # compares two model files


  # Check example gen outputs.
  self.assertTrue(tf.io.gfile.exists(self._train_output_file))
  self.assertTrue(tf.io.gfile.exists(self._eval_output_file))


  self._verify_model_exports()
  self._verify_model_run_exports()

  self._verify_no_eval_model_exports()
  self._verify_model_run_exports()

  # eval_model_dir exists but not model_run

  gdef1 = gpb.GraphDef()

  with open("/Users/sujipark/tfx/pipelines/chicago_taxi_beam/Trainer/model/7/serving_model_dir/graph.pbtxt", 'r') as fh:
      graph_str = fh.read()

  pbtf.Parse(graph_str, gdef1)

  gdef2 = gpb.GraphDef()

  with open('/Users/sujipark/tfx/testtest/Trainer/model/serving_model_dir/graph.pbtxt',  'r') as fh:
      graph_str = fh.read()


  pbtf.Parse(graph_str, gdef2)
  [n.name for n in gdef2.node] == [n.name for n in gdef1.node]

  x= tf.saved_model.load("/Users/sujipark/tfx/imdb_testdata/Trainer/model/serving_model_dir")
  y= tf.saved_model.load("/Users/sujipark/tfx/pipelines/imdb_native_keras/Trainer/model/26/serving_model_dir")
  [v.name for v in x.variables] == [v.name for v in y.variables]

  absl.logging.info("verifying Trainer")
  model_artifact = output_dict['model']
  model_uri = model_artifact.uri

  path = os.path.join(self._record_dir, 'Trainer', 'model')
  self.assertTrue(taxi_pipeline_verifier_utils.compare_model_file_sizes(
      model_uri,
      path,
      self._threshold))

def verify_evaluator(self, output_dict: Dict[Text, List[types.Artifact]]):
  # compares two evaluation proto files.
  absl.logging.info("verifying Evaluator")
  eval_result = tfma.load_eval_result(output_dict['evaluation'].uri)
  expected_eval_result = tfma.load_eval_result(os.path.join(self._record_dir,
                                                            'Evaluator',
                                                            'evaluation'))
  self.assertTrue(taxi_pipeline_verifier_utils.compare_eval_results(
      eval_result,
      expected_eval_result,
      self._threshold))

def verify_validator(self, output_dict: Dict[Text, List[types.Artifact]]):
  # compares two validation proto files
  absl.logging.info("verifying Validator")
  anomalies = io_utils.parse_pbtxt_file(
      os.path.join(output_dict['anomalies'].uri, 'anomalies.pbtxt'),
      anomalies_pb2.Anomalies())
  expected_anomalies = io_utils.parse_pbtxt_file(
      os.path.join(self._record_dir,
                   'ExampleValidator',
                   'anomalies',
                   'anomalies.pbtxt'),
      anomalies_pb2.Anomalies())
  self.assertEqual(expected_anomalies.anomaly_info, anomalies.anomaly_info)

def verify_pusher(self):
  
  pass

def _verify_model_exports(self):
  self.assertTrue(
      tf.io.gfile.exists(path_utils.eval_model_dir(self._model_exports.uri)))
  self.assertTrue(
      tf.io.gfile.exists(
          path_utils.serving_model_dir(self._model_exports.uri)))

def _verify_no_eval_model_exports(self):
  self.assertFalse(
      tf.io.gfile.exists(path_utils.eval_model_dir(self._model_exports.uri)))

def _verify_model_run_exports(self):
  self.assertTrue(
      tf.io.gfile.exists(os.path.dirname(self._model_run_exports.uri)))

