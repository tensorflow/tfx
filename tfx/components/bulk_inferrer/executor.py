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
"""TFX bulk_inferrer executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
from typing import Any, Dict, List, Mapping, Text

from absl import logging
import apache_beam as beam
import tensorflow as tf

from tfx import types
from tfx.components.base import base_executor
from tfx.components.util import model_utils
from tfx.proto import bulk_inferrer_pb2
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils
from tfx_bsl.public.beam import run_inference
from tfx_bsl.public.proto import model_spec_pb2
from google.protobuf import json_format
# TODO(b/140306674): stop using the internal TF API.
from tensorflow.python.saved_model import loader_impl
from tensorflow_serving.apis import prediction_log_pb2

_PREDICTION_LOGS_DIR_NAME = 'prediction_logs'
_CLOUD_PUSH_DESTINATION_RE = re.compile(
    r'^projects\/([^\/]+)\/models\/([^\/]+)\/versions\/([^\/]+)$')


class Executor(base_executor.BaseExecutor):
  """TFX bulk inferer executor."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Runs batch inference on a given model with given input examples.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - examples: examples for inference.
        - model: exported model.
        - model_blessing: model blessing result, optinal.
        - pushed_model: pushed model result, optional.
      output_dict: Output dict from output key to a list of Artifacts.
        - output: bulk inference results.
      exec_properties: A dict of execution properties.
        - model_spec: JSON string of bulk_inferrer_pb2.ModelSpec instance,
                      required for in-memory inference.
        - data_spec: JSON string of bulk_inferrer_pb2.DataSpec instance.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    if 'examples' not in input_dict:
      raise ValueError('\'examples\' is missing in input dict.')
    if 'inference_result' not in output_dict:
      raise ValueError('\'inference_result\' is missing in output dict.')
    output = artifact_utils.get_single_instance(output_dict['inference_result'])
    if 'model' not in input_dict:
      raise ValueError('Input models are not valid, model '
                       'need to be specified.')

    pushed_model = None
    if 'pushed_model' in input_dict:
      pushed_model = artifact_utils.get_single_instance(
          input_dict['pushed_model'])
      if not model_utils.is_model_pushed(pushed_model):
        output.set_int_custom_property('inferred', 0)
        logging.info('Model on %s was not pushed successfully',
                     pushed_model.uri)
        return
    elif 'model_blessing' in input_dict:
      model_blessing = artifact_utils.get_single_instance(
          input_dict['model_blessing'])
      if not model_utils.is_model_blessed(model_blessing):
        output.set_int_custom_property('inferred', 0)
        logging.info('Model on %s was not blessed', model_blessing.uri)
        return
    else:
      logging.info('Exported model will be used for inference.')

    model = artifact_utils.get_single_instance(
        input_dict['model'])
    model_path = path_utils.serving_model_path(model.uri)
    inference_endpoint = model_spec_pb2.InferenceSpecType()
    if pushed_model:
      pushed_destination = pushed_model.get_string_custom_property(
          'pushed_destination')
      matched = _CLOUD_PUSH_DESTINATION_RE.match(pushed_destination)
      if matched:
        ai_platform_prediction_model_spec = (
            model_spec_pb2.AIPlatformPredictionModelSpec(
                project_id=matched.group(1),
                model_name=matched.group(2),
                version_name=matched.group(3)))
        # TODO(b/155325467): Remove the if check after next release of tfx_bsl.
        if hasattr(ai_platform_prediction_model_spec,
                   'use_serialization_config'):
          model_signature = self._get_model_signature(model_path)
          if (len(model_signature.inputs) == 1 and
              list(model_signature.inputs.values())[0].dtype ==
              tf.string.as_datatype_enum):
            ai_platform_prediction_model_spec.use_serialization_config = True
        logging.info('Use hosted model on Cloud AI platform.')
        inference_endpoint.ai_platform_prediction_model_spec.CopyFrom(
            ai_platform_prediction_model_spec)
    else:
      logging.info('Use exported model from %s.', model_path)
      model_spec = bulk_inferrer_pb2.ModelSpec()
      json_format.Parse(exec_properties['model_spec'], model_spec)
      saved_model_spec = model_spec_pb2.SavedModelSpec(
          model_path=model_path,
          tag=model_spec.tag,
          signature_name=model_spec.model_signature_name)
      inference_endpoint.saved_model_spec.CopyFrom(saved_model_spec)

    data_spec = bulk_inferrer_pb2.DataSpec()
    json_format.Parse(exec_properties['data_spec'], data_spec)
    example_uris = {}
    if data_spec.example_splits:
      for example in input_dict['examples']:
        for split in artifact_utils.decode_split_names(example.split_names):
          if split in data_spec.example_splits:
            example_uris[split] = os.path.join(example.uri, split)
    else:
      for example in input_dict['examples']:
        for split in artifact_utils.decode_split_names(example.split_names):
          example_uris[split] = os.path.join(example.uri, split)

    output_path = os.path.join(output.uri, _PREDICTION_LOGS_DIR_NAME)
    self._run_model_inference(example_uris, output_path, inference_endpoint)
    logging.info('BulkInferrer generates prediction log to %s', output_path)
    output.set_int_custom_property('inferred', 1)

  def _run_model_inference(
      self, example_uris: Mapping[Text, Text], output_path: Text,
      inference_endpoint: model_spec_pb2.InferenceSpecType) -> None:
    """Runs model inference on given example data.

    Args:
      example_uris: Mapping of example split name to example uri.
      output_path: Path to output generated prediction logs.
      inference_endpoint: Model inference endpoint.

    Returns:
      None
    """

    with self._make_beam_pipeline() as pipeline:
      data_list = []
      for split, example_uri in example_uris.items():
        data = (
            pipeline | 'ReadData[{}]'.format(split) >> beam.io.ReadFromTFRecord(
                file_pattern=io_utils.all_files_pattern(example_uri)))
        data_list.append(data)
      _ = (
          [data for data in data_list]
          | 'FlattenExamples' >> beam.Flatten(pipeline=pipeline)
          | 'ParseExamples' >> beam.Map(tf.train.Example.FromString)
          | 'RunInference' >> run_inference.RunInference(inference_endpoint)
          | 'WritePredictionLogs' >> beam.io.WriteToTFRecord(
              output_path,
              file_name_suffix='.gz',
              coder=beam.coders.ProtoCoder(prediction_log_pb2.PredictionLog)))
    logging.info('Inference result written to %s.', output_path)

  def _get_model_signature(self, model_path: Text) -> Any:
    """Returns a model signature."""

    saved_model_pb = loader_impl.parse_saved_model(model_path)
    meta_graph_def = None
    for graph_def in saved_model_pb.meta_graphs:
      if graph_def.meta_info_def.tags == [
          tf.compat.v1.saved_model.tag_constants.SERVING
      ]:
        meta_graph_def = graph_def
    if not meta_graph_def:
      raise RuntimeError(
          'Tag tf.compat.v1.saved_model.tag_constants.SERVING'
          ' does not exist in saved model: %s. This is required'
          ' for remote inference.' % model_path)
    if tf.saved_model.PREDICT_METHOD_NAME in meta_graph_def.signature_def:
      return meta_graph_def.signature_def[tf.saved_model.PREDICT_METHOD_NAME]
    if (tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY in
        meta_graph_def.signature_def):
      return meta_graph_def.signature_def[
          tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    raise RuntimeError(
        'Cannot find serving signature in saved model: %s,'
        ' tf.saved_model.PREDICT_METHOD_NAME or '
        ' tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY is needed.' %
        model_path)
