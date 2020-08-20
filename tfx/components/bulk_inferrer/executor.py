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

import importlib
import os
from typing import Any, Dict, List, Mapping, Text

from absl import logging
import apache_beam as beam
import tensorflow as tf

from google.protobuf import json_format
from tensorflow_serving.apis import prediction_log_pb2
from tfx import types
from tfx.components.base import base_executor
from tfx.components.util import model_utils
from tfx.proto import bulk_inferrer_pb2
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils


_PREDICTION_LOGS_DIR_NAME = 'prediction_logs'


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
        - model_blessing: model blessing result
      output_dict: Output dict from output key to a list of Artifacts.
        - output: bulk inference results.
      exec_properties: A dict of execution properties.
        - model_spec: JSON string of bulk_inferrer_pb2.ModelSpec instance.
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
    if 'model_blessing' in input_dict:
      model_blessing = artifact_utils.get_single_instance(
          input_dict['model_blessing'])
      if not model_utils.is_model_blessed(model_blessing):
        output.set_int_custom_property('inferred', 0)
        logging.info('Model on %s was not blessed', model_blessing.uri)
        return
    else:
      logging.info('Model blessing is not provided, exported model will be '
                   'used.')

    model = artifact_utils.get_single_instance(
        input_dict['model'])
    model_path = path_utils.serving_model_path(model.uri)
    logging.info('Use exported model from %s.', model_path)

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
    model_spec = bulk_inferrer_pb2.ModelSpec()
    json_format.Parse(exec_properties['model_spec'], model_spec)
    output_path = os.path.join(output.uri, _PREDICTION_LOGS_DIR_NAME)
    self._run_model_inference(model_path, example_uris, output_path,
                              model_spec)
    logging.info('BulkInferrer generates prediction log to %s', output_path)
    output.set_int_custom_property('inferred', 1)

  def _run_model_inference(self, model_path: Text,
                           example_uris: Mapping[Text, Text],
                           output_path: Text,
                           model_spec: bulk_inferrer_pb2.ModelSpec) -> None:
    """Runs model inference on given example data.

    Args:
      model_path: Path to model.
      example_uris: Mapping of example split name to example uri.
      output_path: Path to output generated prediction logs.
      model_spec: bulk_inferrer_pb2.ModelSpec instance.

    Returns:
      None
    """

    try:
      from tfx_bsl.public.beam import run_inference
      from tfx_bsl.public.proto import model_spec_pb2
    except ImportError:
      # TODO(b/151468119): Remove this branch after next release.
      run_inference = importlib.import_module('tfx_bsl.beam.run_inference')
      model_spec_pb2 = importlib.import_module('tfx_bsl.proto.model_spec_pb2')
    saved_model_spec = model_spec_pb2.SavedModelSpec(
        model_path=model_path,
        tag=model_spec.tag,
        signature_name=model_spec.model_signature_name)
    # TODO(b/151468119): Remove this branch after next release.
    if getattr(model_spec_pb2, 'InferenceEndpoint', False):
      inference_endpoint = getattr(model_spec_pb2, 'InferenceEndpoint')()
    else:
      inference_endpoint = model_spec_pb2.InferenceSpecType()
    inference_endpoint.saved_model_spec.CopyFrom(saved_model_spec)
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
