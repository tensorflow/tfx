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

import os
from typing import Any, Callable, Dict, List, Optional, Text, Union

from absl import logging
import apache_beam as beam
import tensorflow as tf
from tfx import types
from tfx.components.bulk_inferrer import prediction_to_example_utils
from tfx.components.util import model_utils
from tfx.components.util import tfxio_utils
from tfx.dsl.components.base import base_beam_executor
from tfx.proto import bulk_inferrer_pb2
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import path_utils
from tfx.utils import proto_utils
from tfx_bsl.public.beam import run_inference
from tfx_bsl.public.proto import model_spec_pb2
from tfx_bsl.tfxio import record_based_tfxio

from tensorflow_serving.apis import prediction_log_pb2

try:
  import tensorflow_text as _  # pylint: disable=g-import-not-at-top
except ImportError as e:
  logging.info('tensorflow_text is not available: %s', e)


_PREDICTION_LOGS_FILE_NAME = 'prediction_logs'
_EXAMPLES_FILE_NAME = 'examples'
_TELEMETRY_DESCRIPTORS = ['BulkInferrer']


class Executor(base_beam_executor.BaseBeamExecutor):
  """TFX bulk inferer executor."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Runs batch inference on a given model with given input examples.

    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - examples: examples for inference.
        - model: exported model.
        - model_blessing: model blessing result, optional.
      output_dict: Output dict from output key to a list of Artifacts.
        - output: bulk inference results.
      exec_properties: A dict of execution properties.
        - model_spec: JSON string of bulk_inferrer_pb2.ModelSpec instance.
        - data_spec: JSON string of bulk_inferrer_pb2.DataSpec instance.

    Returns:
      None
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    if output_dict.get(standard_component_specs.INFERENCE_RESULT_KEY):
      inference_result = artifact_utils.get_single_instance(
          output_dict[standard_component_specs.INFERENCE_RESULT_KEY])
    else:
      inference_result = None
    if output_dict.get(standard_component_specs.OUTPUT_EXAMPLES_KEY):
      output_examples = artifact_utils.get_single_instance(
          output_dict[standard_component_specs.OUTPUT_EXAMPLES_KEY])
    else:
      output_examples = None

    if 'examples' not in input_dict:
      raise ValueError('\'examples\' is missing in input dict.')
    if 'model' not in input_dict:
      raise ValueError('Input models are not valid, model '
                       'need to be specified.')
    if standard_component_specs.MODEL_BLESSING_KEY in input_dict:
      model_blessing = artifact_utils.get_single_instance(
          input_dict[standard_component_specs.MODEL_BLESSING_KEY])
      if not model_utils.is_model_blessed(model_blessing):
        logging.info('Model on %s was not blessed', model_blessing.uri)
        return
    else:
      logging.info('Model blessing is not provided, exported model will be '
                   'used.')

    model = artifact_utils.get_single_instance(
        input_dict[standard_component_specs.MODEL_KEY])
    model_path = path_utils.serving_model_path(
        model.uri, path_utils.is_old_model_artifact(model))
    logging.info('Use exported model from %s.', model_path)

    data_spec = bulk_inferrer_pb2.DataSpec()
    proto_utils.json_to_proto(
        exec_properties[standard_component_specs.DATA_SPEC_KEY], data_spec)

    output_example_spec = bulk_inferrer_pb2.OutputExampleSpec()
    if exec_properties.get(standard_component_specs.OUTPUT_EXAMPLE_SPEC_KEY):
      proto_utils.json_to_proto(
          exec_properties[standard_component_specs.OUTPUT_EXAMPLE_SPEC_KEY],
          output_example_spec)

    self._run_model_inference(
        data_spec, output_example_spec,
        input_dict[standard_component_specs.EXAMPLES_KEY], output_examples,
        inference_result, self._get_inference_spec(model_path, exec_properties))

  def _get_inference_spec(
      self, model_path: Text,
      exec_properties: Dict[Text, Any]) -> model_spec_pb2.InferenceSpecType:
    model_spec = bulk_inferrer_pb2.ModelSpec()
    proto_utils.json_to_proto(
        exec_properties[standard_component_specs.MODEL_SPEC_KEY], model_spec)
    saved_model_spec = model_spec_pb2.SavedModelSpec(
        model_path=model_path,
        tag=model_spec.tag,
        signature_name=model_spec.model_signature_name)
    result = model_spec_pb2.InferenceSpecType()
    result.saved_model_spec.CopyFrom(saved_model_spec)
    return result

  def _run_model_inference(
      self,
      data_spec: bulk_inferrer_pb2.DataSpec,
      output_example_spec: bulk_inferrer_pb2.OutputExampleSpec,
      examples: List[types.Artifact],
      output_examples: Optional[types.Artifact],
      inference_result: Optional[types.Artifact],
      inference_endpoint: model_spec_pb2.InferenceSpecType,
  ) -> None:
    """Runs model inference on given examples data.

    Args:
      data_spec: bulk_inferrer_pb2.DataSpec instance.
      output_example_spec: bulk_inferrer_pb2.OutputExampleSpec instance.
      examples: List of `standard_artifacts.Examples` artifacts.
      output_examples: Optional output `standard_artifacts.Examples` artifact.
      inference_result: Optional output `standard_artifacts.InferenceResult`
        artifact.
      inference_endpoint: Model inference endpoint.
    """

    example_uris = {}
    for example_artifact in examples:
      for split in artifact_utils.decode_split_names(
          example_artifact.split_names):
        if data_spec.example_splits:
          if split in data_spec.example_splits:
            example_uris[split] = artifact_utils.get_split_uri(
                [example_artifact], split)
        else:
          example_uris[split] = artifact_utils.get_split_uri([example_artifact],
                                                             split)

    payload_format, _ = tfxio_utils.resolve_payload_format_and_data_view_uri(
        examples)

    tfxio_factory = tfxio_utils.get_tfxio_factory_from_artifact(
        examples,
        _TELEMETRY_DESCRIPTORS,
        schema=None,
        read_as_raw_records=True,
        # We have to specify this parameter in order to create a RawRecord TFXIO
        # but we won't use the RecordBatches so the column name of the raw
        # records does not matter.
        raw_record_column_name='unused')

    if output_examples:
      output_examples.split_names = artifact_utils.encode_split_names(
          sorted(example_uris.keys()))

    with self._make_beam_pipeline() as pipeline:
      data_list = []
      for split, example_uri in example_uris.items():
        tfxio = tfxio_factory([io_utils.all_files_pattern(example_uri)])
        assert isinstance(tfxio, record_based_tfxio.RecordBasedTFXIO), (
            'Unable to use TFXIO {} as it does not support reading raw records.'
            .format(type(tfxio)))
        # pylint: disable=no-value-for-parameter
        data = (pipeline
                | 'ReadData[{}]'.format(split) >> tfxio.RawRecordBeamSource()
                | 'RunInference[{}]'.format(split) >> _RunInference(
                    payload_format, inference_endpoint))
        if output_examples:
          output_examples_split_uri = artifact_utils.get_split_uri(
              [output_examples], split)
          logging.info('Path of output examples split `%s` is %s.', split,
                       output_examples_split_uri)
          _ = (
              data
              | 'WriteExamples[{}]'.format(split) >> _WriteExamples(
                  output_example_spec, output_examples_split_uri))
          # pylint: enable=no-value-for-parameter

        data_list.append(data)

      if inference_result:
        _ = (
            data_list
            | 'FlattenInferenceResult' >> beam.Flatten(pipeline=pipeline)
            | 'WritePredictionLogs' >> beam.io.WriteToTFRecord(
                os.path.join(inference_result.uri, _PREDICTION_LOGS_FILE_NAME),
                file_name_suffix='.gz',
                coder=beam.coders.ProtoCoder(prediction_log_pb2.PredictionLog)))

    if output_examples:
      logging.info('Output examples written to %s.', output_examples.uri)
    if inference_result:
      logging.info('Inference result written to %s.', inference_result.uri)


def _MakeParseFn(
    payload_format: int
) -> Union[Callable[[bytes], tf.train.Example], Callable[
    [bytes], tf.train.SequenceExample]]:
  """Returns a function to parse bytes to payload."""
  if payload_format == example_gen_pb2.PayloadFormat.FORMAT_TF_EXAMPLE:
    return tf.train.Example.FromString
  elif (payload_format ==
        example_gen_pb2.PayloadFormat.FORMAT_TF_SEQUENCE_EXAMPLE):
    return tf.train.SequenceExample.FromString
  else:
    raise NotImplementedError(
        'Payload format %s is not supported.' %
        example_gen_pb2.PayloadFormat.Name(payload_format))


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
def _RunInference(
    pipeline: beam.Pipeline,
    payload_format: int,
    inference_endpoint: model_spec_pb2.InferenceSpecType
) -> beam.pvalue.PCollection:
  """Runs model inference on given examples data."""
  return (
      pipeline
      | 'ParseExamples' >> beam.Map(_MakeParseFn(payload_format))
      | 'RunInference' >> run_inference.RunInference(inference_endpoint))


@beam.ptransform_fn
@beam.typehints.with_input_types(prediction_log_pb2.PredictionLog)
@beam.typehints.with_output_types(beam.pvalue.PDone)
def _WriteExamples(prediction_log: beam.pvalue.PCollection,
                   output_example_spec: bulk_inferrer_pb2.OutputExampleSpec,
                   output_path: Text) -> beam.pvalue.PDone:
  """Converts `prediction_log` to `tf.train.Example` and materializes."""
  return (prediction_log
          | 'ConvertToExamples' >> beam.Map(
              prediction_to_example_utils.convert,
              output_example_spec=output_example_spec)
          | 'WriteExamples' >> beam.io.WriteToTFRecord(
              os.path.join(output_path, _EXAMPLES_FILE_NAME),
              file_name_suffix='.gz',
              coder=beam.coders.ProtoCoder(tf.train.Example)))
