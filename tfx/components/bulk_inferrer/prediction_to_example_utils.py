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
"""Utils for converting prediction_log to example."""

from typing import Any, List, Tuple, Union

import numpy as np
import tensorflow as tf

from tfx.proto import bulk_inferrer_pb2
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import prediction_log_pb2

_FeatureListType = List[Tuple[str, List[Union[str, bytes, float]]]]

# Typehint Any is for compatibility reason.
_OutputExampleSpecType = Union[bulk_inferrer_pb2.OutputExampleSpec, Any]
_PredictOutputType = Union[bulk_inferrer_pb2.PredictOutput, Any]
_ClassifyOutputType = Union[bulk_inferrer_pb2.ClassifyOutput, Any]


def convert(prediction_log: prediction_log_pb2.PredictionLog,
            output_example_spec: _OutputExampleSpecType) -> tf.train.Example:
  """Converts given `prediction_log` to a `tf.train.Example`.

  Args:
    prediction_log: The input prediction log.
    output_example_spec: The spec for how to map prediction results to columns
      in example.

  Returns:
    A `tf.train.Example` converted from the given prediction_log.
  Raises:
    ValueError: If the inference type or signature name in spec does not match
    that in prediction_log.
  """
  specs = output_example_spec.output_columns_spec
  if prediction_log.HasField('multi_inference_log'):
    example, output_features = _parse_multi_inference_log(
        prediction_log.multi_inference_log, output_example_spec)
  else:
    if len(specs) != 1:
      raise ValueError('Got single inference result, so expect single spec in '
                       'output_example_spec: %s' % output_example_spec)
    if prediction_log.HasField('regress_log'):
      if not specs[0].HasField('regress_output'):
        raise ValueError(
            'Regression predictions require a regress_output in output_example_spec: %s'
            % output_example_spec)
      example = tf.train.Example()
      example.CopyFrom(
          prediction_log.regress_log.request.input.example_list.examples[0])
      output_features = [
          (specs[0].regress_output.value_column,
           [prediction_log.regress_log.response.result.regressions[0].value])
      ]
    elif prediction_log.HasField('classify_log'):
      if not specs[0].HasField('classify_output'):
        raise ValueError(
            'Classification predictions require a classify_output in output_example_spec: %s'
            % output_example_spec)
      example, output_features = _parse_classify_log(
          prediction_log.classify_log, specs[0].classify_output)
    elif prediction_log.HasField('predict_log'):
      if not specs[0].HasField('predict_output'):
        raise ValueError(
            'Predict predictions require a predict_output in output_example_spec: %s'
            % output_example_spec)
      example, output_features = _parse_predict_log(prediction_log.predict_log,
                                                    specs[0].predict_output)
    else:
      raise ValueError('Unsupported prediction type in prediction_log: %s' %
                       prediction_log)

  return _add_columns(example, output_features)


def _parse_multi_inference_log(
    multi_inference_log: prediction_log_pb2.MultiInferenceLog,
    output_example_spec: _OutputExampleSpecType) -> tf.train.Example:
  """Parses MultiInferenceLog."""
  spec_map = {
      spec.signature_name or tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
      spec for spec in output_example_spec.output_columns_spec
  }
  example = tf.train.Example()
  example.CopyFrom(multi_inference_log.request.input.example_list.examples[0])
  output_features = []
  for result in multi_inference_log.response.results:
    spec = spec_map[result.model_spec.signature_name]
    if result.HasField('classification_result'):
      output_features += _parse_classification_result(
          result.classification_result, spec.classify_output)
    elif result.HasField('regression_result'):
      output_features.append((spec.regress_output.value_column,
                              [result.regression_result.regressions[0].value]))
    else:
      raise ValueError('Unsupported multi_inferrence_log: %s' %
                       multi_inference_log)
  return example, output_features


def _parse_classify_log(
    classify_log: prediction_log_pb2.ClassifyLog,
    classify_output_spec: _ClassifyOutputType
) -> Tuple[tf.train.Example, _FeatureListType]:
  """Parses ClassiyLog."""
  example = tf.train.Example()
  example.CopyFrom(classify_log.request.input.example_list.examples[0])
  return example, _parse_classification_result(classify_log.response.result,
                                               classify_output_spec)


def _parse_classification_result(
    classification_result: classification_pb2.ClassificationResult,
    classify_output_spec: _ClassifyOutputType) -> _FeatureListType:
  """Parses ClassificationResult."""
  output_features = []
  classes = classification_result.classifications[0].classes
  if classify_output_spec.label_column:
    output_features.append(
        (classify_output_spec.label_column, [c.label for c in classes]))
  if classify_output_spec.score_column:
    output_features.append(
        (classify_output_spec.score_column, [c.score for c in classes]))
  return output_features


def _parse_predict_log(
    predict_log: prediction_log_pb2.PredictLog,
    predict_output_spec: _PredictOutputType
) -> Tuple[tf.train.Example, _FeatureListType]:
  """Parses PredictLog."""
  _, input_tensor_proto = next(iter(predict_log.request.inputs.items()))
  example = tf.train.Example.FromString(input_tensor_proto.string_val[0])
  outputs = predict_log.response.outputs
  output_features = []
  for col in predict_output_spec.output_columns:
    output_tensor_proto = outputs.get(col.output_key)
    output_values = np.squeeze(tf.make_ndarray(output_tensor_proto))
    if output_values.ndim > 1:
      raise ValueError(
          'All output values must be convertible to 1D arrays, but %s was '
          'not. value was %s.' % (col.output_key, output_values))
    if output_values.ndim == 1:
      # Convert the output_values to a list.
      output_values = output_values.tolist()
    else:  # output_values.ndim == 0
      # Get a scalar for output_values.
      output_values = [output_values.item()]
    output_features.append((col.output_column, output_values))
  return example, output_features


def _add_columns(example: tf.train.Example,
                 features: _FeatureListType) -> tf.train.Example:
  """Add given features to `example`."""
  feature_map = example.features.feature
  for col, value in features:
    assert col not in feature_map, ('column name %s already exists in example: '
                                    '%s') % (col, example)
    # Note: we only consider two types, bytes and float for now.
    if isinstance(value[0], (str, bytes)):
      if isinstance(value[0], str):
        bytes_value = [v.encode('utf-8') for v in value]
      else:
        bytes_value = value
      feature_map[col].bytes_list.value[:] = bytes_value
    else:
      feature_map[col].float_list.value[:] = value
  return example
