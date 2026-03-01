# Copyright 2021 Google LLC. All Rights Reserved.
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
"""Utils using struct2tensor to parse ELWC."""

import itertools
from typing import Dict, List, Optional, Union

from struct2tensor import calculate
from struct2tensor import calculate_options
from struct2tensor import path
from struct2tensor import prensor_util
from struct2tensor.expression_impl import proto as proto_expr
import tensorflow as tf
from tfx_bsl.public import tfxio

from tensorflow_serving.apis import input_pb2


_DEFAULT_VALUE_SUFFIX = '_dv'
_SIZE_FEATURE_NAME = 'example_list_size'
_TYPE_LIST_MAP = {
    tf.float32: 'float_list',
    tf.int64: 'int64_list',
    tf.string: 'bytes_list',
}


# Parsing config for a feature in ELWCs.
class Feature:
  """Parsing config for a feature in ELWCs."""

  def __init__(self,
               name: str,
               dtype: tf.DType,
               default_value: Optional[Union[int, float, str]] = None,
               length: Optional[int] = None):
    """Initializer.

    Args:
      name: Name of the feature.
      dtype: Dtype of the feature. One of tf.string, tf.int64 or tf.float32.
        Note that if the actual ELWC does not contain the corresponding oneof
        (for example, if dtype is tf.string, but ELWC contains float_values),
        the feature will be parsed as an empty list
        (also see `default_value` below).
      default_value: Default value of the feature. If specified, must also
       specify `length`. For instances that do not have the specified length,
       it will be padded with the default value.
      length: The length of the feature. If specified, must also specify
        `default_value`.
    """
    self.name = name
    self.dtype = dtype
    self.default_value = default_value
    self.length = length

    assert dtype in _TYPE_LIST_MAP, (
        'Feature %s must have a dtype of tf.string, tf.int64 or tf.float32' %
        name)
    assert ((default_value is None and length is None) or
            (default_value is not None and length is not None)), (
                'Feature %s: default_value and length must both be specified '
                'or not specified' % name)
    if default_value is not None:
      assert ((dtype == tf.string and isinstance(default_value, bytes)) or
              (dtype == tf.int64 and isinstance(default_value, int)) or
              (dtype == tf.float32 and isinstance(default_value, float))), (
                  'Feature %s: type of default_value (%s) must match '
                  'dtype' % (name, type(default_value)))


class ELWCDecoder(tfxio.TFGraphRecordDecoder):
  """A TFGraphRecordDecoder that decodes ExampleListWithContext proto."""

  def __init__(self,
               name: str,
               context_features: List[Feature],
               example_features: List[Feature],
               size_feature_name: Optional[str] = None,
               label_feature: Optional[Feature] = None):
    self._context_features = context_features
    self._example_features = example_features
    self._size_feature_name = size_feature_name
    self._label_feature = label_feature

  def decode_record(self, records):
    example_features = list(self._example_features)
    if self._label_feature is not None:
      example_features.append(self._label_feature)
    # Our ELWC parser is based on struct2tensor
    # (https://github.com/google/struct2tensor).
    result = parse_elwc_with_struct2tensor(
        records, self._context_features,
        example_features,
        size_feature_name=self._size_feature_name)

    if self._label_feature is not None:
      # Cast label into float32.
      # TF-Ranking assumes the label is float32.
      result[self._label_feature.name] = tf.cast(
          result[self._label_feature.name], tf.float32)
      # TF-Ranking library assumes the label is of rank 2.
      # For example, the label ragged tensor of a batch of 2 ELWCs, with 2 and 1
      # examples respectively looks like:
      # [[[elwc1_e1], [elwc1_e2]], [[elwc2_e1]]]
      # However,
      # [[elwc1_e1, elwc1_e2], [elwc2_e1]]
      # is expected.
      # Because we know that each document has exactly one label (the innermost
      # dimension is of size 1), we could use tf.squeeze(..., axis=2), however
      # merge_dims() of a RaggedTensor is faster and equivalent in this case.
      result[self._label_feature.name] = result[
          self._label_feature.name].merge_dims(1, 2)
    return result


def create_keras_inputs(context_features,
                        example_features,
                        size_feature_name=None):
  """Create Keras input layers."""
  context_keras_inputs, example_keras_inputs = {}, {}
  # Create Keras inputs for context features.
  for feature in context_features:
    context_keras_inputs[feature.name] = tf.keras.Input(
        name=feature.name, shape=(None,), dtype=feature.dtype, ragged=True)

  for feature in example_features:
    example_keras_inputs[feature.name] = tf.keras.Input(
        name=feature.name, shape=(None, None), dtype=feature.dtype, ragged=True)

  if size_feature_name is not None:
    context_keras_inputs[size_feature_name] = tf.keras.Input(
        name=size_feature_name, shape=(None,), dtype=tf.int64, ragged=True)

  return context_keras_inputs, example_keras_inputs


def parse_elwc_with_struct2tensor(
    records: tf.Tensor,
    context_features: List[Feature],
    example_features: List[Feature],
    size_feature_name: Optional[str] = None) -> Dict[str, tf.RaggedTensor]:
  """Parses a batch of ELWC records into RaggedTensors using struct2tensor.

  Args:
    records: A dictionary with a single item. The value of this single item is
      the serialized ELWC input.
    context_features: List of context-level features.
    example_features: List of example-level features.
    size_feature_name: A string, the name of a feature for example list sizes.
      If None, which is default, this feature is not generated. Otherwise the
      feature is added to the feature dict.

  Returns:
    A dict that maps feature name to RaggedTensors.

  """

  def get_step_name(feature_name: str):
    """Gets the name of the step (a component in a prensor Path) for a feature.

    A prensor step cannot contain dots ("."), but a feature name can.

    Args:
      feature_name: name of the feature
    Returns:
      a valid step name.
    """
    return feature_name.replace('.', '_dot_')

  def get_default_filled_step_name(feature_name: str):
    return get_step_name(feature_name) + _DEFAULT_VALUE_SUFFIX

  def get_context_feature_path(feature: Feature):
    list_name = _TYPE_LIST_MAP.get(feature.dtype)
    return path.Path(['context', 'features', 'feature[{}]'.format(feature.name),
                      list_name, 'value'])

  def get_example_feature_path(feature: Feature):
    list_name = _TYPE_LIST_MAP.get(feature.dtype)
    return path.Path(['examples', 'features',
                      'feature[{}]'.format(feature.name), list_name, 'value'])

  def get_promote_and_project_maps(features: List[Feature], is_context: bool):
    promote_map = {}
    project_map = {}
    if is_context:
      get_feature_path = get_context_feature_path
      get_promote_destination = lambda leaf_name: path.Path([leaf_name])
    else:
      get_feature_path = get_example_feature_path
      get_promote_destination = lambda leaf_name: path.Path(  # pylint: disable=g-long-lambda
          ['examples', leaf_name])
    for feature in features:
      promote_map[get_step_name(feature.name)] = get_feature_path(feature)
      leaf_name = (get_step_name(feature.name) if feature.default_value is None
                   else get_default_filled_step_name(feature.name))
      project_map[feature.name] = get_promote_destination(leaf_name)
    return promote_map, project_map

  def get_pad_2d_ragged_fn(feature: Feature):
    def pad_2d_ragged(rt):
      dense = rt.to_tensor(shape=[None, feature.length],
                           default_value=feature.default_value)
      flattened = tf.reshape(dense, [-1])
      return tf.RaggedTensor.from_uniform_row_length(
          flattened, feature.length, validate=False)
    return pad_2d_ragged

  context_promote_map, context_keys_to_promoted_paths = (
      get_promote_and_project_maps(context_features, is_context=True))

  examples_promote_map, examples_keys_to_promoted_paths = (
      get_promote_and_project_maps(example_features, is_context=False))

  # Build the struct2tensor query.
  s2t_expr = (
      proto_expr.create_expression_from_proto(
          records, input_pb2.ExampleListWithContext.DESCRIPTOR)
      .promote_and_broadcast(context_promote_map, path.Path([]))
      .promote_and_broadcast(examples_promote_map, path.Path(['examples'])))
  # Pad features that have default_values specified.
  for features, parent_path in [(context_features, path.Path([])),
                                (example_features, path.Path(['examples']))]:
    for feature in features:
      if feature.default_value is not None:
        s2t_expr = s2t_expr.map_ragged_tensors(
            parent_path=parent_path,
            source_fields=[get_step_name(feature.name)],
            operator=get_pad_2d_ragged_fn(feature),
            is_repeated=True,
            dtype=feature.dtype,
            new_field_name=get_default_filled_step_name(feature.name))
  to_project = list(itertools.chain(
      context_keys_to_promoted_paths.values(),
      examples_keys_to_promoted_paths.values()))

  if size_feature_name is not None:
    s2t_expr = s2t_expr.create_size_field(
        path.Path(['examples']), get_step_name(size_feature_name))
    to_project.append(path.Path([get_step_name(size_feature_name)]))

  projection = s2t_expr.project(to_project)

  options = calculate_options.get_options_with_minimal_checks()
  prensor_result = calculate.calculate_prensors(
      [projection], options)[0]
  # a map from path.Path to RaggedTensors.
  projected_with_paths = prensor_util.get_ragged_tensors(
      prensor_result, options)

  context_dict = {
      f: projected_with_paths[context_keys_to_promoted_paths[f]]
      for f in context_keys_to_promoted_paths
  }

  examples_dict = {
      f: projected_with_paths[examples_keys_to_promoted_paths[f]]
      for f in examples_keys_to_promoted_paths
  }

  result = {}

  result.update(context_dict)
  result.update(examples_dict)

  if size_feature_name is not None:
    result[size_feature_name] = projected_with_paths[
        path.Path([get_step_name(size_feature_name)])]

  return result


def make_ragged_densify_layer():
  """Creates a keras layer that densifies a RaggedTensor.

  The layer takes a RaggedTensor as input and outputs a dense tensor.
  The dense tensor will have the same rank as the input RaggedTensor, and
  the size of each dimension will equal to the bounding size of the
  corresponding dimension of the input RaggedTensor. 0 will be used to fill
  the gaps.


  Returns:
    A Keras Layer.
  """
  return tf.keras.layers.Lambda(lambda x: x.to_tensor())
