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
"""Utility functions for building requests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os

from absl import logging
import enum
import six
import tensorflow as tf
from typing import Any, Iterable, List, Mapping, Optional, Text

# TODO(b/140306674): Stop using the internal TF API
from tensorflow.python.saved_model import loader_impl  # pylint: disable=g-direct-tensorflow-import
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import regression_pb2
from tfx import types
from tfx.components.infra_validator import types as iv_types
from tfx.proto import infra_validator_pb2
from tfx.types import artifact_utils
from tfx.utils import path_utils

_TENSORFLOW_SERVING = 'tensorflow_serving'
_DEFAULT_NUM_EXAMPLES = 1

_DEFAULT_TAG_SET = frozenset([tf.saved_model.SERVING])

# We define the following aliases of Any because the actual types are not
# public.
_SavedModel = Any
_SignatureDef = Any


def build_requests(  # pylint: disable=invalid-name
    model_name: Text,
    model: types.Artifact,
    examples: types.Artifact,
    request_spec: infra_validator_pb2.RequestSpec
) -> List[iv_types.Request]:
  """Build model server requests.

  Examples artifact will be used as a data source to build requests. Caller
  should guarantee that the logical format of the Examples artifact should be
  compatible with request type to build.

  Args:
    model_name: A model name that model server recognizes.
    model: A model artifact for model signature analysis.
    examples: An `Examples` artifact for request data source.
    request_spec: A `RequestSpec` config.

  Returns:
    A list of request protos.
  """
  split_name = request_spec.split_name or None
  num_examples = request_spec.num_examples or _DEFAULT_NUM_EXAMPLES

  kind = request_spec.WhichOneof('kind')
  if kind == _TENSORFLOW_SERVING:
    spec = request_spec.tensorflow_serving
    signatures = _parse_saved_model_signatures(
        model_path=path_utils.serving_model_path(model.uri),
        tag_set=spec.tag_set,
        signature_names=spec.signature_names)
    builder = _TFServingRpcRequestBuilder(
        model_name=model_name,
        signatures=signatures)
  else:
    raise NotImplementedError('Unsupported RequestSpec kind {!r}'.format(kind))

  builder.ReadExamplesArtifact(
      examples,
      split_name=split_name,
      num_examples=num_examples)

  return builder.BuildRequests()


# TODO(b/151790176): Move to tfx_bsl, or keep it if TF adds a proper public API.
def _parse_saved_model_signatures(
    model_path: Text,
    tag_set: Iterable[Text],
    signature_names: Iterable[Text]) -> Mapping[Text, _SignatureDef]:
  """Parse SignatureDefs of given signature names from SavedModel.

  Among one or more MetaGraphDefs in SavedModel, the first one that has all the
  tag_set elements is chosen. Selected MetaGraphDef should have signatures for
  all given signature names.

  Args:
    model_path: A path to the SavedModel directory.
    tag_set: A set of tags MetaGraphDef should have.
    signature_names: A list of signature names to retrieve.

  Returns:
    A mapping from signature name to SignatureDef.
  """
  if not tag_set:
    tag_set = {tf.saved_model.SERVING}
    logging.info('tag_set is not given. Using %r instead.', tag_set)
  if not signature_names:
    signature_names = [tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    logging.info('signature_names are not given. Using %r instead.',
                 signature_names)
  loader = loader_impl.SavedModelLoader(model_path)
  meta_graph_def = loader.get_meta_graph_def_from_tags(tag_set)
  result = {}
  for signature_name in signature_names:
    if signature_name not in meta_graph_def.signature_def:
      raise ValueError('SignatureDef of name {} could not be found in '
                       'MetaGraphDef'.format(signature_name))
    result[signature_name] = meta_graph_def.signature_def[signature_name]
  return result


class _LogicalFormat(enum.Enum):
  UNKNOWN = 0
  TF_EXAMPLE = 1


class _BaseRequestBuilder(six.with_metaclass(abc.ABCMeta, object)):
  """Base class for all RequestBuilders."""

  def __init__(self):
    self._records = []  # type: List[bytes]
    self._record_format = _LogicalFormat.UNKNOWN

  # TODO(jjong): The method strongly assumes that the output of ExampleGen is
  # a gzipped TFRecords of tf.Example. We need a better abstraction (e.g. TFXIO)
  # to accept arbitrary file format and convert it to appropriate request types.
  def ReadExamplesArtifact(self, examples: types.Artifact, num_examples: int,
                           split_name: Optional[Text] = None):
    """Read records from Examples artifact.

    Currently it assumes Examples artifact contains serialized tf.Example in
    gzipped TFRecord files.

    Args:
      examples: `Examples` artifact.
      num_examples: Number of examples to read. If the specified value is larger
          than the actual number of examples, all examples would be read.
      split_name: Name of the split to read from the Examples artifact.

    Raises:
      RuntimeError: If read twice.
    """
    if self._records:
      raise RuntimeError('Cannot read records twice.')

    if num_examples < 1:
      raise ValueError('num_examples < 1 (got {})'.format(num_examples))

    available_splits = artifact_utils.decode_split_names(examples.split_names)
    if not available_splits:
      raise ValueError('No split_name is available in given Examples artifact.')
    if split_name is None:
      split_name = available_splits[0]
    if split_name not in available_splits:
      raise ValueError(
          'No split_name {}; available split names: {}'.format(
              split_name, ', '.join(available_splits)))

    # ExampleGen generates artifacts under each split_name directory.
    glob_pattern = os.path.join(examples.uri, split_name, '*.gz')
    try:
      filenames = tf.io.gfile.glob(glob_pattern)
    except tf.errors.NotFoundError:
      filenames = []
    if not filenames:
      raise ValueError('Unable to find examples matching {}.'.format(
          glob_pattern))

    # Assume we have a tf.Example logical format.
    self._record_format = _LogicalFormat.TF_EXAMPLE

    self._ReadFromDataset(
        tf.data.TFRecordDataset(filenames, compression_type='GZIP'),
        num_examples=num_examples)

  def _ReadFromDataset(self, dataset: tf.data.TFRecordDataset,
                       num_examples: int):
    dataset = dataset.take(num_examples)
    if tf.executing_eagerly():
      for raw_example in dataset:
        self._records.append(raw_example.numpy())
    else:
      it = tf.compat.v1.data.make_one_shot_iterator(dataset)
      next_el = it.get_next()
      with tf.Session() as sess:
        while True:
          try:
            raw_example = sess.run(next_el)
            self._records.append(raw_example)
          except tf.errors.OutOfRangeError:
            break

  @abc.abstractmethod
  def BuildRequests(self) -> List[iv_types.Request]:
    """Transform read records (bytes) to the request type."""


class _TFServingRpcRequestBuilder(_BaseRequestBuilder):
  """RequestBuilder for TF Serving RPC requests.

  There are three kinds of request the builder can make:
  -   ClassificationRequest
  -   RegressionRequest
  -   PredictRequest

  Types of request to build is determined by inspecting SavedModel and getting
  SignatureDef from it. What user can configure is the signature names to use.

  To build a ClassificationRequest or a RegressionRequest, logical format of
  the record should be TF_EXAMPLE.

  To build a PredictRequest, its corresponding SignatureDef should have a single
  input argument that accepts serialized record inputs. Its logical format does
  not matter as long as user have a correct parsing logic.
  """

  def __init__(self,
               model_name: Text,
               signatures: Mapping[Text, _SignatureDef]):
    super(_TFServingRpcRequestBuilder, self).__init__()
    self._model_name = model_name
    self._signatures = signatures
    self._examples = []

  @property
  def examples(self) -> List[tf.train.Example]:
    if not self._examples:
      if self._record_format != _LogicalFormat.TF_EXAMPLE:
        raise ValueError('Record format should be TF_EXAMPLE.')
      for record in self._records:
        example = tf.train.Example()
        example.ParseFromString(record)
        self._examples.append(example)
    return self._examples

  def BuildRequests(self) -> List[iv_types.TensorFlowServingRequest]:
    assert self._records, 'Records are empty.'
    result = []
    for signature_name, signature_def in self._signatures.items():
      if signature_def.method_name == tf.saved_model.PREDICT_METHOD_NAME:
        result.extend(
            self._BuildPredictRequests(
                signature_name, self._GetSerializedInputKey(signature_def)))
      elif signature_def.method_name == tf.saved_model.CLASSIFY_METHOD_NAME:
        result.extend(self._BuildClassificationRequests(signature_name))
      elif signature_def.method_name == tf.saved_model.REGRESS_METHOD_NAME:
        result.extend(self._BuildRegressionRequests(signature_name))
      else:
        raise ValueError('Unknown method name {}'.format(
            signature_def.method_name))
    return result

  def _GetSerializedInputKey(self, signature_def: _SignatureDef):
    """Gets key for SignatureDef input that consumes serialized record.

    To build a PredictRequest, SignatureDef inputs should have a single input
    argument that accepts serialized record inputs. The input TensorSpec should
    have dtype=DT_STRING and shape=TensorShape([None]).

    Args:
      signature_def: A SignatureDef proto message.

    Returns:
      An input key for the serialized input.
    """
    signature_input_keys = list(signature_def.inputs.keys())
    if len(signature_input_keys) == 1:
      input_key = signature_input_keys[0]
      input_spec = signature_def.inputs[input_key]
      if (input_spec.dtype == tf.dtypes.string.as_datatype_enum
          and input_spec.tensor_shape == tf.TensorShape([None]).as_proto()):
        return input_key
    # TODO(b/151697719): General Predict method signature support.
    raise ValueError(
        'Unable to find valid input key from SignatureDef. In order to make '
        'PredictRequest, model should define signature that accepts serialized '
        'record inputs, i.e. signature with single input whose dtype=DT_STRING '
        'and shape=TensorShape([None]).')

  def _BuildClassificationRequests(self, signature_name: Text):
    for example in self.examples:
      request = classification_pb2.ClassificationRequest()
      request.model_spec.name = self._model_name
      request.model_spec.signature_name = signature_name
      request.input.example_list.examples.append(example)
      yield request

  def _BuildRegressionRequests(self, signature_name: Text):
    for example in self.examples:
      request = regression_pb2.RegressionRequest()
      request.model_spec.name = self._model_name
      request.model_spec.signature_name = signature_name
      request.input.example_list.examples.append(example)
      yield request

  def _BuildPredictRequests(self, signature_name: Text,
                            serialized_input_key: Text):
    for record in self._records:
      request = predict_pb2.PredictRequest()
      request.model_spec.name = self._model_name
      request.model_spec.signature_name = signature_name
      request.inputs[serialized_input_key].CopyFrom(
          tf.make_tensor_proto([record]))
      yield request
