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

import os

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-import
from typing import List, Optional, Text

from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import regression_pb2
from tfx import types
from tfx.components.infra_validator import types as infra_validator_types
from tfx.proto import infra_validator_pb2
from tfx.types import artifact_utils

TensorFlowServingRpcKind = infra_validator_pb2.TensorFlowServingRpcKind
_TENSORFLOW_SERVING = 'tensorflow_serving'
_DEFAULT_MAX_EXAMPLES = 1


def build_requests(  # pylint: disable=invalid-name
    model_name: Text,
    examples: types.Artifact,
    request_spec: infra_validator_pb2.RequestSpec
) -> List[infra_validator_types.Request]:
  """Build a list of request protos to be queried against the model server.

  Args:
    model_name: Name of the model. For example, tensorflow `SavedModel` is saved
      under directory `{model_name}/{version}`. The same directory structure is
      reused in a tensorflow serving, and you need to specify `model_name` in
      the request to access it.
    examples: An `Examples` artifact which contains gzipped TFRecord file
      containing `tf.train.Example`.
    request_spec: A `RequestSpec` config.

  Returns:
    A list of request protos.
  """
  split_name = request_spec.split_name or None
  builder = RequestBuilder(
      max_examples=request_spec.max_examples or _DEFAULT_MAX_EXAMPLES,
      model_name=model_name
  )
  builder.ReadFromExamplesArtifact(examples, split_name=split_name)

  kind = request_spec.WhichOneof('serving_binary')
  if kind == _TENSORFLOW_SERVING:
    spec = request_spec.tensorflow_serving
    if spec.signature_name:
      builder.SetSignatureName(spec.signature_name)
    if spec.rpc_kind == TensorFlowServingRpcKind.CLASSIFY:
      return builder.BuildClassificationRequests()
    elif spec.rpc_kind == TensorFlowServingRpcKind.REGRESS:
      return builder.BuildRegressionRequests()
    else:
      raise ValueError('Invalid TensorFlowServingRpcKind {}'.format(
          spec.rpc_kind))
  else:
    raise ValueError('Invalid RequestSpec {}'.format(request_spec))


class RequestBuilder(object):
  """`RequestBuilder` reads `Example`s and builds requests for model server.

  You must read `Example`s using `ReadFromXXX()` method before building
  requests.

  Usage:

  ```python
  builder = RequestBuilder(max_examples=10, model_name='mnist')
  builder.ReadFromExamplesArtifact(examples, split_name='eval')
  requests = builder.BuildClassificationRequests()
  ```
  """

  def __init__(self, max_examples: int, model_name: Text):
    if max_examples <= 0:
      raise ValueError('max_examples should be > 0.')
    self._max_examples = max_examples
    self._model_name = model_name
    self._signature_name = ''
    self._examples = []  # type: List[tf.train.Example]

  @property
  def num_examples(self) -> int:
    return len(self._examples)

  def SetSignatureName(self, signature_name: Text):
    self._signature_name = signature_name

  # TODO(jjong): The method strongly assumes that the output of ExampleGen is
  # a gzipped TFRecords of tf.Example. We need a better abstraction (e.g. TFXIO)
  # to accept arbitrary file format and convert it to appropriate request types.
  def ReadFromExamplesArtifact(self, examples: types.Artifact,
                               split_name: Optional[Text] = None):
    """Read up to `self._max_examples` `tf.Example`s from `Examples` artifact.

    Args:
      examples: `Examples` artifact.
      split_name: Name of the split to read from given `example`.
    """
    available_splits = artifact_utils.decode_split_names(examples.split_names)
    if not available_splits:
      raise ValueError('No split_name is available in given Examples artifact.')
    if split_name is None:
      split_name = available_splits[0]
    if split_name not in available_splits:
      raise ValueError('No split_name {}; available split names: {}'.format(
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

    # If max_examples are fulfilled, no-op.
    if len(self._examples) >= self._max_examples:
      return

    self._ReadFromDataset(
        tf.data.TFRecordDataset(filenames, compression_type='GZIP'))

  def _ReadFromDataset(self, dataset):
    """Read up to `self._max_examples` `Example`s from `tf.data.Dataset`.

    Args:
      dataset: `tf.data.Dataset` instance.
    """
    num_examples = self._max_examples - len(self._examples)
    dataset = dataset.take(num_examples)
    if tf.executing_eagerly():
      for example in dataset:
        self._examples.append(tf.train.Example.FromString(example.numpy()))
    else:
      it = tf.compat.v1.data.make_one_shot_iterator(dataset)
      next_el = it.get_next()
      with tf.Session() as sess:
        while True:
          try:
            example_bytes = sess.run(next_el)
            self._examples.append(tf.train.Example.FromString(example_bytes))
          except tf.errors.OutOfRangeError:
            break

  def BuildClassificationRequests(
      self) -> List[classification_pb2.ClassificationRequest]:
    """Build `ClassificationRequest`s from read `Example`s.

    Returns:
      A list of `ClassificationRequest` instance.

    Raises:
      RuntimeError: if you haven't read examples in advance.
    """
    if not self._examples:
      raise RuntimeError('Read examples before building requests')
    return [self._ExampleToClassificationRequest(example)
            for example in self._examples]

  def BuildRegressionRequests(self) -> List[regression_pb2.RegressionRequest]:
    """Build `RegressionRequest`s from read `Example`s.

    Returns:
      A list of `RegressionRequest` instance.

    Raises:
      RuntimeError: if you haven't read examples in advance.
    """
    if not self._examples:
      raise RuntimeError('Read examples before building requests')
    return [self._ExampleToRegressionRequest(example)
            for example in self._examples]

  def _ExampleToClassificationRequest(
      self,
      example: tf.train.Example
  ) -> classification_pb2.ClassificationRequest:
    """Convert single Example to ClassificationRequest.

    Args:
      example: `Example` instance to convert.

    Returns:
      A converted `ClassificationRequest` instance.
    """
    request = classification_pb2.ClassificationRequest()
    request.model_spec.name = self._model_name
    request.model_spec.signature_name = self._signature_name
    request.input.example_list.examples.append(example)
    return request

  def _ExampleToRegressionRequest(
      self,
      example: tf.train.Example
  ) -> regression_pb2.RegressionRequest:
    """Convert single Example to RegressionRequest.

    Args:
      example: `Example` instance to convert.

    Returns:
      A converted `RegressionRequest` instance.
    """
    request = regression_pb2.RegressionRequest()
    request.model_spec.name = self._model_name
    request.model_spec.signature_name = self._signature_name
    request.input.example_list.examples.append(example)
    return request
