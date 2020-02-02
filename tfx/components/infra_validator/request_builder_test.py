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
"""Tests for tfx.utils.request_builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import mock
import tensorflow as tf

from google.protobuf import json_format
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import regression_pb2
from tfx.components.infra_validator import request_builder
from tfx.proto import infra_validator_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class RequestBuilderTest(tf.test.TestCase):

  def setUp(self):
    super(RequestBuilderTest, self).setUp()
    self._examples = standard_artifacts.Examples()
    self._examples.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    self._examples.uri = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'testdata',
        'csv_example_gen'
    )

  def testInit_FailsIfMaxExamplesInvalid(self):
    with self.assertRaises(ValueError):
      request_builder.RequestBuilder(max_examples=-1, model_name='foo')

  def testReadFromExamplesArtifact_UptoMaxExamples(self):
    builder = request_builder.RequestBuilder(max_examples=1, model_name='foo')

    builder.ReadFromExamplesArtifact(self._examples)

    self.assertEqual(builder.num_examples, 1)

  def testReadFromExamplesArtifact_FailsIfNoSplitNames(self):
    examples = standard_artifacts.Examples()
    examples.split_names = artifact_utils.encode_split_names([])
    examples.uri = self._examples.uri
    builder = request_builder.RequestBuilder(max_examples=1, model_name='foo')

    with self.assertRaisesRegexp(ValueError, 'No split_name is available'):
      builder.ReadFromExamplesArtifact(examples)

  def testReadFromExamplesArtifact_FailsIfSplitNotExists(self):
    builder = request_builder.RequestBuilder(max_examples=1, model_name='foo')

    with self.assertRaisesRegexp(ValueError, 'No split_name asdf'):
      builder.ReadFromExamplesArtifact(self._examples, split_name='asdf')

  def testReadFromExamplesArtifact_FailsIfFileNotExists(self):
    invalid_uri_examples = standard_artifacts.Examples()
    invalid_uri_examples.split_names = self._examples.split_names
    invalid_uri_examples.uri = '/not/existing/path'
    builder = request_builder.RequestBuilder(max_examples=1, model_name='foo')

    with self.assertRaisesRegexp(ValueError, 'Unable to find examples'):
      builder.ReadFromExamplesArtifact(invalid_uri_examples)

  def testReadFromExamplesArtifact_NoOpIfExamplesFulfilled(self):
    builder = request_builder.RequestBuilder(max_examples=1, model_name='foo')
    builder.ReadFromExamplesArtifact(self._examples)

    with mock.patch.object(builder, '_ReadFromDataset') as read_impl:
      builder.ReadFromExamplesArtifact(self._examples)

    read_impl.assert_not_called()

  def testBuildClassificationRequest(self):
    builder = request_builder.RequestBuilder(max_examples=1, model_name='foo')
    builder.ReadFromExamplesArtifact(self._examples)

    requests = builder.BuildClassificationRequests()

    for request in requests:
      self.assertIsInstance(request, classification_pb2.ClassificationRequest)
      self.assertEqual(request.model_spec.name, 'foo')
      self.assertEqual(request.model_spec.signature_name, '')
      for example in request.input.example_list.examples:
        self.assertValidTaxiExample(example)

  def testBuildRegressionRequest(self):
    builder = request_builder.RequestBuilder(max_examples=1, model_name='foo')
    builder.ReadFromExamplesArtifact(self._examples)

    requests = builder.BuildRegressionRequests()

    for request in requests:
      self.assertIsInstance(request, regression_pb2.RegressionRequest)
      self.assertEqual(request.model_spec.name, 'foo')
      self.assertEqual(request.model_spec.signature_name, '')
      for example in request.input.example_list.examples:
        self.assertValidTaxiExample(example)

  def assertValidTaxiExample(self, tf_example: tf.train.Example):
    features = tf_example.features.feature
    self.assertIntFeature(features['trip_start_day'])
    self.assertIntFeature(features['pickup_community_area'])
    self.assertStringFeature(features['payment_type'])
    self.assertFloatFeature(features['trip_miles'])
    self.assertIntFeature(features['trip_start_timestamp'])
    self.assertFloatFeature(features['pickup_latitude'])
    self.assertFloatFeature(features['pickup_longitude'])
    self.assertIntFeature(features['trip_start_month'])
    self.assertIntFeature(features['trip_start_hour'])
    self.assertFloatFeature(features['trip_seconds'])

  def assertFloatFeature(self, feature: tf.train.Feature):
    self.assertEqual(len(feature.float_list.value), 1)

  def assertIntFeature(self, feature: tf.train.Feature):
    self.assertEqual(len(feature.int64_list.value), 1)

  def assertStringFeature(self, feature: tf.train.Feature):
    self.assertEqual(len(feature.bytes_list.value), 1)


def _create_request_spec(request_spec_dict):
  request_spec = infra_validator_pb2.RequestSpec()
  json_format.ParseDict(request_spec_dict, request_spec)
  return request_spec


class BuildRequestsTest(tf.test.TestCase):

  def setUp(self):
    super(BuildRequestsTest, self).setUp()
    patcher = mock.patch('tfx.components.infra_validator.request_builder.RequestBuilder')  # pylint: disable=line-too-long
    self.builder_cls = patcher.start()
    self.builder = self.builder_cls.return_value
    self.addCleanup(patcher.stop)

  def testTensorFlowServingClassify(self):
    # Prepare arguments.
    request_spec = _create_request_spec({
        'tensorflow_serving': {
            'rpc_kind': 'CLASSIFY'
        }
    })
    examples = mock.Mock()

    # Call build_requests.
    request_builder.build_requests(
        model_name='foo',
        examples=examples,
        request_spec=request_spec)

    # Check RequestBuilder calls.
    self.builder_cls.assert_called_with(model_name='foo', max_examples=1)
    self.builder.BuildClassificationRequests.assert_called()

  def testTensorFlowServingRegress(self):
    # Prepare arguments.
    request_spec = _create_request_spec({
        'tensorflow_serving': {
            'rpc_kind': 'REGRESS'
        }
    })
    examples = standard_artifacts.Examples()

    # Call build_requests.
    request_builder.build_requests(
        model_name='foo',
        examples=examples,
        request_spec=request_spec)

    # Check RequestBuilder calls.
    self.builder_cls.assert_called_with(model_name='foo', max_examples=1)
    self.builder.BuildRegressionRequests.assert_called()

  def testSplitNames(self):
    # Prepare arguments.
    request_spec = _create_request_spec({
        'tensorflow_serving': {
            'rpc_kind': 'CLASSIFY'
        },
        'split_name': 'train'
    })
    examples = standard_artifacts.Examples()

    # Call build_requests.
    request_builder.build_requests(
        model_name='foo',
        examples=examples,
        request_spec=request_spec)

    # Check RequestBuilder calls.
    self.builder.ReadFromExamplesArtifact.assert_called_with(
        examples, split_name='train')

  def testMaxExamples(self):
    # Prepare arguments.
    request_spec = _create_request_spec({
        'tensorflow_serving': {
            'rpc_kind': 'CLASSIFY'
        },
        'max_examples': 123
    })
    examples = standard_artifacts.Examples()

    # Call build_requests.
    request_builder.build_requests(
        model_name='foo',
        examples=examples,
        request_spec=request_spec)

    # Check RequestBuilder calls.
    self.builder_cls.assert_called_with(model_name='foo', max_examples=123)

  def testSignatureName(self):
    # Prepare arguments.
    request_spec = _create_request_spec({
        'tensorflow_serving': {
            'rpc_kind': 'CLASSIFY',
            'signature_name': 'my_signature_name'
        }
    })
    examples = standard_artifacts.Examples()

    # Call build_requests.
    request_builder.build_requests(
        model_name='foo',
        examples=examples,
        request_spec=request_spec)

    # Check RequestBuilder calls.
    self.builder.SetSignatureName.assert_called_with('my_signature_name')

  def testEmptyServingBinary(self):
    # Prepare empty request spec and examples.
    request_spec = _create_request_spec({})
    examples = standard_artifacts.Examples()

    with self.assertRaisesRegexp(ValueError, 'Invalid RequestSpec'):
      request_builder.build_requests(
          model_name='foo',
          examples=examples,
          request_spec=request_spec)

  def testInvalidTensorFlowServingRpcKind(self):
    # Prepare arguments.
    request_spec = _create_request_spec({
        'tensorflow_serving': {
            'rpc_kind': 'TF_SERVING_RPC_KIND_UNSPECIFIED'
        }
    })
    examples = standard_artifacts.Examples()

    with self.assertRaisesRegexp(ValueError,
                                 'Invalid TensorFlowServingRpcKind'):
      request_builder.build_requests(
          model_name='foo',
          examples=examples,
          request_spec=request_spec)


if __name__ == '__main__':
  tf.test.main()
