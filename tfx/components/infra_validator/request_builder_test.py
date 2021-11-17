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

import os
from typing import Any, Dict
import unittest
from unittest import mock

import tensorflow as tf
from tfx.components.infra_validator import request_builder
from tfx.proto import infra_validator_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import path_utils

from google.protobuf import json_format
# TODO(b/140306674): Stop using the internal TF API
from tensorflow.core.protobuf import meta_graph_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.core.protobuf import saved_model_pb2  # pylint: disable=g-direct-tensorflow-import
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import regression_pb2


_TEST_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'testdata')
_CSV_EXAMPLE_GEN_URI = os.path.join(_TEST_DATA_ROOT, 'csv_example_gen')
_ESTIMATOR_MODEL_URI = os.path.join(_TEST_DATA_ROOT, 'trainer', 'current')
_KERAS_MODEL_URI = os.path.join(_TEST_DATA_ROOT, 'trainer', 'keras')


def _make_saved_model(payload: Dict[str, Any]):
  result = saved_model_pb2.SavedModel()
  json_format.ParseDict(payload, result)
  return result


def _make_signature_def(payload: Dict[str, Any]):
  result = meta_graph_pb2.SignatureDef()
  json_format.ParseDict(payload, result)
  return result


def _make_request_spec(payload: Dict[str, Any]):
  result = infra_validator_pb2.RequestSpec()
  json_format.ParseDict(payload, result)
  return result


class TestParseSavedModelSignature(tf.test.TestCase):

  def _MockSavedModel(self, saved_model_dict):
    saved_model_proto = _make_saved_model(saved_model_dict)
    saved_model_path = os.path.join(self.get_temp_dir(), 'saved_model.pb')
    with open(saved_model_path, 'wb') as f:
      f.write(saved_model_proto.SerializeToString())
    return os.path.dirname(saved_model_path)

  def testParseSavedModelSignature(self):
    model_path = self._MockSavedModel({
        'meta_graphs': [
            {
                'meta_info_def': {
                    'tags': ['serve']
                },
                'signature_def': {
                    'foo': {
                        'method_name': 'tensorflow/serving/predict',
                        'inputs': {
                            'x': {
                                'name': 'serving_default_input:0',
                                'dtype': 'DT_FLOAT',
                                'tensor_shape': {
                                    'dim': [
                                        {'size': -1},
                                        {'size': 784},
                                    ]
                                }
                            }
                        },
                        'outputs': {
                            'y': {
                                'name': 'StatefulPartitionedCall:0',
                                'dtype': 'DT_FLOAT',
                                'tensor_shape': {
                                    'dim': [
                                        {'size': -1},
                                        {'size': 10},
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        ]
    })

    signatures = request_builder._parse_saved_model_signatures(
        model_path, tag_set={'serve'}, signature_names=['foo'])

    self.assertEqual(signatures['foo'].inputs['x'].dtype,
                     tf.dtypes.float32.as_datatype_enum)
    self.assertEqual(signatures['foo'].inputs['x'].tensor_shape,
                     tf.TensorShape([None, 784]).as_proto())
    self.assertEqual(signatures['foo'].outputs['y'].dtype,
                     tf.dtypes.float32.as_datatype_enum)
    self.assertEqual(signatures['foo'].outputs['y'].tensor_shape,
                     tf.TensorShape([None, 10]).as_proto())

  def testParseSavedModelSignature_FailIfNoMetaGraph(self):
    model_path = self._MockSavedModel({
        'meta_graphs': []
    })

    with self.assertRaisesRegex(
        RuntimeError,
        'MetaGraphDef associated with tags .* could not be found'):
      request_builder._parse_saved_model_signatures(
          model_path, tag_set={'serve'}, signature_names=['foo'])

  def testParseSavedModelSignature_FailIfTagSetNotMatch(self):
    model_path = self._MockSavedModel({
        'meta_graphs': [
            {
                'meta_info_def': {
                    'tags': ['a', 'b']
                }
            }
        ]
    })

    with self.assertRaisesRegex(
        RuntimeError,
        'MetaGraphDef associated with tags .* could not be found'):
      request_builder._parse_saved_model_signatures(
          model_path, tag_set={'a', 'c'}, signature_names=['foo'])

  def testParseSavedModelSignature_FailIfSignatureNotFound(self):
    model_path = self._MockSavedModel({
        'meta_graphs': [
            {
                'meta_info_def': {
                    'tags': ['serve']
                },
                'signature_def': {
                    'foo': {}
                }
            }
        ]
    })

    with self.assertRaisesRegex(
        ValueError, 'SignatureDef of name bar could not be found'):
      request_builder._parse_saved_model_signatures(
          model_path, tag_set={'serve'}, signature_names=['foo', 'bar'])

  def testParseSavedModelSignature_DefaultTagSet(self):
    model_path = self._MockSavedModel({
        'meta_graphs': [
            {
                'meta_info_def': {
                    'tags': ['serve']
                },
                'signature_def': {
                    'foo': {}
                }
            }
        ]
    })

    signatures = request_builder._parse_saved_model_signatures(
        model_path, tag_set=set(), signature_names=['foo'])

    self.assertTrue(signatures)

  def testParseSavedModelSignature_DefaultSignatureName(self):
    model_path = self._MockSavedModel({
        'meta_graphs': [
            {
                'meta_info_def': {
                    'tags': ['foo']
                },
                'signature_def': {
                    'serving_default': {},
                }
            }
        ]
    })

    signatures = request_builder._parse_saved_model_signatures(
        model_path, tag_set={'foo'}, signature_names=[])

    self.assertTrue(signatures)


class _MockBuilder(request_builder._BaseRequestBuilder):

  def BuildRequests(self):
    raise NotImplementedError()


class BaseRequestBuilderTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._examples = standard_artifacts.Examples()
    self._examples.uri = _CSV_EXAMPLE_GEN_URI
    self._examples.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])

  def testReadExamplesArtifact(self):
    builder = _MockBuilder()

    builder.ReadExamplesArtifact(self._examples, num_examples=1)

    self.assertEqual(len(builder._records), 1)
    self.assertIsInstance(builder._records[0], bytes)

  def testReadExamplesArtifact_FailIfSplitNamesEmpty(self):
    builder = _MockBuilder()
    examples = standard_artifacts.Examples()
    examples.uri = self._examples.uri

    with self.assertRaises(ValueError):
      builder.ReadExamplesArtifact(examples, num_examples=1)

  def testReadExamplesArtifact_FailIfSplitNameInvalid(self):
    builder = _MockBuilder()

    with self.assertRaises(ValueError):
      builder.ReadExamplesArtifact(self._examples, num_examples=1,
                                   split_name='non-existing-split')

  def testReadExamplesArtifact_FailReadTwice(self):
    builder = _MockBuilder()

    builder.ReadExamplesArtifact(self._examples, num_examples=1)
    with self.assertRaises(RuntimeError):
      builder.ReadExamplesArtifact(self._examples, num_examples=1)


class TFServingRpcRequestBuilderTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._examples = standard_artifacts.Examples()
    self._examples.uri = _CSV_EXAMPLE_GEN_URI
    self._examples.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])

  def _GetEstimatorModelSignature(self, signature_names=()):
    model_path = path_utils.serving_model_path(_ESTIMATOR_MODEL_URI)
    return request_builder._parse_saved_model_signatures(
        model_path, tag_set={'serve'}, signature_names=signature_names)

  def _GetKerasModelSignature(self):
    model_path = path_utils.serving_model_path(_KERAS_MODEL_URI)
    return request_builder._parse_saved_model_signatures(
        model_path, tag_set={'serve'}, signature_names=['serving_default'])

  @unittest.skipIf(
      tf.__version__ < '2',
      'The test uses testdata only compatible with TF2.')
  def testBuildRequests_EstimatorModel_ServingDefault(self):
    builder = request_builder._TFServingRpcRequestBuilder(
        model_name='foo',
        signatures=self._GetEstimatorModelSignature())
    builder.ReadExamplesArtifact(self._examples, num_examples=1)

    result = builder.BuildRequests()

    self.assertEqual(len(result), 1)
    self.assertIsInstance(result[0], classification_pb2.ClassificationRequest)
    self.assertEqual(result[0].model_spec.name, 'foo')
    self.assertEqual(result[0].model_spec.signature_name, 'serving_default')

  @unittest.skipIf(
      tf.__version__ < '2',
      'The test uses testdata only compatible with TF2.')
  def testBuildRequests_EstimatorModel_Classification(self):
    builder = request_builder._TFServingRpcRequestBuilder(
        model_name='foo',
        signatures=self._GetEstimatorModelSignature(
            signature_names=['classification']))
    builder.ReadExamplesArtifact(self._examples, num_examples=1)

    result = builder.BuildRequests()

    self.assertEqual(len(result), 1)
    self.assertIsInstance(result[0], classification_pb2.ClassificationRequest)
    self.assertEqual(result[0].model_spec.name, 'foo')
    self.assertEqual(result[0].model_spec.signature_name, 'classification')

  @unittest.skipIf(
      tf.__version__ < '2',
      'The test uses testdata only compatible with TF2.')
  def testBuildRequests_EstimatorModel_Regression(self):
    builder = request_builder._TFServingRpcRequestBuilder(
        model_name='foo',
        signatures=self._GetEstimatorModelSignature(
            signature_names=['regression']))
    builder.ReadExamplesArtifact(self._examples, num_examples=1)

    result = builder.BuildRequests()

    self.assertEqual(len(result), 1)
    self.assertIsInstance(result[0], regression_pb2.RegressionRequest)
    self.assertEqual(result[0].model_spec.name, 'foo')
    self.assertEqual(result[0].model_spec.signature_name, 'regression')

  @unittest.skipIf(
      tf.__version__ < '2',
      'The test uses testdata only compatible with TF2.')
  def testBuildRequests_EstimatorModel_Predict(self):
    builder = request_builder._TFServingRpcRequestBuilder(
        model_name='foo',
        signatures=self._GetEstimatorModelSignature(
            signature_names=['predict']))
    builder.ReadExamplesArtifact(self._examples, num_examples=1)

    result = builder.BuildRequests()

    self.assertEqual(len(result), 1)
    self.assertIsInstance(result[0], predict_pb2.PredictRequest)
    self.assertEqual(result[0].model_spec.name, 'foo')
    self.assertEqual(result[0].model_spec.signature_name, 'predict')
    self.assertEqual(len(result[0].inputs), 1)
    input_key = list(result[0].inputs.keys())[0]
    self.assertEqual(result[0].inputs[input_key].dtype,
                     tf.dtypes.string.as_datatype_enum)

  @unittest.skipIf(
      tf.__version__ < '2',
      'The test uses testdata only compatible with TF2.')
  def testBuildRequests_KerasModel(self):
    builder = request_builder._TFServingRpcRequestBuilder(
        model_name='foo',
        signatures=self._GetKerasModelSignature())
    builder.ReadExamplesArtifact(self._examples, num_examples=1)

    result = builder.BuildRequests()

    self.assertEqual(len(result), 1)
    self.assertIsInstance(result[0], predict_pb2.PredictRequest)
    self.assertEqual(result[0].model_spec.name, 'foo')
    self.assertEqual(result[0].model_spec.signature_name, 'serving_default')

  def testBuildRequests_PredictMethod(self):
    builder = request_builder._TFServingRpcRequestBuilder(
        model_name='foo',
        signatures={
            # Has only one argument with dtype=DT_STRING and shape=(None,).
            # This is the only valid form that InfraValidator accepts today.
            'serving_default': _make_signature_def({
                'method_name': 'tensorflow/serving/predict',
                'inputs': {
                    'x': {
                        'name': 'serving_default_examples:0',
                        'dtype': 'DT_STRING',
                        'tensor_shape': {
                            'dim': [
                                {'size': -1},
                            ]
                        }
                    }
                },
                'outputs': {
                    'y': {
                        'name': 'StatefulPartitionedCall:0',
                        'dtype': 'DT_FLOAT',
                        'tensor_shape': {
                            'dim': [
                                {'size': -1},
                                {'size': 10},
                            ]
                        }
                    }
                },
            })
        })
    builder.ReadExamplesArtifact(self._examples, num_examples=1)

    result = builder.BuildRequests()

    self.assertEqual(len(result), 1)
    self.assertIsInstance(result[0], predict_pb2.PredictRequest)
    self.assertEqual(result[0].inputs['x'].dtype,
                     tf.dtypes.string.as_datatype_enum)

  def testBuildRequests_PredictMethod_FailOnInvalidSignature(self):
    builder = request_builder._TFServingRpcRequestBuilder(
        model_name='foo',
        signatures={
            # Signature argument is not for serialized tf.Example (i.e. dtype !=
            # DT_STRING or shape != (None,)).
            'serving_default': _make_signature_def({
                'method_name': 'tensorflow/serving/predict',
                'inputs': {
                    'x': {
                        'name': 'serving_default_input:0',
                        'dtype': 'DT_FLOAT',
                        'tensor_shape': {
                            'dim': [
                                {'size': -1},
                                {'size': 784},
                            ]
                        }
                    }
                },
                'outputs': {
                    'y': {
                        'name': 'StatefulPartitionedCall:0',
                        'dtype': 'DT_FLOAT',
                        'tensor_shape': {
                            'dim': [
                                {'size': -1},
                                {'size': 10},
                            ]
                        }
                    }
                },
            })
        })
    builder.ReadExamplesArtifact(self._examples, num_examples=1)

    with self.assertRaisesRegex(
        ValueError, 'Unable to find valid input key from SignatureDef'):
      builder.BuildRequests()


class TestBuildRequests(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._model_name = 'foo'
    self._examples = standard_artifacts.Examples()
    self._examples.uri = _CSV_EXAMPLE_GEN_URI
    self._examples.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    self._model = standard_artifacts.Model()
    self._model.uri = _ESTIMATOR_MODEL_URI

  def _PrepareTFServingRequestBuilder(self):
    patcher = mock.patch.object(
        request_builder, '_TFServingRpcRequestBuilder',
        wraps=request_builder._TFServingRpcRequestBuilder)
    builder_cls = patcher.start()
    self.addCleanup(patcher.stop)
    return builder_cls

  def testBuildRequests_TFServing(self):
    builder_cls = self._PrepareTFServingRequestBuilder()
    builder = builder_cls.return_value

    request_builder.build_requests(
        model_name='foo',
        model=self._model,
        examples=self._examples,
        request_spec=_make_request_spec({
            'tensorflow_serving': {
                'signature_names': ['serving_default']
            },
            'split_name': 'eval',
            'num_examples': 1
        })
    )

    builder_cls.assert_called_with(
        model_name='foo',
        signatures={'serving_default': mock.ANY})
    builder.ReadExamplesArtifact.assert_called_with(
        self._examples,
        split_name='eval',
        num_examples=1)
    builder.BuildRequests.assert_called()

  def testBuildRequests_NumberOfRequests(self):
    result = request_builder.build_requests(
        model_name='foo',
        model=self._model,
        examples=self._examples,
        request_spec=_make_request_spec({
            'tensorflow_serving': {
                'signature_names': ['classification', 'regression']
            },
            'split_name': 'eval',
            'num_examples': 3
        })
    )

    # Total 6 requests (3 requests for each signature)
    self.assertEqual(len(result), 6)
    self.assertEqual(
        len([r for r in result
             if r.model_spec.signature_name == 'classification']), 3)
    self.assertEqual(
        len([r for r in result
             if r.model_spec.signature_name == 'regression']), 3)

  def testBuildRequests_DefaultArgument(self):
    builder_cls = self._PrepareTFServingRequestBuilder()
    builder = builder_cls.return_value

    request_builder.build_requests(
        model_name='foo',
        model=self._model,
        examples=self._examples,
        request_spec=_make_request_spec({
            'tensorflow_serving': {
                # 'signature_names': ['serving_default']
            },
            # 'split_name': 'eval',
            # 'num_examples': 1
        })
    )

    builder.ReadExamplesArtifact.assert_called_with(
        self._examples,
        split_name=None,  # Without split_name (will choose any split).
        num_examples=1)   # Default num_examples = 1.


if __name__ == '__main__':
  tf.test.main()
