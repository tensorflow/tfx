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
"""Tests for tfx.orchestration.portable.runtime_parameter_utils."""
import os

import tensorflow as tf

from tfx.orchestration.portable import runtime_parameter_utils
from tfx.proto.orchestration import pipeline_pb2
from tfx.utils import test_case_utils

from ml_metadata.proto import metadata_store_pb2


class RuntimeParameterUtilsTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    self._connection_config = metadata_store_pb2.ConnectionConfig()
    self._connection_config.sqlite.SetInParent()
    self._testdata_dir = os.path.join(os.path.dirname(__file__), 'testdata')

  def testFullySubstituteRuntimeParameter(self):
    pipeline = pipeline_pb2.Pipeline()
    expected = pipeline_pb2.Pipeline()
    self.load_proto_from_text(
        os.path.join(self._testdata_dir,
                     'pipeline_with_runtime_parameter.pbtxt'), pipeline)
    self.load_proto_from_text(
        os.path.join(self._testdata_dir,
                     'pipeline_with_runtime_parameter_substituted.pbtxt'),
        expected)
    parameters = runtime_parameter_utils.substitute_runtime_parameter(
        pipeline, {
            'context_name_rp': 'my_context',
            'prop_one_rp': 2,
            'prop_two_rp': 'X'
        })
    self.assertProtoEquals(pipeline, expected)
    self.assertEqual(len(parameters), 3)
    self.assertEqual(parameters['context_name_rp'], 'my_context')
    self.assertEqual(parameters['prop_one_rp'], 2)
    self.assertEqual(parameters['prop_two_rp'], 'X')

  def testPartiallySubstituteRuntimeParameter(self):
    pipeline = pipeline_pb2.Pipeline()
    expected = pipeline_pb2.Pipeline()
    self.load_proto_from_text(
        os.path.join(self._testdata_dir,
                     'pipeline_with_runtime_parameter.pbtxt'), pipeline)
    self.load_proto_from_text(
        os.path.join(
            self._testdata_dir,
            'pipeline_with_runtime_parameter_partially_substituted.pbtxt'),
        expected)
    parameters = runtime_parameter_utils.substitute_runtime_parameter(
        pipeline, {
            'context_name_rp': 'my_context',
        })
    self.assertProtoEquals(pipeline, expected)
    self.assertEqual(len(parameters), 3)
    self.assertEqual(parameters['context_name_rp'], 'my_context')
    self.assertEqual(parameters['prop_one_rp'], 1)
    self.assertIsNone(parameters['prop_two_rp'])

  def testSubstituteRuntimeParameterFail(self):
    pipeline = pipeline_pb2.Pipeline()
    self.load_proto_from_text(
        os.path.join(self._testdata_dir,
                     'pipeline_with_runtime_parameter.pbtxt'), pipeline)
    with self.assertRaisesRegex(RuntimeError, 'Runtime parameter type'):
      runtime_parameter_utils.substitute_runtime_parameter(
          pipeline,
          {
              'context_name_rp': 0,  # Wrong type, will lead to failure.
              'prop_one_rp': 2,
              'prop_two_rp': 'X'
          })


if __name__ == '__main__':
  tf.test.main()
