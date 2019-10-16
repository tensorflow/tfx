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
"""Tests for tfx.components.schema_gen.executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tfx.components.schema_gen import executor
from tfx.types import standard_artifacts
from tfx.utils import io_utils


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()

    self.source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')

    self.train_stats_artifact = standard_artifacts.ExampleStatistics(
        split='train')
    self.train_stats_artifact.uri = os.path.join(self.source_data_dir,
                                                 'statistics_gen/train/')

    self.output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)

    self.schema_output = standard_artifacts.Schema()
    self.schema_output.uri = os.path.join(self.output_data_dir, 'schema_output')

    self.schema = standard_artifacts.Schema()
    self.schema.uri = os.path.join(self.source_data_dir, 'fixed_schema/')

    self.expected_schema = standard_artifacts.Schema()
    self.expected_schema.uri = os.path.join(self.source_data_dir, 'schema_gen/')

    self.input_dict = {
        'stats': [self.train_stats_artifact],
        'schema': None
    }
    self.output_dict = {
        'output': [self.schema_output],
    }
    self.exec_properties = {'infer_feature_shape': False}

  def _assertSchemaEqual(self, expected_schema, actual_schema):
    schema_reader = io_utils.SchemaReader()
    expected_schema_proto = schema_reader.read(
        os.path.join(expected_schema.uri, executor._DEFAULT_FILE_NAME))
    actual_schema_proto = schema_reader.read(
        os.path.join(actual_schema.uri, executor._DEFAULT_FILE_NAME))
    self.assertProtoEquals(expected_schema_proto, actual_schema_proto)

  def testDoWithStatistics(self):
    schema_gen_executor = executor.Executor()
    schema_gen_executor.Do(self.input_dict, self.output_dict,
                           self.exec_properties)
    self.assertNotEqual(0, len(tf.io.gfile.listdir(self.schema_output.uri)))
    self._assertSchemaEqual(self.expected_schema, self.schema_output)

  def testDoWithSchema(self):
    self.input_dict['schema'] = [self.schema]
    self.input_dict.pop('stats')
    schema_gen_executor = executor.Executor()
    schema_gen_executor.Do(self.input_dict, self.output_dict,
                           self.exec_properties)
    self.assertNotEqual(0, len(tf.io.gfile.listdir(self.schema_output.uri)))
    self._assertSchemaEqual(self.schema, self.schema_output)

  def testDoWithNonExistentSchema(self):
    non_existent_schema = standard_artifacts.Schema()
    non_existent_schema.uri = '/path/to/non_existent/schema'

    self.input_dict['schema'] = [non_existent_schema]
    self.input_dict.pop('stats')

    with self.assertRaises(ValueError):
      schema_gen_executor = executor.Executor()
      schema_gen_executor.Do(self.input_dict, self.output_dict,
                             self.exec_properties)


if __name__ == '__main__':
  tf.test.main()
