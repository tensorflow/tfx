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
"""Tests for tfx.components.schema_gen.import_schema_gen.executor."""

import os

import tensorflow as tf
from tfx.components.schema_gen import executor as schema_gen_executor
from tfx.components.schema_gen.import_schema_gen import executor as _executor
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import io_utils
from tfx.utils import test_case_utils
from google.protobuf import text_format


class ExecutorTest(test_case_utils.TfxTest):

  def setUp(self):
    super().setUp()
    output_artifact = standard_artifacts.Schema()
    output_artifact.uri = self.tmp_dir
    self.output_dict = {standard_component_specs.SCHEMA_KEY: [output_artifact]}

    self.source_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'testdata',
        'schema_gen', 'schema.pbtxt')
    self.exec_properties = {
        standard_component_specs.SCHEMA_FILE_KEY: self.source_file_path
    }

  def testMissingSchemaFile(self):
    executor = _executor.Executor()

    with self.assertRaisesRegex(ValueError, 'Schema file path is missing'):
      executor.Do({}, self.output_dict, {})

    # Underlying filesystem may emit different errors for files not found.
    with self.assertRaises((tf.errors.NotFoundError, IOError)):
      executor.Do({}, self.output_dict,
                  {standard_component_specs.SCHEMA_FILE_KEY: 'invalid_path'})

    # If the given file is not a proper schema textproto. For example, Python?
    with self.assertRaises(text_format.ParseError):
      executor.Do({}, self.output_dict,
                  {standard_component_specs.SCHEMA_FILE_KEY: __file__})

  def testInvalidOutput(self):
    with self.assertRaises(KeyError):
      _executor.Executor().Do({}, {}, self.exec_properties)
    with self.assertRaisesRegex(ValueError, 'expected list length of one'):
      _executor.Executor().Do({}, {
          standard_component_specs.SCHEMA_KEY:
              [standard_artifacts.Schema(),
               standard_artifacts.Schema()]
      }, self.exec_properties)

  def testSuccess(self):
    _executor.Executor().Do({}, self.output_dict, self.exec_properties)
    reader = io_utils.SchemaReader()
    expected_proto = reader.read(self.source_file_path)
    imported_proto = reader.read(
        os.path.join(self.tmp_dir, schema_gen_executor.DEFAULT_FILE_NAME))
    self.assertEqual(expected_proto, imported_proto)


if __name__ == '__main__':
  tf.test.main()
