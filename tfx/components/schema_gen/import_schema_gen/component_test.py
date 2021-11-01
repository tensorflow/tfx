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
"""Tests for tfx.components.schema_gen.import_schema_gen."""

import tensorflow as tf
from tfx.components.schema_gen.import_schema_gen import component

from tfx.types import standard_artifacts
from tfx.types import standard_component_specs


class SchemaGenTest(tf.test.TestCase):

  def testConstruct(self):
    schema_gen = component.ImportSchemaGen('dummy')
    self.assertEqual(
        standard_artifacts.Schema.TYPE_NAME,
        schema_gen.outputs[standard_component_specs.SCHEMA_KEY].type_name)
    self.assertEqual(
        schema_gen.spec.exec_properties[
            standard_component_specs.SCHEMA_FILE_KEY], 'dummy')


if __name__ == '__main__':
  tf.test.main()
