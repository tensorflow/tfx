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
"""Tests for tfx.components.data_view.binder_executor."""
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tfx.components.experimental.data_view import binder_executor
from tfx.components.experimental.data_view import constants
from tfx.types import standard_artifacts


class BinderExecutorTest(tf.test.TestCase):

  def testDo(self):
    data_view = standard_artifacts.DataView()
    data_view.uri = '/old/data_view'
    data_view.id = 1
    data_view.mlmd_artifact.create_time_since_epoch = 123

    existing_custom_property = 'payload_format'
    input_examples = standard_artifacts.Examples()
    input_examples.uri = '/examples/1'
    input_examples.set_string_custom_property(
        existing_custom_property, 'VALUE1')

    input_dict = {
        'input_examples': [input_examples],
        'data_view': [data_view],
    }
    exec_properties = {}
    output_dict = {
        'output_examples': [
            standard_artifacts.Examples()
        ]
    }

    executor = binder_executor.DataViewBinderExecutor()
    executor.Do(input_dict, output_dict, exec_properties)

    output_examples = output_dict.get('output_examples')
    self.assertIsNotNone(output_examples)
    self.assertLen(output_examples, 1)
    oe = output_examples[0]
    self.assertEqual(
        oe.get_string_custom_property(
            constants.DATA_VIEW_URI_PROPERTY_KEY), '/old/data_view')
    self.assertEqual(
        oe.get_string_custom_property(
            constants.DATA_VIEW_CREATE_TIME_KEY), '123')

    # output should share the URI with the input.
    self.assertEqual(oe.uri, input_examples.uri)
    # other custom properties should be inherited.
    self.assertEqual(
        oe.get_string_custom_property(existing_custom_property),
        input_examples.get_string_custom_property(existing_custom_property))


if __name__ == '__main__':
  tf.test.main()
