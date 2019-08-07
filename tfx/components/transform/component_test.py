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
"""Tests for tfx.components.transform.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx import types
from tfx.components.transform import component
from tfx.types import channel_utils
from tfx.types import standard_artifacts


class ComponentTest(tf.test.TestCase):

  def setUp(self):
    super(ComponentTest, self).setUp()
    self.input_data = channel_utils.as_channel([
        standard_artifacts.Examples(split='train'),
        standard_artifacts.Examples(split='eval'),
    ])
    self.schema = channel_utils.as_channel(
        [types.Artifact(type_name='SchemaPath')])

  def _verify_outputs(self, transform):
    self.assertEqual('TransformPath',
                     transform.outputs.transform_output.type_name)
    self.assertEqual('ExamplesPath',
                     transform.outputs.transformed_examples.type_name)

  def test_construct_from_module_file(self):
    module_file = '/path/to/preprocessing.py'
    transform = component.Transform(
        input_data=self.input_data,
        schema=self.schema,
        module_file=module_file,
    )
    self._verify_outputs(transform)
    self.assertEqual(module_file, transform.spec.exec_properties['module_file'])

  def test_construct_from_preprocessing_fn(self):
    preprocessing_fn = 'path.to.my_preprocessing_fn'
    transform = component.Transform(
        input_data=self.input_data,
        schema=self.schema,
        preprocessing_fn=preprocessing_fn,
    )
    self._verify_outputs(transform)
    self.assertEqual(preprocessing_fn,
                     transform.spec.exec_properties['preprocessing_fn'])

  def test_construct_missing_user_module(self):
    with self.assertRaises(ValueError):
      _ = component.Transform(
          input_data=self.input_data,
          schema=self.schema,
      )

  def test_construct_duplicate_user_module(self):
    with self.assertRaises(ValueError):
      _ = component.Transform(
          input_data=self.input_data,
          schema=self.schema,
          module_file='/path/to/preprocessing.py',
          preprocessing_fn='path.to.my_preprocessing_fn',
      )


if __name__ == '__main__':
  tf.test.main()
