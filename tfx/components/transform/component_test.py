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

import os
import tensorflow as tf
from tfx.components.transform import component
from tfx.utils import channel
from tfx.utils import types


class TransformTest(tf.test.TestCase):

  def test_construct(self):
    source_data_dir = os.path.join(
        os.path.dirname(__file__), 'testdata', 'taxi')
    preprocessing_fn_file = os.path.join(source_data_dir, 'module',
                                         'preprocess.py')
    transform = component.Transform(
        input_data=channel.as_channel([
            types.TfxType(type_name='ExamplesPath', split='train'),
            types.TfxType(type_name='ExamplesPath', split='eval'),
        ]),
        schema=channel.as_channel([types.TfxType(type_name='SchemaPath')]),
        module_file=preprocessing_fn_file,
    )
    self.assertEqual('TransformPath',
                     transform.outputs.transform_output.type_name)
    self.assertEqual('ExamplesPath',
                     transform.outputs.transformed_examples.type_name)


if __name__ == '__main__':
  tf.test.main()
