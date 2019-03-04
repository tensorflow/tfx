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
"""Tests for tfx.components.model_validator.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx.components.model_validator import component
from tfx.utils import channel
from tfx.utils import types


class ComponentTest(tf.test.TestCase):

  def test_construct(self):
    examples = types.TfxType(type_name='ExamplesPath')
    model = types.TfxType(type_name='ModelExportPath')
    model_validator = component.ModelValidator(
        examples=channel.as_channel([examples]),
        model=channel.as_channel([model]))
    self.assertEqual('ModelBlessingPath',
                     model_validator.outputs.blessing.type_name)
    self.assertEqual('ModelValidationPath',
                     model_validator.outputs.results.type_name)


if __name__ == '__main__':
  tf.test.main()
