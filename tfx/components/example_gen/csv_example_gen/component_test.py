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
"""Tests for tfx.components.example_gen.csv_example_gen.component."""

import tensorflow as tf
from tfx.components.example_gen.csv_example_gen import component
from tfx.types import standard_artifacts


class ComponentTest(tf.test.TestCase):

  def testConstruct(self):
    csv_example_gen = component.CsvExampleGen(input_base='path')
    self.assertEqual(standard_artifacts.Examples.TYPE_NAME,
                     csv_example_gen.outputs['examples'].type_name)


if __name__ == '__main__':
  tf.test.main()
