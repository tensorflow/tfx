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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx.components.example_gen.csv_example_gen import component
from tfx.utils import channel
from tfx.utils import types


class ComponentTest(tf.test.TestCase):

  def test_construct(self):
    input_base = types.TfxArtifact(type_name='ExternalPath')
    csv_example_gen = component.CsvExampleGen(
        input_base=channel.as_channel([input_base]))
    self.assertEqual('ExamplesPath',
                     csv_example_gen.outputs.examples.type_name)
    artifact_collection = csv_example_gen.outputs.examples.get()
    self.assertEqual('train', artifact_collection[0].split)
    self.assertEqual('eval', artifact_collection[1].split)


if __name__ == '__main__':
  tf.test.main()
