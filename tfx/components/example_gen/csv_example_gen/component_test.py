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
from tfx.proto import example_gen_pb2
from tfx.utils import channel
from tfx.utils import types


class ComponentTest(tf.test.TestCase):

  def test_construct(self):
    input_base = types.TfxType(type_name='ExternalPath')
    csv_example_gen = component.CsvExampleGen(
        input_base=channel.as_channel([input_base]))
    self.assertEqual('ExamplesPath', csv_example_gen.outputs.examples.type_name)
    artifact_collection = csv_example_gen.outputs.examples.get()
    self.assertEqual('train', artifact_collection[0].split)
    self.assertEqual('eval', artifact_collection[1].split)

  def test_construct_with_output_config(self):
    input_base = types.TfxType(type_name='ExternalPath')
    csv_example_gen = component.CsvExampleGen(
        input_base=channel.as_channel([input_base]),
        output_config=example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=2),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1),
                example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=1)
            ])))
    self.assertEqual('ExamplesPath', csv_example_gen.outputs.examples.type_name)
    artifact_collection = csv_example_gen.outputs.examples.get()
    self.assertEqual('train', artifact_collection[0].split)
    self.assertEqual('eval', artifact_collection[1].split)
    self.assertEqual('test', artifact_collection[2].split)


if __name__ == '__main__':
  tf.test.main()
