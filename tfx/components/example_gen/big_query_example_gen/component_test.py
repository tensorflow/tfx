# Lint as: python2, python3
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
"""Tests for tfx.components.example_gen.big_query_example_gen.component."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx.components.example_gen.big_query_example_gen import component
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class ComponentTest(tf.test.TestCase):

  def testConstruct(self):
    big_query_example_gen = component.BigQueryExampleGen(query='query')
    self.assertEqual(standard_artifacts.Examples.TYPE_NAME,
                     big_query_example_gen.outputs['examples'].type_name)
    artifact_collection = big_query_example_gen.outputs['examples'].get()
    self.assertEqual(1, len(artifact_collection))
    self.assertEqual(['train', 'eval'],
                     artifact_utils.decode_split_names(
                         artifact_collection[0].split_names))

  def testConstructWithOutputConfig(self):
    big_query_example_gen = component.BigQueryExampleGen(
        query='query',
        output_config=example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=2),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1),
                example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=1)
            ])))
    self.assertEqual(standard_artifacts.Examples.TYPE_NAME,
                     big_query_example_gen.outputs['examples'].type_name)
    artifact_collection = big_query_example_gen.outputs['examples'].get()
    self.assertEqual(1, len(artifact_collection))
    self.assertEqual(['train', 'eval', 'test'],
                     artifact_utils.decode_split_names(
                         artifact_collection[0].split_names))

  def testConstructWithInputConfig(self):
    big_query_example_gen = component.BigQueryExampleGen(
        input_config=example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='train', pattern='query1'),
            example_gen_pb2.Input.Split(name='eval', pattern='query2'),
            example_gen_pb2.Input.Split(name='test', pattern='query3')
        ]))
    self.assertEqual(standard_artifacts.Examples.TYPE_NAME,
                     big_query_example_gen.outputs['examples'].type_name)
    artifact_collection = big_query_example_gen.outputs['examples'].get()
    self.assertEqual(1, len(artifact_collection))
    self.assertEqual(['train', 'eval', 'test'],
                     artifact_utils.decode_split_names(
                         artifact_collection[0].split_names))


if __name__ == '__main__':
  tf.test.main()
