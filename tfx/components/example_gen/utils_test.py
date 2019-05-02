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
"""Tests for tfx.components.example_gen.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports

import tensorflow as tf
from tfx.components.example_gen import utils
from tfx.proto import example_gen_pb2


class UtilsTest(tf.test.TestCase):

  def test_make_output_split_names(self):
    split_names = utils.generate_output_split_names(
        input_config=example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='train', pattern='train/*'),
            example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
        ]),
        output_config=example_gen_pb2.Output())
    self.assertListEqual(['train', 'eval'], split_names)

    split_names = utils.generate_output_split_names(
        input_config=example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='single', pattern='single/*')
        ]),
        output_config=example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=2),
                example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=1)
            ])))
    self.assertListEqual(['train', 'eval'], split_names)

  def test_make_default_output_config(self):
    output_config = utils.make_default_output_config(
        utils.make_default_input_config())
    self.assertEqual(2, len(output_config.split_config.splits))

    output_config = utils.make_default_output_config(
        example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='train', pattern='train/*'),
            example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
        ]))
    self.assertEqual(0, len(output_config.split_config.splits))


if __name__ == '__main__':
  tf.test.main()
