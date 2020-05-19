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
"""Tests for tfx.components.example_gen.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Text
# Standard Imports

import tensorflow as tf

from tfx.components.example_gen import utils
from tfx.orchestration import data_types
from tfx.proto import example_gen_pb2
from tfx.utils import json_utils


class UtilsTest(tf.test.TestCase):

  def testDictToExample(self):
    instance_dict = {
        'int': 10,
        'float': 5.0,
        'str': 'abc',
        'int_list': [1, 2],
        'float_list': [3.0],
        'str_list': ['ab', 'cd'],
        'none': None,
        'empty_list': [],
    }
    example = utils.dict_to_example(instance_dict)
    self.assertProtoEquals(
        """
        features {
          feature {
            key: "empty_list"
            value {
            }
          }
          feature {
            key: "float"
            value {
              float_list {
                value: 5.0
              }
            }
          }
          feature {
            key: "float_list"
            value {
              float_list {
                value: 3.0
              }
            }
          }
          feature {
            key: "int"
            value {
              int64_list {
                value: 10
              }
            }
          }
          feature {
            key: "int_list"
            value {
              int64_list {
                value: 1
                value: 2
              }
            }
          }
          feature {
            key: "none"
            value {
            }
          }
          feature {
            key: "str"
            value {
              bytes_list {
                value: "abc"
              }
            }
          }
          feature {
            key: "str_list"
            value {
              bytes_list {
                value: "ab"
                value: "cd"
              }
            }
          }
        }
        """, example)

  def testMakeOutputSplitNames(self):
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

  def testMakeDefaultOutputConfig(self):
    output_config = utils.make_default_output_config(
        utils.make_default_input_config())
    self.assertEqual(2, len(output_config.split_config.splits))

    output_config = utils.make_default_output_config(
        example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='train', pattern='train/*'),
            example_gen_pb2.Input.Split(name='eval', pattern='eval/*')
        ]))
    self.assertEqual(0, len(output_config.split_config.splits))

  def testMakeOutputSplitNamesWithParameter(self):
    split_name_param = data_types.RuntimeParameter(
        name='split-name', ptype=Text, default=u'train')
    split_names = utils.generate_output_split_names(
        input_config={
            'splits': [{
                'name': split_name_param,
                'pattern': 'train/*'
            }, {
                'name': 'eval',
                'pattern': 'eval/*'
            }]
        },
        output_config=example_gen_pb2.Output())
    # Assert the json serialized version because RuntimeParameters only get
    # serialized after that.
    self.assertEqual(
        json_utils.dumps([split_name_param, 'eval']),
        json_utils.dumps(split_names))

    split_names = utils.generate_output_split_names(
        input_config=example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='single', pattern='single/*')
        ]),
        output_config={
            'split_config': {
                'splits': [{
                    'name': split_name_param,
                    'hash_buckets': 2
                }, {
                    'name': 'eval',
                    'hash_buckets': 1
                }]
            }
        })
    # Assert the json serialized version because RuntimeParameters only get
    # serialized after that.
    self.assertEqual(
        json_utils.dumps([split_name_param, 'eval']),
        json_utils.dumps(split_names))

  def testMakeDefaultOutputConfigWithParameter(self):
    split_name_param = data_types.RuntimeParameter(
        name='split-name', ptype=Text, default=u'train')
    output_config = utils.make_default_output_config({
        'splits': [{
            'name': split_name_param,
            'pattern': 'train/*'
        }, {
            'name': 'eval',
            'pattern': 'eval/*'
        }]
    })
    self.assertEqual(0, len(output_config.split_config.splits))


if __name__ == '__main__':
  tf.test.main()
