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

import os
import re
from typing import Text
# Standard Imports

import tensorflow as tf

from tfx.components.example_gen import utils
from tfx.orchestration import data_types
from tfx.proto import example_gen_pb2
from tfx.utils import io_utils
from tfx.utils import json_utils


class UtilsTest(tf.test.TestCase):

  def setUp(self):
    super(UtilsTest, self).setUp()
    # Create input splits.
    test_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    self._input_base_path = os.path.join(test_dir, 'input_base')
    tf.io.gfile.makedirs(self._input_base_path)

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

  def testGlobToRegex(self):
    glob_pattern = 'a(b)c'
    self.assertEqual(1, re.compile(glob_pattern).groups)
    regex_pattern = utils._glob_to_regex(glob_pattern)
    self.assertEqual(0, re.compile(regex_pattern).groups)
    self.assertEqual(glob_pattern,
                     re.match(regex_pattern, glob_pattern).group())

  def testCalculateSplitsFingerprint(self):
    split1 = os.path.join(self._input_base_path, 'split1', 'data')
    io_utils.write_string_file(split1, 'testing')
    os.utime(split1, (0, 1))
    split2 = os.path.join(self._input_base_path, 'split2', 'data')
    io_utils.write_string_file(split2, 'testing2')
    os.utime(split2, (0, 3))

    splits = [
        example_gen_pb2.Input.Split(name='s1', pattern='split1/*'),
        example_gen_pb2.Input.Split(name='s2', pattern='split2/*')
    ]
    fingerprint, span = utils.calculate_splits_fingerprint_and_span(
        self._input_base_path, splits)
    self.assertEqual(
        fingerprint,
        'split:s1,num_files:1,total_bytes:7,xor_checksum:1,sum_checksum:1\n'
        'split:s2,num_files:1,total_bytes:8,xor_checksum:3,sum_checksum:3')
    self.assertEqual(span, '0')

  def testSpanNoMatching(self):
    splits = [
        example_gen_pb2.Input.Split(name='s1', pattern='span{SPAN}/split1/*'),
        example_gen_pb2.Input.Split(name='s2', pattern='span{SPAN}/split2/*')
    ]
    with self.assertRaisesRegexp(ValueError,
                                 'Cannot not find matching for split'):
      utils.calculate_splits_fingerprint_and_span(self._input_base_path, splits)

  def testSpanWrongFormat(self):
    wrong_span = os.path.join(self._input_base_path, 'spanx', 'split1', 'data')
    io_utils.write_string_file(wrong_span, 'testing_wrong_span')

    splits = [
        example_gen_pb2.Input.Split(name='s1', pattern='span{SPAN}/split1/*'),
        example_gen_pb2.Input.Split(name='s2', pattern='span{SPAN}/split2/*')
    ]
    with self.assertRaisesRegexp(ValueError, 'Cannot not find span number'):
      utils.calculate_splits_fingerprint_and_span(self._input_base_path, splits)

  def testSpanMatches(self):
    # Test align of span number.
    span1_split1 = os.path.join(self._input_base_path, 'span01', 'split1',
                                'data')
    io_utils.write_string_file(span1_split1, 'testing11')
    span1_split2 = os.path.join(self._input_base_path, 'span01', 'split2',
                                'data')
    io_utils.write_string_file(span1_split2, 'testing12')
    span2_split1 = os.path.join(self._input_base_path, 'span02', 'split1',
                                'data')
    io_utils.write_string_file(span2_split1, 'testing21')

    splits = [
        example_gen_pb2.Input.Split(name='s1', pattern='span{SPAN}/split1/*'),
        example_gen_pb2.Input.Split(name='s2', pattern='span{SPAN}/split2/*')
    ]
    with self.assertRaisesRegexp(
        ValueError, 'Latest span should be the same for each split'):
      utils.calculate_splits_fingerprint_and_span(self._input_base_path, splits)

    # Test if latest span is selected when span aligns for each split.
    span2_split2 = os.path.join(self._input_base_path, 'span02', 'split2',
                                'data')
    io_utils.write_string_file(span2_split2, 'testing22')

    splits = [
        example_gen_pb2.Input.Split(name='s1', pattern='span{SPAN}/split1/*'),
        example_gen_pb2.Input.Split(name='s2', pattern='span{SPAN}/split2/*')
    ]
    _, span = utils.calculate_splits_fingerprint_and_span(
        self._input_base_path, splits)
    self.assertEqual(span, '02')


if __name__ == '__main__':
  tf.test.main()
