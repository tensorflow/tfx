# Copyright 2020 Google LLC. All Rights Reserved.
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
"""Tests for tfx.components.util.examples_utils."""

import tensorflow as tf
from tfx.components.example_gen import utils
from tfx.components.util import examples_utils
from tfx.proto import example_gen_pb2
from tfx.types import standard_artifacts


class ExamplesUtilsTest(tf.test.TestCase):

  def test_get_payload_format(self):
    examples = standard_artifacts.Examples()
    self.assertEqual(examples_utils.get_payload_format(examples),
                     example_gen_pb2.PayloadFormat.FORMAT_TF_EXAMPLE)
    self.assertEqual(examples_utils.get_payload_format_string(examples),
                     'FORMAT_TF_EXAMPLE')

    examples.set_string_custom_property(utils.PAYLOAD_FORMAT_PROPERTY_NAME,
                                        'FORMAT_PROTO')
    self.assertEqual(examples_utils.get_payload_format(examples),
                     example_gen_pb2.PayloadFormat.FORMAT_PROTO)
    self.assertEqual(examples_utils.get_payload_format_string(examples),
                     'FORMAT_PROTO')

  def test_get_split_pattern(self):
    examples = standard_artifacts.Examples()
    examples.uri = '/test/uri'
    examples.split_names = '["train"]'
    self.assertEqual(
        examples_utils.get_split_file_patterns([examples], 'train'),
        ['/test/uri/Split-train/*'],
    )
    with self.assertRaises(ValueError):
      examples_utils.get_split_file_patterns([examples], 'missing_split')

  def test_get_split_pattern_with_custom_pattern(self):
    examples1 = standard_artifacts.Examples()
    examples1.uri = '/test/uri'
    examples1.split_names = '["train"]'

    examples2 = standard_artifacts.Examples()
    examples2.uri = '/test/uri'
    k, v = examples_utils.get_custom_split_patterns_key_and_property(
        {'train': 'subdir/train-*-of-*'}
    )
    examples2.set_string_custom_property(k, v)

    self.assertEqual(
        ['/test/uri/Split-train/*', '/test/uri/subdir/train-*-of-*'],
        examples_utils.get_split_file_patterns([examples1, examples2], 'train'),
    )
    with self.assertRaises(ValueError):
      examples_utils.get_split_file_patterns(
          [examples1, examples2], 'missing_split'
      )

  def test_get_payload_format_invalid_artifact_type(self):
    artifact = standard_artifacts.Schema()
    with self.assertRaises(AssertionError):
      examples_utils.get_payload_format(artifact)

  def test_set_payload_format(self):
    examples = standard_artifacts.Examples()
    examples_utils.set_payload_format(
        examples, example_gen_pb2.PayloadFormat.FORMAT_PROTO)
    self.assertEqual(
        examples.get_string_custom_property(utils.PAYLOAD_FORMAT_PROPERTY_NAME),
        'FORMAT_PROTO')

  def test_set_payload_format_invalid_artifact_type(self):
    artifact = standard_artifacts.Schema()
    with self.assertRaises(AssertionError):
      examples_utils.set_payload_format(
          artifact, example_gen_pb2.PayloadFormat.FORMAT_PROTO)
