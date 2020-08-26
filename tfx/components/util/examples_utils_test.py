# Lint as: python2, python3
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

# TODO(b/149535307): Remove __future__ imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


if __name__ == '__main__':
  tf.test.main()
