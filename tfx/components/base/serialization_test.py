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
"""Tests for tfx.components.base.serialization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Text
import unittest

# Standard Imports

import six
import tensorflow as tf

from tfx.components.base.serialization import SourceCopySerializer
from tfx.types import standard_artifacts
from tfx.types.annotations import ComponentOutput
from tfx.types.annotations import InputArtifact
from tfx.types.annotations import OutputArtifact


def _my_decorator(value):
  return value


@unittest.skipIf(six.PY2, 'Not compatible with Python 2.')
class SerializationTest(tf.test.TestCase):

  def testSourceCopySerializer(self):
    # pylint: disable=unused-argument
    @_my_decorator
    def func_a(
        # A comment.
        examples: InputArtifact[standard_artifacts.Examples],
        model: OutputArtifact[standard_artifacts.Model],
        schema_uri: InputArtifact[standard_artifacts.Schema],
        statistics_uri: OutputArtifact[standard_artifacts.ExampleStatistics],
        num_steps: int
    ) -> ComponentOutput(
        precision=float, recall=float, message=Text, serialized_value=bytes):
      """My docstring."""
      return {
          'precision': 0.9,
          'recall': 0.8,
          'message': 'foo',
          'serialized_value': b'bar'
      }

    # pylint: enable=unused-argument

    expected_value = '''\
def func_a(examples, model, schema_uri, statistics_uri, num_steps):
  """My docstring."""
  return {
      'precision': 0.9,
      'recall': 0.8,
      'message': 'foo',
      'serialized_value': b'bar'
  }
'''
    print(repr(SourceCopySerializer.encode(func_a)))

    self.assertEqual(SourceCopySerializer.encode(func_a), expected_value)


if __name__ == '__main__':
  tf.test.main()
