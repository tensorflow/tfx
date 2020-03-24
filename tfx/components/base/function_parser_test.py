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
"""Tests for tfx.components.base.function_parser."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Text
import unittest

# Standard Imports

import six
import tensorflow as tf

from tfx.components.base.function_parser import ArgFormats
from tfx.components.base.function_parser import parse_typehint_component_function
from tfx.types import standard_artifacts
from tfx.types.annotations import ComponentOutput
from tfx.types.annotations import InputArtifact
from tfx.types.annotations import InputArtifactUri
from tfx.types.annotations import OutputArtifact
from tfx.types.annotations import OutputArtifactUri


@unittest.skipIf(six.PY2, 'Not compatible with Python 2.')
class FunctionParserTest(tf.test.TestCase):

  def testSimpleFunctionParse(self):

    def func_a(a: int, b: int, unused_c: Text,
               unused_d: bytes) -> ComponentOutput(c=float):
      return {'c': float(a + b)}

    inputs, outputs, arg_formats, returned_values = (
        parse_typehint_component_function(func_a))
    self.assertEqual(
        inputs, {
            'a': standard_artifacts.IntegerType,
            'b': standard_artifacts.IntegerType,
            'unused_c': standard_artifacts.StringType,
            'unused_d': standard_artifacts.BytesType,
        })
    self.assertEqual(outputs, {
        'c': standard_artifacts.FloatType,
    })
    self.assertEqual(arg_formats, [
        (ArgFormats.ARTIFACT_VALUE, 'a'),
        (ArgFormats.ARTIFACT_VALUE, 'b'),
        (ArgFormats.ARTIFACT_VALUE, 'unused_c'),
        (ArgFormats.ARTIFACT_VALUE, 'unused_d'),
    ])
    self.assertEqual(returned_values, set(['c']))

  def testArtifactFunctionParse(self):
    # pylint: disable=unused-argument
    def func_a(
        examples: InputArtifact[standard_artifacts.Examples],
        model: OutputArtifact[standard_artifacts.Model],
        schema_uri: InputArtifactUri[standard_artifacts.Schema],
        statistics_uri: OutputArtifactUri[standard_artifacts.ExampleStatistics],
        num_steps: int
    ) -> ComponentOutput(
        precision=float, recall=float, message=Text, serialized_value=bytes):
      return {
          'precision': 0.9,
          'recall': 0.8,
          'message': 'foo',
          'serialized_value': b'bar'
      }

    # pylint: enable=unused-argument
    inputs, outputs, arg_formats, returned_values = (
        parse_typehint_component_function(func_a))
    self.assertEqual(
        inputs, {
            'examples': standard_artifacts.Examples,
            'schema_uri': standard_artifacts.Schema,
            'num_steps': standard_artifacts.IntegerType,
        })
    self.assertEqual(
        outputs, {
            'model': standard_artifacts.Model,
            'statistics_uri': standard_artifacts.ExampleStatistics,
            'precision': standard_artifacts.FloatType,
            'recall': standard_artifacts.FloatType,
            'message': standard_artifacts.StringType,
            'serialized_value': standard_artifacts.BytesType,
        })
    self.assertEqual(arg_formats, [
        (ArgFormats.INPUT_ARTIFACT, 'examples'),
        (ArgFormats.OUTPUT_ARTIFACT, 'model'),
        (ArgFormats.INPUT_ARTIFACT_URI, 'schema_uri'),
        (ArgFormats.OUTPUT_ARTIFACT_URI, 'statistics_uri'),
        (ArgFormats.ARTIFACT_VALUE, 'num_steps'),
    ])
    self.assertEqual(
        returned_values,
        set(['precision', 'recall', 'message', 'serialized_value']))

  def testFunctionParseErrors(self):
    # Non-function arguments.
    with self.assertRaisesRegexp(
        ValueError, 'Expected a typehint-annotated Python function'):
      parse_typehint_component_function(object())
    with self.assertRaisesRegexp(
        ValueError, 'Expected a typehint-annotated Python function'):
      parse_typehint_component_function('foo')

    # Unannotated lambda.
    with self.assertRaisesRegexp(
        ValueError,
        'must have a ComponentOutput instance as its return value typehint'):
      parse_typehint_component_function(lambda x: True)

    # Function with *args and **kwargs.
    with self.assertRaisesRegexp(
        ValueError, r'does not support \*args or \*\*kwargs arguments'):

      def func_a(a: int, b: int, *unused_args) -> ComponentOutput(c=float):
        return float(a + b)

      parse_typehint_component_function(func_a)
    with self.assertRaisesRegexp(
        ValueError, r'does not support \*args or \*\*kwargs arguments'):

      def func_b(a: int, b: int, **unused_kwargs) -> ComponentOutput(c=float):
        return float(a + b)

      parse_typehint_component_function(func_b)

    # Default arguments are not supported yet.
    # TODO(ccy): add support for default arguments.
    with self.assertRaisesRegexp(
        ValueError, 'currently does not support optional arguments'):

      def func_c(a: int, b: int = 1) -> ComponentOutput(c=float):
        return float(a + b)

      parse_typehint_component_function(func_c)

    # Not all arguments annotated with typehints.
    with self.assertRaisesRegexp(
        ValueError, 'must have all arguments annotated with typehint'):

      def func_d(a: int, b) -> ComponentOutput(c=float):
        return float(a + b)

      parse_typehint_component_function(func_d)

    # Invalid input typehint.
    with self.assertRaisesRegexp(ValueError, 'Unknown type hint annotation'):

      def func_e(a: int, b: Dict[int, int]) -> ComponentOutput(c=float):
        return float(a + b)

      parse_typehint_component_function(func_e)

    # Invalid output typehint.
    with self.assertRaisesRegexp(ValueError, 'Unknown type hint annotation'):

      def func_f(a: int, b: int) -> ComponentOutput(c='whatever'):
        return float(a + b)

      parse_typehint_component_function(func_f)

    # Output artifact in the wrong place.
    with self.assertRaisesRegexp(
        ValueError,
        'Output artifacts .* should be declared as function parameters'):

      def func_g(a: int,
                 b: int) -> ComponentOutput(c=standard_artifacts.Examples):
        return float(a + b)

      parse_typehint_component_function(func_g)
    with self.assertRaisesRegexp(
        ValueError,
        'Output artifacts .* should be declared as function parameters'):

      def func_h(
          a: int, b: int
      ) -> ComponentOutput(c=OutputArtifact[standard_artifacts.Examples]):
        return float(a + b)

      parse_typehint_component_function(func_h)


if __name__ == '__main__':
  tf.test.main()
